"""The spend gate: each refused case is a way money was spent by mistake. The launch check is tested as the driver runs it,
with ssh stubbed; nothing here contacts a pod. The commit gate is aeftimia/commit-gate's, tested in that repository.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from rewardhack import gates

PROJECT = Path(__file__).resolve().parent.parent
GATE = ['bash', str(PROJECT / 'ops' / 'gate.sh')]
PASS, FAIL = [sys.executable, '-c', 'pass'], [sys.executable, '-c', 'raise SystemExit("3 failed")']
DRIVER = 'bash ops/drive_runpod.sh 1.2.3.4 22 ops/remote_runpod_smoke.sh results/x 60'
TERM = 'ZEBRA'


def run(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)


def new_repo(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    run(root, 'init', '-q', '-b', 'main')
    run(root, 'config', 'user.email', 'test@example.com')
    run(root, 'config', 'user.name', 'Test')
    run(root, 'add', '-A')
    run(root, 'commit', '-q', '-m', 'start')
    return root


@pytest.fixture
def repo(tmp_path):
    return new_repo(tmp_path / 'repo', {
        'ops/remote_runpod_smoke.sh': '# deadline: 5\necho job\n',
        'ops/drive_runpod.sh':  # the real driver's check-only mode, in miniature
            "[[ ${LEAN_ISOLATION:-} != required ]] || { echo 'LEAN_ISOLATION=required cannot work on this pod'; exit 2; }\n"
            "[[ ${DRIVE_CHECK_ONLY:-} == 1 ]] && { echo 'check passed'; exit 0; }\nexit 9\n",
        'ops/pod_pins.env': (PROJECT / 'ops' / 'pod_pins.env').read_text(),
        'src/.keep': '',
        'spending/ledger.csv': (PROJECT / 'spending' / 'ledger.csv').read_text().splitlines()[0] + '\n',
        '.claude/commit-gate-terms.txt': f'# what stays out\n\\b{TERM}\\b\n',
        '.gitignore': '.claude/\n.tools/\n',
    })


def settle(repo, deadline=60, basis='2x measured 30 min', tests=PASS, remote='ops/remote_runpod_smoke.sh', **kw):
    return gates.ready(repo, remote, deadline, basis, 'gpt-oss-20b-gpu', 2.70, kw.get('other_usd', 0.0), kw.get('launch_cap'),
                       kw.get('settings', {}), tests)


def problems(repo, command):
    return gates.command_problems(repo, command)


# --- settling a launch ---------------------------------------------------------------------------------------------

def test_the_judge_cap_must_be_inside_the_budget_check(repo):
    judged = dict(FORWARD_JUDGE_KEY='1')
    assert 'let the API judge spend $1.00' in settle(repo, settings=judged)[1][0]
    assert 'spend $5.00' in settle(repo, other_usd=1.0, settings=dict(judged, JUDGE_MAX_USD='5'))[1][0]
    assert settle(repo, other_usd=1.0, settings=judged)[1] == []
    done = subprocess.run(GATE + ['ready', '--remote', 'ops/remote_job.sh', '--deadline', '60', '--basis', '2x measured 30 min',
                                  '--line', 'gpt-oss-20b-gpu', '--hourly', '2.70'], capture_output=True, text=True)
    assert done.returncode == 2 and '--other-usd' in done.stderr


def test_the_driver_is_asked_before_anything_is_deployed_and_the_grace_is_counted(repo):
    assert 'LEAN_ISOLATION=required cannot work' in settle(repo, settings=dict(LEAN_ISOLATION='required'))[1][0]
    assert 'not settings ops/drive_runpod.sh reads: SMOKE_steps' in settle(repo, settings=dict(SMOKE_steps='4'))[1][0]
    assert 'at least 5 min' in settle(repo, deadline=4, basis='2x measured 2 min')[1][0]
    # Besides the deadline: the driver's wait for SSH (SSH_UP_SECONDS and ARM_SECONDS, 10 min), the provisional timer, and the runner's 30 s kill-after
    # with the grace, rounded up together (review round 30).
    record, refused = settle(repo, settings=dict(LEAN_ISOLATION='none', AUTOSTOP_GRACE_SECONDS='3600', SMOKE_STEPS='4'))
    assert refused == [] and record['worst_case_minutes'] == 60 + 10 + 15 + 61
    record, refused = settle(repo, settings=dict(KEEP_POD='1'))
    assert refused == [] and record['worst_case_minutes'] == 96 and record['worst_case_usd'] == pytest.approx(2.70*96/60)
    (repo / 'ops' / 'drive_runpod.sh').write_text('exit 9\n')
    assert 'has no DRIVE_CHECK_ONLY mode' in settle(repo)[1][0]


def test_the_worst_case_reads_the_kill_after_from_the_pod_pins(repo):
    pins = repo / 'ops' / 'pod_pins.env'
    pins.write_text(pins.read_text().replace('KILL_AFTER_SECONDS=30\n', 'KILL_AFTER_SECONDS=600\n'))
    run(repo, 'commit', '-qam', 'longer kill-after')
    assert settle(repo)[0]['worst_case_minutes'] == 60 + 10 + 15 + 20
    pins.write_text(pins.read_text().replace('KILL_AFTER_SECONDS=600\n', ''))
    run(repo, 'commit', '-qam', 'no kill-after')
    assert 'must set KILL_AFTER_SECONDS once' in settle(repo)[1][0]


@pytest.mark.parametrize('deadline, basis, reason', [
    (59, '2x measured 30 min', 'under the 60 min'),
    (60, '1.5x measured 30 min', 'at least 2x'),
    (60, '1.4x estimated 20 min', 'at least 1.5x'),
    (19, 'fits the $0.99 left', 'must read'),
    (10, '2x measured 390 s', 'under the 13 min'),
])
def test_a_cap_without_a_basis_at_the_multiple_is_refused(repo, deadline, basis, reason):
    record, refused = settle(repo, deadline=deadline, basis=basis)
    assert record is None and reason in refused[0] and not (repo / gates.RECORD).exists()


@pytest.mark.parametrize('dirty', ['ops/new.sh', 'requirements-vllm.txt'])
def test_failing_tests_a_dirty_tree_or_a_spent_budget_write_no_record(repo, dirty):
    assert 'tests do not pass' in settle(repo, tests=FAIL)[1][0]
    assert 'budget' in settle(repo, launch_cap=1.0)[1][0]
    (repo / dirty).write_text('uncommitted\n')
    assert 'uncommitted' in settle(repo)[1][0]
    assert not (repo / gates.RECORD).exists()


# --- the launch check the driver makes ------------------------------------------------------------------------------

def launch(repo, settings=None, remote='ops/remote_runpod_smoke.sh', deadline=60):
    return gates.driver_launch(repo, remote, deadline, dict(PATH='/bin') if settings is None else settings)


def test_a_launch_needs_a_record_for_this_script_deadline_and_settings(repo):
    assert 'no launch record' in launch(repo)[0]
    settle(repo, settings=dict(KEEP_POD='1'))
    assert 'not ops/remote_other.sh' in '\n'.join(launch(repo, dict(KEEP_POD='1'), remote='ops/remote_other.sh'))
    assert '60 min deadline, not 19' in '\n'.join(launch(repo, dict(KEEP_POD='1'), deadline=19))
    # Read from the environment the driver runs in, so a setting exported long before is counted like one typed in front of it.
    assert 'not the ones that were checked' in '\n'.join(launch(repo, dict(KEEP_POD='1', JUDGE_MAX_USD='50', FORWARD_JUDGE_KEY='1')))
    assert 'not the ones that were checked' in '\n'.join(launch(repo, dict(PATH='/bin')))
    assert launch(repo, dict(KEEP_POD='1', PATH='/bin', HOME='/x')) == []  # what the driver does not read is not compared
    assert 'already started a launch' in launch(repo, dict(KEEP_POD='1'))[0]  # one record, one launch: its budget counted one


def test_the_record_stops_covering_a_changed_tree_commit_or_ledger(repo):
    settle(repo)
    (repo / 'src' / 'fix.py').write_text('x = 1\n')
    assert 'uncommitted changes' in '\n'.join(gates.launch_problems(repo))
    run(repo, 'add', '-A')
    run(repo, 'commit', '-q', '-m', 'fix')
    assert 'HEAD has moved' in '\n'.join(gates.launch_problems(repo))
    settle(repo)
    assert gates.launch_problems(repo) == []
    with (repo / 'spending' / 'ledger.csv').open('a') as ledger:
        ledger.write('2026-11-25,runpod,x,y,z,0.2,2.70,0,0.54,done,results/x,gpt-oss-20b-gpu\n')
    assert 'ledger has changed' in '\n'.join(gates.launch_problems(repo))


@pytest.fixture
def project(tmp_path):
    """A copy of this project's driver, gate and library in a repository of its own, with ssh, rsync and sleep stubbed."""
    root = tmp_path / 'project'
    files = {'ops/remote_x.sh': 'echo job\n', 'spending/ledger.csv': (PROJECT / 'spending' / 'ledger.csv').read_text().splitlines()[0] + '\n',
             '.gitignore': '.claude/\n.tools/\n__pycache__/\n'}
    files.update({name: (PROJECT / name).read_text() for name in ('ops/drive_runpod.sh', 'ops/pod_pins.env', 'ops/gate.sh', 'ops/remote_job.sh', 'entrypoints/gate.py')})
    shutil.copytree(PROJECT / 'src' / 'rewardhack', root / 'src' / 'rewardhack', ignore=shutil.ignore_patterns('__pycache__'))
    new_repo(root, files)
    (root / '.tools').mkdir()
    (root / '.tools' / 'venv').symlink_to(Path(sys.executable).parent.parent)  # ops/gate.sh finds the project's Python here
    stubs = tmp_path / 'bin'
    stubs.mkdir()
    for name, body in {'ssh': f'echo "$*" >> {tmp_path}/ssh-calls\nexit 255\n', 'rsync': 'exit 1\n', 'sleep': 'exit 0\n'}.items():
        (stubs / name).write_text('#!/bin/bash\n' + body)
        (stubs / name).chmod(0o755)
    environment = {name: value for name, value in os.environ.items() if not gates.DRIVER_SETTINGS.fullmatch(name)}
    return root, dict(environment, PATH=f"{stubs}:{os.environ['PATH']}"), tmp_path / 'ssh-calls'


def drive(project, host='192.0.2.1', **settings):
    root, environment, calls = project
    destination = f'results/{uuid.uuid4().hex}'  # every attempted launch reserves its own results directory
    done = subprocess.run(['bash', 'ops/drive_runpod.sh', host, '22', 'ops/remote_x.sh', destination, '60'], cwd=root,
                          env=dict(environment, **settings), capture_output=True, text=True)
    reached = calls.exists()
    if reached:
        calls.unlink()
    return done.returncode, done.stdout + done.stderr, reached


def test_the_driver_refuses_to_start_unless_the_launch_record_covers_it(project):
    root = project[0]
    code, said, reached = drive(project)
    assert code == 2 and 'no launch record' in said and not reached
    assert gates.ready(root, 'ops/remote_x.sh', 60, '2x measured 30 min', 'gpt-oss-20b-gpu', 2.70, 0.0, None, {}, PASS)[1] == []
    code, said, reached = drive(project, KEEP_POD='1')
    assert code == 2 and 'not the ones that were checked' in said and not reached
    code, said, reached = drive(project)
    assert reached and 'launched_at' in json.loads((root / gates.RECORD).read_text()), said  # let through, and the record used up
    code, said, reached = drive(project)
    assert code == 2 and 'already started a launch' in said and not reached


def test_no_host_skips_the_launch_check(project):
    """The driver's tests use example.invalid with a stubbed gate.sh; the real driver gives that name no pass."""
    for host in ('example.invalid', 'example.invalid.com', 'invalid', '192.0.2.1'):
        code, said, reached = drive(project, host=host)
        assert code == 2 and 'no launch record' in said and not reached, host


# --- the Bash hook ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('command', [
    "ssh -p 22 root@1.2.3.4 'nohup bash pilot-repo/ops/remote_runpod_smoke.sh > out/log 2>&1 &'",
    "ssh root@1.2.3.4 'bash pilot-repo/ops/remote_job.sh pilot-repo/ops/remote_runpod_smoke.sh 3600'",
    'ssh -o ConnectTimeout=10 root@1.2.3.4 bash pilot-repo/ops/remote_runpod_any.sh',  # any remote_*.sh, not only the smoke script
    "ssh root@1.2.3.4 bash -s <<'EOF'\ncd pilot-repo\nbash ops/remote_runpod_smoke.sh\nEOF",
    "bash -c \"ssh root@1.2.3.4 'bash pilot-repo/ops/remote_runpod_smoke.sh'\"",
    'runpodctl create pod --gpuType "RTX PRO 6000"',
    'runpodctl pod create --gpuType "RTX PRO 6000"',
])
def test_a_job_or_a_machine_started_around_the_driver_is_refused(repo, command):
    assert problems(repo, command)


def test_a_job_with_the_timer_and_a_new_machine_still_need_the_record(repo):
    job = "ssh root@1.2.3.4 'REQUIRE_AUTOSTOP=1 nohup bash pilot-repo/ops/remote_job.sh pilot-repo/ops/remote_runpod_smoke.sh 3600 &'"
    create = 'runpodctl create pod --gpuType "RTX PRO 6000"'
    assert 'no launch record' in problems(repo, job)[0] and 'no launch record' in '\n'.join(problems(repo, create))
    settle(repo)
    assert problems(repo, job) == [] and problems(repo, create) == []


@pytest.mark.parametrize('command', [
    "ssh -p 22 root@1.2.3.4 'bash pilot-repo/ops/remote_job.sh --remove-now'",
    "ssh root@1.2.3.4 'bash -s -- --arm-only 900' < ops/remote_job.sh",
    'runpodctl remove pod abc123',
    'grep -n REQUIRE_AUTOSTOP ops/remote_job.sh ops/drive_runpod.sh',
    "ssh root@1.2.3.4 'cat out/autostop.log; tail -5 pilot-repo/ops/remote_runpod_smoke.sh'",
    # The driver checks its own launch now, however it is started: the hook no longer reads it.
    DRIVER, f'eval "{DRIVER}"', f'nohup {DRIVER} > driver.log 2>&1 &',
])
def test_removal_reading_and_the_driver_are_not_held_up_by_the_hook(repo, command):
    assert problems(repo, command) == []


def test_a_metered_judge_needs_a_cap_and_real_prices(repo):
    base = 'python entrypoints/run.py --out x --judge api'
    assert len(problems(repo, base)) == 2  # no cap, and no model with a price
    assert len(problems(repo, base + ' --judge-max-usd 1 --judge-model unlisted')) == 1
    assert problems(repo, base + ' --judge-max-usd 1 --judge-model gpt-5.4-nano') == []
    assert problems(repo, base + ' --judge-url http://127.0.0.1:8000/v1') == []
    assert problems(repo, 'python entrypoints/run.py --out x --judge triage --dry-run') == []
    assert 'quoting could not be read' in problems(repo, base + ' --note "it\'s')[0]


def test_the_hook_refuses_with_exit_2_and_passes_other_tools():
    def hook(call):
        return subprocess.run(GATE + ['hook'], input=json.dumps(call), capture_output=True, text=True)
    done = hook(dict(tool_name='Bash', tool_input=dict(command="ssh root@1.2.3.4 'bash pilot-repo/ops/remote_runpod_smoke.sh'")))
    assert done.returncode == 2 and 'skips ops/remote_job.sh' in done.stderr
    assert hook(dict(tool_name='Bash', tool_input=dict(command='git commit -m "anything at all"'))).returncode == 0  # git's own hooks gate commits
    assert hook(dict(tool_name='Read', tool_input=dict(file_path='x'))).returncode == 0
    assert hook(dict(tool_name='Bash')).returncode == 2  # a call the gate cannot read is refused, not passed


# --- the commit gate: commit-gate (github.com/aeftimia/commit-gate), pinned in .pre-commit-config.yaml ---------------------

CONFIG = yaml.safe_load((PROJECT / '.pre-commit-config.yaml').read_text())
GATE_REPO = 'https://github.com/aeftimia/commit-gate'


def test_the_config_pins_commit_gate_with_every_hook_and_one_size_limit():
    gate = [repo for repo in CONFIG['repos'] if repo['repo'] == GATE_REPO]
    assert len(gate) == 1 and len(gate[0]['rev']) == 40  # a commit SHA, not a tag
    hooks = {hook['id']: hook for hook in gate[0]['hooks']}
    assert set(hooks) == {'withheld-terms', 'withheld-terms-message', 'withheld-terms-push', 'gitleaks', 'gitleaks-message', 'gitleaks-push'}
    for name in ('gitleaks', 'gitleaks-message', 'gitleaks-push'):
        assert hooks[name]['args'] == ['--config', '.gitleaks.toml'], name
    # The push hook's size limit and check-added-large-files' at commit time are one limit, stated twice in two formats.
    large = [hook for repo in CONFIG['repos'] for hook in repo['hooks'] if hook['id'] == 'check-added-large-files']
    assert f"--maxkb={hooks['withheld-terms-push']['args'][1]}" in large[0]['args']
    assert CONFIG['default_install_hook_types'] == ['pre-commit', 'commit-msg']  # pre-push: `commit-gate install-pre-push`


def test_the_dev_extra_installs_the_pinned_commit_gate():
    rev = [repo for repo in CONFIG['repos'] if repo['repo'] == GATE_REPO][0]['rev']
    assert f'commit-gate @ git+{GATE_REPO}@{rev}' in (PROJECT / 'pyproject.toml').read_text()
