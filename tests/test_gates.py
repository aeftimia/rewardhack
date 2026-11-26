"""The spend gate and the commit gate: each refused case is a way money was spent or a file nearly published by mistake.

The commit gate's terms hooks are tested as git runs them through pre-commit's entries, in temporary repositories; the launch check as the
driver runs it, with ssh stubbed. Nothing here contacts a pod.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
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


def attempt(repo, *args):
    """A git command as a person would type it, hooks and all."""
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)


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
    record, refused = settle(repo, settings=dict(LEAN_ISOLATION='none', AUTOSTOP_GRACE_SECONDS='3600', SMOKE_STEPS='4'))
    assert refused == [] and record['worst_case_minutes'] == 60 + 15 + 60
    record, refused = settle(repo, settings=dict(KEEP_POD='1'))
    assert refused == [] and record['worst_case_minutes'] == 85 and record['worst_case_usd'] == pytest.approx(2.70*85/60)
    (repo / 'ops' / 'drive_runpod.sh').write_text('exit 9\n')
    assert 'has no DRIVE_CHECK_ONLY mode' in settle(repo)[1][0]


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
    files.update({name: (PROJECT / name).read_text() for name in ('ops/drive_runpod.sh', 'ops/gate.sh', 'ops/remote_job.sh', 'entrypoints/gate.py')})
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
    done = subprocess.run(['bash', 'ops/drive_runpod.sh', host, '22', 'ops/remote_x.sh', 'results/x', '60'], cwd=root,
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
    'ssh -o ConnectTimeout=10 root@1.2.3.4 bash pilot-repo/ops/remote_runpod_genbench.sh',
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


# --- the commit gate, as git runs it ---------------------------------------------------------------------------------

CONFIG = yaml.safe_load((PROJECT / '.pre-commit-config.yaml').read_text())
HOOKS = [hook for repo in CONFIG['repos'] for hook in repo['hooks'] if 'entry' in hook]  # those that run this project's entries
# How pre-commit hands each stage's details to a hook: the message file as an argument, the remote and the first pushed
# commit in its environment (a push that only deletes runs no hooks).
SHIMS = {'pre-commit': '{entry}', 'commit-msg': '{entry} "$1"',
         'pre-push': 'read -r local sha remote _\n[ "$sha" = 0000000000000000000000000000000000000000 ] && exit 0\n'
                     'PRE_COMMIT_REMOTE_NAME="$1" PRE_COMMIT_TO_REF="$sha" PRE_COMMIT_LOCAL_BRANCH="$local" PRE_COMMIT_REMOTE_BRANCH="$remote" {entry}'}


def test_the_config_runs_each_check_at_each_stage_it_installs():
    assert CONFIG['default_install_hook_types'] == list(gates.TERMS_CHECKS) == list(gates.GITLEAKS_RUNS)
    for check in gates.COMMIT_GATE:
        wired = {hook['stages'][0]: hook for hook in HOOKS if hook['entry'].startswith(f'bash ops/gate.sh {check} ')}
        assert set(wired) == set(gates.TERMS_CHECKS), check
        for stage, hook in wired.items():
            assert hook['entry'] == f'bash ops/gate.sh {check} {stage}' and hook['stages'] == [stage] and hook['always_run']
            assert hook.get('pass_filenames', True) == (stage == 'commit-msg'), (check, stage)
    large = [hook for repo in CONFIG['repos'] for hook in repo['hooks'] if hook['id'] == 'check-added-large-files']
    assert f'--maxkb={gates.MAX_KB}' in large[0]['args']
    pinned = [line for line in (PROJECT / 'pyproject.toml').read_text().splitlines() if 'pre-commit==' in line]
    assert f'"pre-commit=={CONFIG["minimum_pre_commit_version"]}"' in pinned[0]


@pytest.fixture
def gated(repo):
    """`repo` with the terms hooks installed as pre-commit runs them (gitleaks, not installed here, is tested apart)."""
    hooks = repo / '.git' / 'hooks'
    for hook in HOOKS:
        if hook['entry'].startswith('bash ops/gate.sh terms '):
            entry = hook['entry'].replace('ops/gate.sh', shlex.quote(str(PROJECT / 'ops' / 'gate.sh')), 1)
            (hooks / hook['stages'][0]).write_text(f'#!/bin/sh\nset -e\n{SHIMS[hook["stages"][0]].format(entry=entry)}\n')
            (hooks / hook['stages'][0]).chmod(0o755)
    return repo


def refused(done, *reasons):
    said = done.stdout + done.stderr
    return done.returncode != 0 and all(reason in said for reason in reasons)


def test_a_withheld_term_is_refused_in_a_file_and_in_a_message(gated):
    (gated / 'notes.md').write_text(f'fine\npaid from the {TERM} balance\n')
    run(gated, 'add', 'notes.md')
    assert refused(attempt(gated, 'commit', '-m', 'notes'), 'notes.md:2 matches withheld term 1 of .claude/commit-gate-terms.txt')
    (gated / 'notes.md').write_text('fine\nzebras are not the term\n')
    run(gated, 'add', 'notes.md')
    assert refused(attempt(gated, 'commit', '-m', f'ledger: {TERM} funds'), 'the commit message:1 matches withheld term 1')
    (gated / 'message.txt').write_text(f"Ledger row\n\nIt's the {TERM} balance.\n")
    assert refused(attempt(gated, 'commit', '-F', str(gated / 'message.txt')), 'the commit message:3')
    done = attempt(gated, 'commit', '-m', "notes, with a 'quote'")
    assert done.returncode == 0, done.stderr
    # `commit -a` commits from a temporary index: that is the one read.
    (gated / 'ops' / 'remote_runpod_smoke.sh').write_text(f'echo {TERM}\n')
    assert refused(attempt(gated, 'commit', '-am', 'ok'), 'ops/remote_runpod_smoke.sh:1')
    assert attempt(gated, 'log', '--format=%s', '-1').stdout.strip() == "notes, with a 'quote'"


def test_files_no_term_could_be_read_in_are_refused(gated):
    (gated / 'training_args.bin').write_bytes(b'\x80\x02\x00\x00weights')
    (gated / 'late_nul.txt').write_bytes(b'text\n' * 3000 + b'\0')
    (gated / 'utf16.txt').write_text(f'{TERM}\n', encoding='utf-16')
    (gated / 'latin1.txt').write_bytes(b'caf\xe9\n')
    run(gated, 'add', '-f', '.')
    done = attempt(gated, 'commit', '-m', 'everything')
    assert refused(done, 'training_args.bin is a binary file', 'late_nul.txt is a binary file', 'utf16.txt is a binary file',
                   'latin1.txt is not UTF-8')


def test_with_no_terms_file_every_commit_is_refused(gated):
    (gated / '.claude' / 'commit-gate-terms.txt').unlink()
    (gated / 'notes.md').write_text('harmless\n')
    run(gated, 'add', 'notes.md')
    assert refused(attempt(gated, 'commit', '-m', 'notes'), 'commit-gate-terms.txt is not there')


def test_a_commit_in_a_linked_worktree_is_gated_with_the_main_checkouts_terms(gated, tmp_path):
    worktree = tmp_path / 'side'
    run(gated, 'worktree', 'add', '-q', '-b', 'side', str(worktree))
    assert not (worktree / '.claude').exists()  # the terms are only in the main checkout, untracked
    (worktree / 'notes.md').write_text(f'{TERM}\n')
    run(worktree, 'add', 'notes.md')
    assert refused(attempt(worktree, 'commit', '-m', 'notes'), 'notes.md:1 matches withheld term 1')
    (worktree / 'notes.md').write_text('harmless\n')
    run(worktree, 'add', 'notes.md')
    done = attempt(worktree, 'commit', '-m', 'notes')  # passes only if the terms file was found from the worktree
    assert done.returncode == 0, done.stderr
    assert refused(attempt(worktree, 'commit', '--allow-empty', '-m', f'{TERM}'), 'the commit message:1')


def test_a_push_reads_every_commit_it_publishes(gated, tmp_path):
    remote = tmp_path / 'remote.git'
    run(tmp_path, 'init', '-q', '--bare', str(remote))
    run(gated, 'remote', 'add', 'origin', str(remote))
    done = attempt(gated, 'push', '-q', 'origin', 'main')
    assert done.returncode == 0, done.stderr
    # Made with the commit hooks skipped: the push must still find the term in a commit that a later one removed, and in a message.
    (gated / 'notes.md').write_text(f'{TERM}\n')
    run(gated, 'add', 'notes.md')
    run(gated, 'commit', '-q', '--no-verify', '-m', 'notes')
    run(gated, 'rm', '-q', 'notes.md')
    run(gated, 'commit', '-q', '--no-verify', '-m', f'drop the {TERM} note')
    done = attempt(gated, 'push', 'origin', 'main')
    assert refused(done, 'notes.md:1 matches withheld term 1', 'the message of commit')
    assert attempt(remote, 'log', '--format=%s', 'main').stdout.strip() == 'start', 'the refused commits reached the remote'
    run(gated, 'reset', '-q', '--hard', 'origin/main')
    (gated / 'clean.md').write_text('harmless\n')
    run(gated, 'add', 'clean.md')
    run(gated, 'commit', '-q', '-m', 'clean')
    done = attempt(gated, 'push', '-q', 'origin', 'main', 'main:other')  # a new branch is read only for what the remote lacks
    assert done.returncode == 0, done.stderr
    assert attempt(gated, 'push', '-q', 'origin', ':other').returncode == 0  # deleting a remote branch publishes nothing


def test_a_push_of_several_refs_reads_them_all(gated, tmp_path):
    # pre-commit hands its hooks a range for the first pushed ref only; the second ref here carries the term.
    remote = tmp_path / 'remote.git'
    run(tmp_path, 'init', '-q', '--bare', str(remote))
    run(gated, 'remote', 'add', 'origin', str(remote))
    run(gated, 'push', '-q', 'origin', 'main')
    run(gated, 'commit', '-q', '--allow-empty', '-m', 'clean')
    run(gated, 'switch', '-q', '-c', 'zz-side')
    (gated / 'notes.md').write_text(f'{TERM}\n')
    run(gated, 'add', 'notes.md')
    run(gated, 'commit', '-q', '--no-verify', '-m', 'notes')
    run(gated, 'switch', '-q', 'main')
    assert refused(attempt(gated, 'push', 'origin', 'main', 'zz-side'), 'notes.md:1 matches withheld term 1')
    # Stricter than the push itself: an unpushed branch carrying the term holds back the others too.
    assert refused(attempt(gated, 'push', 'origin', 'main'), 'notes.md:1')


def test_a_pushed_commit_no_local_ref_reaches_is_read(gated, tmp_path):
    remote = tmp_path / 'remote.git'
    run(tmp_path, 'init', '-q', '--bare', str(remote))
    run(gated, 'remote', 'add', 'origin', str(remote))
    run(gated, 'push', '-q', 'origin', 'main')
    run(gated, 'switch', '-q', '-c', 'gone')
    (gated / 'notes.md').write_text(f'{TERM}\n')
    run(gated, 'add', 'notes.md')
    run(gated, 'commit', '-q', '--no-verify', '-m', 'notes')
    sha = attempt(gated, 'rev-parse', 'HEAD').stdout.strip()
    run(gated, 'switch', '-q', 'main')
    run(gated, 'branch', '-q', '-D', 'gone')
    assert refused(attempt(gated, 'push', 'origin', f'{sha}:refs/heads/x'), 'notes.md:1 matches withheld term 1')


def test_a_push_reads_the_size_of_every_pushed_file(gated, tmp_path):
    # check-added-large-files reads the working tree: a big file committed past the hooks and deleted again is not there.
    remote = tmp_path / 'remote.git'
    run(tmp_path, 'init', '-q', '--bare', str(remote))
    run(gated, 'remote', 'add', 'origin', str(remote))
    run(gated, 'push', '-q', 'origin', 'main')
    (gated / 'rows.jsonl').write_bytes(b'{"a": 1}\n' * (gates.MAX_KB * 1024 // 9 + 1))
    run(gated, 'add', 'rows.jsonl')
    run(gated, 'commit', '-q', '--no-verify', '-m', 'rows')
    run(gated, 'rm', '-q', 'rows.jsonl')
    run(gated, 'commit', '-q', '--no-verify', '-m', 'drop rows')
    assert refused(attempt(gated, 'push', 'origin', 'main'), f'rows.jsonl is {gates.MAX_KB} KB in a pushed commit')


def pushable(gated, tmp_path, name='origin'):
    remote = tmp_path / f'{name}.git'
    run(tmp_path, 'init', '-q', '--bare', str(remote))
    run(gated, 'remote', 'add', name, str(remote))
    return remote


def test_only_what_the_destination_has_is_not_read_again(gated, tmp_path):
    # A branch that only a private mirror has is read when it goes to origin; what origin has is not read again.
    pushable(gated, tmp_path)
    run(gated, 'push', '-q', 'origin', 'main')
    pushable(gated, tmp_path, 'mirror')
    run(gated, 'switch', '-q', '-c', 'feat')
    (gated / 'notes.md').write_text(f'{TERM}\n')
    run(gated, 'add', 'notes.md')
    run(gated, 'commit', '-q', '--no-verify', '-m', 'notes')
    run(gated, 'push', '-q', '--no-verify', 'mirror', 'feat')
    run(gated, 'switch', '-q', 'main')
    run(gated, 'branch', '-q', '-D', 'feat')
    assert refused(attempt(gated, 'push', 'origin', 'mirror/feat:refs/heads/feat'), 'notes.md:1 matches withheld term 1')
    run(gated, 'commit', '-q', '--allow-empty', '-m', 'clean')
    run(gated, 'push', '-q', '--no-verify', 'origin', 'main')  # the mirror's branch is still unpushed to origin
    run(gated, 'commit', '-q', '--allow-empty', '-m', 'clean 2')
    assert refused(attempt(gated, 'push', 'origin', 'main'), 'notes.md:1')  # held back while the mirror's branch is not on origin


def test_odd_paths_submodules_and_tag_chains_are_read(gated, tmp_path):
    pushable(gated, tmp_path)
    run(gated, 'push', '-q', 'origin', 'main')
    # macOS will not make a file with this name, so it goes straight into the index.
    blob = subprocess.run(['git', '-C', str(gated), 'hash-object', '-w', '--stdin'], input=b'harmless\n', capture_output=True, check=True).stdout
    subprocess.run([b'git', b'-C', bytes(gated), b'update-index', b'--add', b'--cacheinfo', b'100644,' + blob.strip() + b',caf\xe9.md'],
                   capture_output=True, check=True)
    assert refused(attempt(gated, 'commit', '-m', 'notes'), 'the path caf\\xe9.md is not UTF-8')
    run(gated, 'reset', '-q', '--hard')
    head = attempt(gated, 'rev-parse', 'HEAD').stdout.strip()
    run(gated, 'update-index', '--add', '--cacheinfo', f'160000,{head},vendor/{TERM}')
    assert refused(attempt(gated, 'commit', '-m', 'submodule'), f'the path vendor/{TERM}:1')
    run(gated, 'reset', '-q', '--hard')
    run(gated, 'commit', '-q', '--allow-empty', '-m', 'clean')
    run(gated, 'tag', '-a', 'inner', '-m', f'the {TERM} release')
    run(gated, 'tag', '-a', 'outer', 'inner', '-m', 'wraps it')
    run(gated, 'tag', '-d', 'inner')
    assert refused(attempt(gated, 'push', 'origin', 'outer'), 'the message of tag')


def test_a_tag_on_a_tree_does_not_break_a_push_and_is_read_when_pushed(gated, tmp_path):
    pushable(gated, tmp_path)
    run(gated, 'push', '-q', 'origin', 'main')
    run(gated, 'tag', '-a', 'tree', 'HEAD^{tree}', '-m', f'the {TERM} tree')
    run(gated, 'commit', '-q', '--allow-empty', '-m', 'clean')
    done = attempt(gated, 'push', '-q', 'origin', 'main')
    assert done.returncode == 0, done.stderr
    tree_tag = attempt(gated, 'rev-parse', 'tree').stdout.strip()
    assert 'the message of tag' in ' '.join(gates.terms_problems(gated, 'pre-push', 'origin', [tree_tag], []))


def test_pushes_the_hooks_cannot_read_are_refused(gated, tmp_path):
    pushable(gated, tmp_path)
    head = attempt(gated, 'rev-parse', 'HEAD').stdout.strip()
    # pre-commit sets no PRE_COMMIT_TO_REF for a push that starts a new history there.
    assert 'starts a new history' in gates.terms_problems(gated, 'pre-push', 'origin', [], [])[0]
    assert 'white space' in gates.terms_problems(gated, 'pre-push', '/tmp/a remote.git', [head], [])[0]
    run(gated, 'config', 'remote.origin/x.url', str(tmp_path / 'elsewhere.git'))  # `git remote add` refuses this; config does not
    assert 'remotes named under it (origin/x)' in gates.terms_problems(gated, 'pre-push', 'origin', [head], [])[0]


def test_paths_tag_messages_and_ref_names_are_read(gated, tmp_path):
    pushable(gated, tmp_path)
    run(gated, 'push', '-q', 'origin', 'main')
    (gated / f'{TERM}.md').write_text('harmless\n')
    run(gated, 'add', '.')
    assert refused(attempt(gated, 'commit', '-m', 'notes'), f'the path {TERM}.md:1 matches withheld term 1')
    run(gated, 'commit', '-q', '--no-verify', '-m', 'notes')
    assert refused(attempt(gated, 'push', 'origin', 'main'), f'the path {TERM}.md:1')
    run(gated, 'reset', '-q', '--hard', 'origin/main')
    run(gated, 'commit', '-q', '--allow-empty', '-m', 'clean')
    run(gated, 'tag', '-a', 'v1', '-m', f'the {TERM} release')
    assert refused(attempt(gated, 'push', 'origin', 'v1'), 'the message of tag')
    run(gated, 'tag', '-d', 'v1')
    assert refused(attempt(gated, 'push', 'origin', f'main:refs/heads/{TERM}'), 'the ref name refs/heads/')
    done = attempt(gated, 'push', '-q', 'origin', 'main')
    assert done.returncode == 0, done.stderr


@pytest.fixture
def fake_gitleaks(tmp_path, monkeypatch):
    """A `gitleaks` on PATH that logs each call (its arguments and standard input) and fails when its input says LEAK."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'gitleaks.log'
    (bin_dir / 'gitleaks').write_text(f'#!/bin/sh\necho "$*" >> {log}\nif [ "$1" = stdin ]; then\n  input=$(cat)\n  echo "  $input" >> {log}\n'
                                      '  case $input in *LEAK*) exit 1;; esac\nfi\nexit 0\n')
    (bin_dir / 'gitleaks').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return log


def test_gitleaks_reads_the_staged_changes_the_message_and_every_unpushed_commit(repo, fake_gitleaks):
    flags = ' '.join(gates.GITLEAKS)
    assert gates.gitleaks_problems(repo, 'pre-commit') == []
    assert fake_gitleaks.read_text() == f'git --pre-commit --staged {flags}\n'
    message = repo / 'MSG'
    scissors = '# ------------------------ >8 ------------------------\n'
    message.write_text(f'fine\n# Please enter the commit message\n{scissors}# Do not modify or remove the line above.\n'
                       '# Everything below it will be ignored.\n-LEAK removed\n')
    assert gates.gitleaks_problems(repo, 'commit-msg', message) == []  # a secret being removed, in the `commit -v` diff, is not the message
    message.write_text(f'fine\n{scissors}the key is LEAK\n')  # with -m or -F git keeps a scissors line and what follows
    assert gates.gitleaks_problems(repo, 'commit-msg', message) == ['gitleaks stdin exited 1: see its findings above']
    message.write_text('the key is LEAK\n')
    assert gates.gitleaks_problems(repo, 'commit-msg', message) == ['gitleaks stdin exited 1: see its findings above']
    run(repo, 'commit', '-q', '--allow-empty', '-m', 'carries a LEAK in its message')
    sha = attempt(repo, 'rev-parse', 'HEAD').stdout.strip()
    fake_gitleaks.write_text('')
    assert gates.gitleaks_problems(repo, 'pre-push', 'origin', [sha], ['main']) == ['gitleaks stdin exited 1: see its findings above']
    calls = fake_gitleaks.read_text().splitlines()
    assert calls[:2] == [f'git --log-opts=--diff-merges=first-parent --branches --tags HEAD --remotes {sha} --not --remotes=origin {flags}',
                         f'stdin {flags}']
    assert 'carries a LEAK' in calls[2] and 'start' in '\n'.join(calls[3:])


def test_gitleaks_is_not_run_beside_a_gitleaksignore(repo, fake_gitleaks):
    # gitleaks 8.30.1 reads a .gitleaksignore at the top of the scanned tree whatever --gitleaks-ignore-path says.
    (repo / '.gitleaksignore').write_text('anything\n')
    assert '.gitleaksignore exists' in gates.gitleaks_problems(repo, 'pre-commit')[0]
    assert not fake_gitleaks.exists()
