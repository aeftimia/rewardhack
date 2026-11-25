"""The spend gate and the commit gate: each refused case is a way money was spent or a file nearly published by mistake."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rewardhack import gates

PROJECT = Path(__file__).resolve().parent.parent
PASS, FAIL = [sys.executable, '-c', 'pass'], [sys.executable, '-c', 'raise SystemExit("3 failed")']
DRIVER = 'bash ops/drive_runpod.sh 1.2.3.4 22 ops/remote_runpod_smoke.sh results/x 60'
TERM = 'ZEBRA'


def run(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    for folder in ('ops', 'src', 'spending', '.claude', '.tools'):
        (root / folder).mkdir(parents=True)
    (root / 'ops' / 'remote_runpod_smoke.sh').write_text('# deadline: 5\necho job\n')
    (root / 'ops' / 'drive_runpod.sh').write_text(  # the real driver's check-only mode, in miniature
        "[[ ${LEAN_ISOLATION:-} != required ]] || { echo 'LEAN_ISOLATION=required cannot work on this pod'; exit 2; }\n"
        "[[ ${DRIVE_CHECK_ONLY:-} == 1 ]] && { echo 'check passed'; exit 0; }\nexit 9\n")
    (root / 'spending' / 'ledger.csv').write_text((PROJECT / 'spending' / 'ledger.csv').read_text().splitlines()[0] + '\n')
    (root / '.claude' / 'commit-gate-terms.txt').write_text(f'# what stays out\n\\b{TERM}\\b\n')
    (root / '.gitignore').write_text('.claude/\n.tools/\n')
    run(root, 'init', '-q', '-b', 'main')
    run(root, 'config', 'user.email', 'test@example.com')
    run(root, 'config', 'user.name', 'Test')
    run(root, 'add', '-A')
    run(root, 'commit', '-q', '-m', 'start')
    return root


def settle(repo, deadline=60, basis='2x measured 30 min', tests=PASS, **kw):
    return gates.ready(repo, 'ops/remote_runpod_smoke.sh', deadline, basis, 'gpt-oss-20b-gpu', 2.70, kw.get('other_usd', 0.0), kw.get('launch_cap'),
                       kw.get('settings', {}), tests)


def problems(repo, command):
    return gates.command_problems(repo, command, str(repo))


# --- spend gate ----------------------------------------------------------------------------------------------------

def test_a_driver_with_no_launch_record_is_refused(repo):
    assert 'no launch record' in problems(repo, DRIVER)[0]


def test_a_settled_launch_starts_and_only_that_launch(repo):
    record, refused = settle(repo, settings=dict(KEEP_POD='1'))
    assert refused == [] and record['worst_case_minutes'] == 85 and record['worst_case_usd'] == pytest.approx(2.70*85/60)
    assert problems(repo, 'KEEP_POD=1 ' + DRIVER) == []
    assert 'not ops/remote_other.sh' in '\n'.join(problems(repo, DRIVER.replace('remote_runpod_smoke', 'remote_other')))
    assert '60 min deadline, not 19' in '\n'.join(problems(repo, DRIVER[:-2] + '19'))


def test_the_judge_cap_must_be_inside_the_budget_check(repo):
    judged = dict(FORWARD_JUDGE_KEY='1')
    assert 'let the API judge spend $1.00' in settle(repo, settings=judged)[1][0]
    assert 'spend $5.00' in settle(repo, other_usd=1.0, settings=dict(judged, JUDGE_MAX_USD='5'))[1][0]
    assert settle(repo, other_usd=1.0, settings=judged)[1] == []
    assert problems(repo, 'FORWARD_JUDGE_KEY=1 ' + DRIVER) == []
    for other in (DRIVER, 'FORWARD_JUDGE_KEY=1 JUDGE_MAX_USD=50 ' + DRIVER, 'export FORWARD_JUDGE_KEY=1 JUDGE_MAX_USD=50; ' + DRIVER,
                  'JUDGE_MAX_USD=50; FORWARD_JUDGE_KEY=1 ' + DRIVER):
        assert problems(repo, other), other
    done =subprocess.run([sys.executable, str(PROJECT / 'entrypoints' / 'gate.py'), 'ready', '--remote', 'ops/remote_job.sh', '--deadline', '60',
                           '--basis', '2x measured 30 min', '--line', 'gpt-oss-20b-gpu', '--hourly', '2.70'], capture_output=True, text=True)
    assert done.returncode == 2 and '--other-usd' in done.stderr


@pytest.mark.parametrize('command', [
    DRIVER + ' 2>&1 | tee driver.log',
    DRIVER.replace('bash ', 'bash -x '),
    DRIVER.replace('bash ', 'bash -eu '),
    'caffeinate -i ' + DRIVER,
    'if ' + DRIVER + '; then echo ok; fi',
    'D=60; ' + DRIVER[:-2] + '$D',
    'xargs ' + DRIVER,
    'nohup ' + DRIVER + ' > driver.log 2>&1 &',
])
def test_the_launch_check_is_not_skipped_by_how_the_driver_is_invoked(repo, command):
    assert problems(repo, command), command  # no record yet: every one of these must reach the check or be refused as unreadable
    settle(repo)
    still = problems(repo, command)
    assert bool(still) == ('$D' in command or 'xargs' in command), still  # the two the gate cannot read stay refused


ARGS = DRIVER[len('bash ops/drive_runpod.sh'):]


@pytest.mark.parametrize('command', [
    f"bash <<'EOF'\n{DRIVER}\nEOF",
    f"bash <<< '{DRIVER}'",
    f'eval "{DRIVER}"',
    f'echo `{DRIVER}`',
    f'bash -s --{ARGS} < ops/drive_runpod.sh',
    f'cat ops/drive_runpod.sh | bash -s --{ARGS}',
    f'S=ops/drive_runpod.sh; bash $S{ARGS}',
    f'bash ops/drive_run*.sh{ARGS}',
    f'env -S "{DRIVER}"',
    f"python3 -c 'import subprocess; subprocess.run(\"{DRIVER}\", shell=True)'",
    f"python3 - <<'EOF'\nimport subprocess\nsubprocess.run('{DRIVER}', shell=True)\nEOF",
    f"KEEP_POD=1 bash -c '{DRIVER}'",
    f"JUDGE_MAX_USD=50 FORWARD_JUDGE_KEY=1 bash -c '{DRIVER}'",
    f'{DRIVER} ; {DRIVER}',
    f'{DRIVER} || {DRIVER}',
])
def test_a_driver_the_check_cannot_follow_is_refused_with_or_without_a_record(repo, command):
    assert problems(repo, command), command
    settle(repo)
    assert problems(repo, command), command


@pytest.mark.parametrize('command', [
    'cat ops/drive_runpod.sh | head -20',
    'grep -c SMOKE ops/drive_runpod.sh && wc -l ops/drive_runpod.sh',
    '[[ -f ops/drive_runpod.sh ]] && echo present',
    'bash -n ops/drive_runpod.sh',
    'git log --oneline -- ops/drive_runpod.sh | cat',
    "git commit -q -F - <<'EOF'\nThe driver, ops/drive_runpod.sh, refuses a misspelt setting\nEOF",
    "bash -c 'grep -n trap ops/drive_runpod.sh'",
])
def test_reading_the_driver_is_still_let_through(repo, command):
    assert problems(repo, command) == []


def test_the_driver_is_asked_before_anything_is_deployed_and_the_grace_is_counted(repo):
    assert 'LEAN_ISOLATION=required cannot work' in settle(repo, settings=dict(LEAN_ISOLATION='required'))[1][0]
    assert 'not settings ops/drive_runpod.sh reads: SMOKE_steps' in settle(repo, settings=dict(SMOKE_steps='4'))[1][0]
    assert 'at least 5 min' in settle(repo, deadline=4, basis='2x measured 2 min')[1][0]
    record, refused = settle(repo, settings=dict(LEAN_ISOLATION='none', AUTOSTOP_GRACE_SECONDS='3600', SMOKE_STEPS='4'))
    assert refused == [] and record['worst_case_minutes'] == 60 + 15 + 60
    assert 'not the ones that were checked' in '\n'.join(problems(repo, 'LEAN_ISOLATION=none ' + DRIVER))
    assert problems(repo, 'SMOKE_STEPS=4 AUTOSTOP_GRACE_SECONDS=3600 LEAN_ISOLATION=none KEEP_UNRELATED=1 ' + DRIVER) == []
    (repo / 'ops' / 'drive_runpod.sh').write_text('exit 9\n')
    assert 'has no DRIVE_CHECK_ONLY mode' in settle(repo)[1][0]


def test_a_record_covers_one_launch(repo):
    settle(repo)
    launches = []
    assert gates.command_problems(repo, DRIVER, str(repo), launches=launches) == [] and launches == ['ops/remote_runpod_smoke.sh']
    gates.mark_launched(repo)  # what the hook does when it lets a driver command through
    assert 'already started a launch' in problems(repo, DRIVER)[0]
    reads = []
    assert gates.command_problems(repo, 'cat ops/drive_runpod.sh; bash -n ops/drive_runpod.sh', str(repo), launches=reads) == [] and reads == []


def test_the_record_stops_covering_a_changed_tree_commit_or_ledger(repo):
    settle(repo)
    (repo / 'src' / 'fix.py').write_text('x = 1\n')
    assert 'uncommitted changes' in '\n'.join(problems(repo, DRIVER))
    run(repo, 'add', '-A')
    run(repo, 'commit', '-q', '-m', 'fix')
    assert 'HEAD has moved' in '\n'.join(problems(repo, DRIVER))
    settle(repo)
    assert problems(repo, DRIVER) == []
    with (repo / 'spending' / 'ledger.csv').open('a') as ledger:
        ledger.write('2026-11-25,runpod,x,y,z,0.2,2.70,0,0.54,done,results/x,gpt-oss-20b-gpu\n')
    assert 'ledger has changed' in '\n'.join(problems(repo, DRIVER))


@pytest.mark.parametrize('deadline, basis, reason', [
    (59, '2x measured 30 min', 'under the 60 min'),
    (60, '1.5x measured 30 min', 'at least 2x'),
    (60, '2x estimated 20 min', 'at least 3x'),
    (19, 'fits the $0.99 left', 'must read'),
    (10, '2x measured 390 s', 'under the 13 min'),
])
def test_a_cap_without_a_basis_at_the_multiple_is_refused(repo, deadline, basis, reason):
    record, refused = settle(repo, deadline=deadline, basis=basis)
    assert record is None and reason in refused[0] and not (repo / gates.RECORD).exists()


def test_failing_tests_a_dirty_tree_or_a_spent_budget_write_no_record(repo):
    assert 'tests do not pass' in settle(repo, tests=FAIL)[1][0]
    assert 'budget' in settle(repo, launch_cap=1.0)[1][0]
    (repo / 'ops' / 'new.sh').write_text('echo\n')
    assert 'uncommitted' in settle(repo)[1][0]
    assert not (repo / gates.RECORD).exists()


@pytest.mark.parametrize('command', [
    "ssh -p 22 root@1.2.3.4 'nohup bash pilot-repo/ops/remote_runpod_smoke.sh > out/log 2>&1 &'",
    "ssh root@1.2.3.4 'bash pilot-repo/ops/remote_job.sh pilot-repo/ops/remote_runpod_smoke.sh 3600'",
    'ssh -o ConnectTimeout=10 root@1.2.3.4 bash pilot-repo/ops/remote_runpod_genbench.sh',
    'runpodctl create pod --gpuType "RTX PRO 6000"',
    'bash ops/launch_aws.sh --type g6e.xlarge --hours 1 --model m --hourly 1.8 --approved',
])
def test_a_job_or_a_machine_started_around_the_driver_is_refused(repo, command):
    assert problems(repo, command)


def test_a_job_with_the_timer_still_needs_the_record(repo):
    command = "ssh root@1.2.3.4 'REQUIRE_AUTOSTOP=1 nohup bash pilot-repo/ops/remote_job.sh pilot-repo/ops/remote_runpod_smoke.sh 3600 &'"
    assert 'no launch record' in problems(repo, command)[0]
    settle(repo)
    assert problems(repo, command) == []


@pytest.mark.parametrize('command', [
    "ssh -p 22 root@1.2.3.4 'bash pilot-repo/ops/remote_job.sh --remove-now'",
    "ssh root@1.2.3.4 'bash -s -- --arm-only 900' < ops/remote_job.sh",
    'runpodctl remove pod abc123',
    'cat ops/drive_runpod.sh | head -20',
    'grep -n REQUIRE_AUTOSTOP ops/remote_job.sh ops/drive_runpod.sh',
    'bash ops/launch_aws.sh --type g6e.xlarge --hours 1 --model m --hourly 1.8',
    "ssh root@1.2.3.4 'cat out/autostop.log; tail -5 out/remote.log'",
])
def test_removal_and_reading_are_never_held_up(repo, command):
    assert problems(repo, command) == []


def test_a_metered_judge_needs_a_cap_and_real_prices(repo):
    base = 'python entrypoints/run.py --out x --judge api'
    assert len(problems(repo, base)) == 3
    assert len(problems(repo, base + ' --judge-max-usd 1 --judge-usd-in 0 --judge-usd-out 0.4')) == 1
    assert problems(repo, base + ' --judge-max-usd 1 --judge-usd-in 0.05 --judge-usd-out 0.4') == []
    assert problems(repo, base + ' --judge-url http://127.0.0.1:8000/v1') == []
    assert problems(repo, 'python entrypoints/run.py --out x --judge triage --dry-run') == []


# --- commit gate ---------------------------------------------------------------------------------------------------

def test_a_withheld_term_is_refused_in_a_file_and_in_a_message(repo):
    (repo / 'notes.md').write_text(f'fine\npaid from the {TERM} balance\n')
    assert problems(repo, 'git add notes.md') == ['notes.md:2 matches withheld term 1 of .claude/commit-gate-terms.txt']
    assert problems(repo, 'git add -A && git commit -m "notes"')
    assert problems(repo, f'git commit -m "ledger: {TERM} funds"')
    assert problems(repo, f"git commit -m \"$(cat <<'EOF'\nLedger row\n\nIt's the {TERM} balance.\nEOF\n)\"")
    assert problems(repo, f"git commit -q -F - <<'EOF'\nLedger row\n\nIt's the {TERM} balance.\nEOF")
    assert problems(repo, "git commit -q -F - <<'EOF'\nLedger row\n\nIt's settled.\nEOF\ngit log --oneline -1") == []
    (repo / 'message.txt').write_text(f'{TERM}\n')
    assert problems(repo, 'git commit -F message.txt')
    (repo / 'notes.md').write_text('fine\nzebras are not the term\n')
    assert problems(repo, 'git add notes.md && git commit -m "notes, with a \'quote\'"') == []


def test_secrets_binaries_and_oversized_files_are_refused(repo):
    (repo / 'config.py').write_text("KEY = 'sk-" + 'a1B2' * 8 + "'\n")
    (repo / 'training_args.bin').write_bytes(b'\x80\x02\x00\x00weights')
    (repo / 'tokenizer.json').write_bytes(b'{"a": 1}\n' * (gates.MAX_BYTES // 9 + 1))
    (repo / 'deploy.pem').write_text('not even a key\n')
    found = '\n'.join(problems(repo, 'git add -f .'))
    for reason in ('config.py:1 looks like an OpenAI-style key', 'training_args.bin is a binary file', 'tokenizer.json is 15 MB',
                   'deploy.pem is named like a secret'):
        assert reason in found
    assert 'sk-a1B2' not in found


def test_a_commit_reads_what_is_staged_and_a_push_what_is_committed(repo):
    (repo / 'notes.md').write_text(f'{TERM}\n')
    assert problems(repo, 'git commit -m ok') == []
    run(repo, 'add', 'notes.md')
    assert problems(repo, 'git commit -m ok')
    run(repo, 'commit', '-q', '-m', 'ok')
    assert problems(repo, 'git status && git log --oneline') == []
    assert problems(repo, 'git push origin main') == ['notes.md:1 matches withheld term 1 of .claude/commit-gate-terms.txt']
    (repo / 'ops' / 'remote_runpod_smoke.sh').write_text(f'echo {TERM}\n')
    assert problems(repo, 'git commit -am ok')


def test_another_repository_is_not_this_gates_business(repo, tmp_path):
    other = tmp_path / 'other'
    other.mkdir()
    run(other, 'init', '-q')
    (other / 'a.md').write_text(f'{TERM}\n')
    assert problems(repo, f'cd {other} && git add a.md') == []
    assert problems(repo, f'git -C {other} add a.md') == []


def test_no_terms_file_and_unreadable_quoting_refuse(repo):
    assert 'quoting could not be read' in problems(repo, 'git commit -m "it\'s')[0]
    assert problems(repo, 'echo "it\'s') == []
    (repo / '.claude' / 'commit-gate-terms.txt').unlink()
    with pytest.raises(FileNotFoundError):
        problems(repo, 'git add -A')


def test_the_hook_refuses_with_exit_2_and_passes_other_tools():
    def hook(call):
        return subprocess.run([sys.executable, str(PROJECT / 'entrypoints' / 'gate.py'), 'hook'], input=json.dumps(call),
                              capture_output=True, text=True)
    done = hook(dict(tool_name='Bash', tool_input=dict(command="ssh root@1.2.3.4 'bash pilot-repo/ops/remote_runpod_smoke.sh'"), cwd=str(PROJECT)))
    assert done.returncode == 2 and 'skips ops/remote_job.sh' in done.stderr
    assert hook(dict(tool_name='Bash', tool_input=dict(command='ls ops'), cwd=str(PROJECT))).returncode == 0
    assert hook(dict(tool_name='Read', tool_input=dict(file_path='x'), cwd=str(PROJECT))).returncode == 0
    assert hook(dict(tool_name='Bash')).returncode == 2  # a call the gate cannot read is refused, not passed
