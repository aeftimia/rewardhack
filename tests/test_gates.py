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
    (root / 'ops' / 'remote_runpod_smoke.sh').write_text('echo job\n')
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
    return gates.ready(repo, 'ops/remote_runpod_smoke.sh', deadline, basis, 'gpt-oss-20b-gpu', 2.70, 0.0, kw.get('launch_cap'), tests)


def problems(repo, command):
    return gates.command_problems(repo, command, str(repo))


# --- spend gate ----------------------------------------------------------------------------------------------------

def test_a_driver_with_no_launch_record_is_refused(repo):
    assert 'no launch record' in problems(repo, DRIVER)[0]


def test_a_settled_launch_starts_and_only_that_launch(repo):
    record, refused = settle(repo)
    assert refused == [] and record['worst_case_minutes'] == 85 and record['worst_case_usd'] == pytest.approx(2.70*85/60)
    assert problems(repo, 'KEEP_POD=1 ' + DRIVER) == []
    assert 'not ops/remote_other.sh' in '\n'.join(problems(repo, DRIVER.replace('remote_runpod_smoke', 'remote_other')))
    assert '60 min deadline, not 19' in '\n'.join(problems(repo, DRIVER[:-2] + '19'))


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
