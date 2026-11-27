"""Values the pod scripts and the library both need are stated once and held equal here (owner, 26 Nov). A copy that drifts
fails this test instead of quietly running another checkpoint, verifier, server version or budget."""
import re
from pathlib import Path

from rewardhack import gates, verifier

ROOT = Path(__file__).resolve().parents[1]


def shell_values(path):
    return dict(re.findall(r'^([A-Z][A-Z0-9_]*)=(\S+)', (ROOT/path).read_text(), re.M))


PINS = shell_values('ops/pod_pins.env')


def test_the_pod_pins_agree_with_the_library():
    assert PINS['LEAN_VERSION'] == verifier.LEAN_VERSION
    assert f"127.0.0.1:{PINS['ROLLOUT_PORT']}'" in (ROOT/'src/rewardhack/vllm_rollouts.py').read_text()  # --rollout-url's default
    run = (ROOT/'entrypoints/run.py').read_text()  # the trainer's default model is the judge model served on the pod
    assert f"default='{PINS['POD_JUDGE']}'" in run and f"default='{PINS['POD_JUDGE_REV']}'" in run
    assert PINS['ROLLOUT_PORT'] != PINS['JUDGE_PORT']


def test_no_pod_script_restates_a_pinned_value():
    """Every value in ops/pod_pins.env appears in no other ops script: they read it."""
    for script in sorted((ROOT/'ops').glob('*.sh')):
        text = script.read_text()
        for name, value in PINS.items():
            if len(value) > 6:  # revisions, model names, versions; ports are checked through their names below
                assert value not in text, f'{script.name} restates {name}={value}: source ops/pod_pins.env instead'
        assert not re.search(r'\b80(11|12)\b', text), f'{script.name} restates a port: use $ROLLOUT_PORT or $JUDGE_PORT'
        # A version written into a script: install from requirements-vllm.txt.
        assert not re.search(r'vllm==[0-9]', text), f'{script.name} pins vLLM: install from requirements-vllm.txt'


def test_the_launch_gate_counts_what_the_driver_and_runner_actually_do():
    """gates.py's worst case uses the provisional timer, the default grace and the default judge cap; the scripts that
    set them must use the same numbers, or the budget check undercounts (DRY audit, 26 Nov)."""
    driver = (ROOT/'ops/drive_runpod.sh').read_text()
    runner = (ROOT/'ops/remote_job.sh').read_text()
    assert f'--arm-only {gates.PROVISIONAL_MINUTES*60}' in driver
    assert f'AUTOSTOP_GRACE_SECONDS:-{gates.DRIVER_GRACE_SECONDS}}}' in runner
    assert driver.count(f'JUDGE_MAX_USD:-{gates.DRIVER_JUDGE_MAX_USD:g}}}') == driver.count('JUDGE_MAX_USD:-1') > 0
    # What the pod is sent when a setting is unset (notes section 80 moved these out of the launch line).
    assert f'else SENT_JUDGE_MAX_USD={gates.DRIVER_JUDGE_MAX_USD:g}; fi' in driver
    assert f'else SENT_GRACE={gates.DRIVER_GRACE_SECONDS}; fi' in driver


def test_every_part_one_job_names_known_arms_and_numeric_penalties():
    """A typo in ops/pod_pins.env's PART_ONE_RUNS would surface only on the pod, after setup was paid (review round 25)."""
    import subprocess
    from rewardhack.outcomes import ARMS
    from rewardhack.prompts import VARIANTS
    from rewardhack.reporting import MODES
    rows = subprocess.run(['bash', '-c', 'source ops/pod_pins.env; printf "%s\\n" "${PART_ONE_RUNS[@]}"'], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.split('\n')[:-1]
    assert rows[0] == '' and len(rows) >= 2  # row 0 is the preflight run, not a Part 1 job
    for row in rows[1:]:
        names = []
        for run in row.split():
            name, arm, penalty, variant, report = run.split(':')
            assert re.fullmatch(r'[a-z0-9]+', name) and arm in ARMS and 0 <= float(penalty) <= 1, run
            assert variant in VARIANTS and report in MODES, run
            names.append(name)
        assert len(names) == len(set(names)), row

    # The approved missing 20b cells cover the three rewards across the three new
    # prompt/format combinations exactly once. Completed neutral/prose cells stay out.
    from itertools import product
    pending = [tuple(row.split(':')[1:]) for row in rows[5:]]
    expected = {(arm, penalty, variant, report)
                for (arm, penalty), (variant, report) in product(
                    [('baseline', '0'), ('report_correct', '0'), ('report_correct', '0.5')],
                    [('neutral', 'typed'), ('refute', 'prose'), ('refute', 'typed')])}
    assert len(pending) == len(set(pending)) == 9
    assert set(pending) == expected
    assert rows[4] == 'lambda0:report_correct:0:neutral:prose'


def test_the_runner_kills_after_the_pinned_wait():
    """ops/remote_job.sh is also sent over SSH on stdin, where it cannot source ops/pod_pins.env, so it writes the value
    out; it must equal the pin the job scripts' inner limits use (review round 28)."""
    runner = (ROOT/'ops/remote_job.sh').read_text()
    assert f"KILL_AFTER_SECONDS={PINS['KILL_AFTER_SECONDS']}\n" in runner
    assert '--kill-after="$KILL_AFTER_SECONDS"' in runner and '$(($2+KILL_AFTER_SECONDS+GRACE))' in runner


def test_qwen_rollout_command_matches_reference_shape():
    """Capture the real rollout argv without a GPU. The 8k reference used 4x4 draws and a 10,240-token context."""
    import subprocess

    def command(environment, limit, generations):
        shell = """source ops/pod_lib.sh
in_env() { printf '%s\\n' "$VLLM_SERVER_DEV_MODE" "$VLLM_ALLOW_RUNTIME_LORA_UPDATING" "$@"; }
serve_rollouts "$1" 1 "$2" "$3"
"""
        return subprocess.run(['bash', '-c', shell, 'test', environment, str(limit), str(generations)],
                              cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()

    reference = command('vllm', 8192, int(PINS['DEFAULT_TRAIN_GENERATIONS']))
    replay = command('probe', 16384, int(PINS['DEFAULT_TRAIN_GENERATIONS']))
    assert reference[:6] == ['1', 'True', 'vllm', 'vllm', 'serve', PINS['QWEN3_32B']]
    assert reference[reference.index('--max-num-seqs') + 1] == '16'
    context = reference.index('--max-model-len') + 1
    assert reference[context] == '10240'
    reference[2], reference[context] = 'probe', '18432'
    assert replay == reference
    smaller = command('vllm', 8192, 2)
    assert smaller[smaller.index('--max-num-seqs') + 1] == '8'


def test_every_part_one_job_names_its_model():
    """Round 33: a job added to PART_ONE_RUNS without a PART_ONE_MODEL row would have read as model 0 (gpt-oss-20b)."""
    import subprocess
    lengths = subprocess.run(['bash', '-c', 'source ops/pod_pins.env; echo ${#PART_ONE_RUNS[@]} ${#PART_ONE_MODEL[@]} '
                              '${#MODEL_REPOS[@]} ${#MODEL_REVS[@]} ${#MODEL_ARGS[@]} ${#MODEL_GPUS[@]} ${#MODEL_ROLLOUT_FRACTION[@]} '
                              '${#MODEL_WORKSPACE_GB[@]} ${#MODEL_DOWNLOAD_LIMITS[@]} ${#MODEL_SETUP_LIMITS[@]}; '
                              'printf "%s\\n" "${PART_ONE_MODEL[@]}"'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.split('\n')
    jobs, models, *columns = map(int, lengths[0].split())
    assert jobs == models and len(set(columns)) == 1, lengths[0]
    assert all(0 <= int(index) < columns[0] for index in lengths[1:-1])
