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
        assert 'vllm==' not in text, f'{script.name} pins vLLM: install from requirements-vllm.txt'


def test_the_launch_gate_counts_what_the_driver_and_runner_actually_do():
    """gates.py's worst case uses the provisional timer, the default grace and the default judge cap; the scripts that
    set them must use the same numbers, or the budget check undercounts (DRY audit, 26 Nov)."""
    driver = (ROOT/'ops/drive_runpod.sh').read_text()
    runner = (ROOT/'ops/remote_job.sh').read_text()
    assert f'--arm-only {gates.PROVISIONAL_MINUTES*60}' in driver
    assert f'AUTOSTOP_GRACE_SECONDS:-{gates.DRIVER_GRACE_SECONDS}}}' in runner
    assert driver.count(f'JUDGE_MAX_USD:-{gates.DRIVER_JUDGE_MAX_USD:g}}}') == driver.count('JUDGE_MAX_USD:-') > 0
