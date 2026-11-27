"""Execute the deployed stage loop through the real CLI's argument validation."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('job', range(4, 14))
def test_screening_jobs_validate_with_their_report_judge(job, tmp_path, monkeypatch):
    script = (ROOT/'ops/remote_runpod_smoke.sh').read_text()
    start = script.index('  for run in "${RUNS[@]}"; do')
    end = script.index('\n  done', start) + len('\n  done')
    shell = '''source ops/pod_pins.env
read -ra RUNS <<< "${PART_ONE_RUNS[$1]}"
TRAIN=(--steps 60)
step() { shift; printf '%s\\n' "$@"; }
''' + script[start:end]
    argv = subprocess.run(['bash', '-c', shell, 'screening-test', str(job)], cwd=ROOT,
                          check=True, text=True, capture_output=True).stdout.splitlines()
    spec = importlib.util.spec_from_file_location('screening_run', ROOT/'entrypoints/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = []
    monkeypatch.setattr(module, 'run', parsed.append)
    # BASE puts the API judge before stage arguments. Typed cells must override it
    # explicitly; removing that override must fail actual CLI validation here.
    monkeypatch.setattr(sys, 'argv', ['run.py', 'train', '--out', str(tmp_path), '--lean', 'none',
                                    '--judge', 'api', '--judge-model', 'gpt-5.4-nano',
                                    '--judge-max-usd', '1', *argv])
    module.main()
    [args] = parsed
    assert args.judge == ('none' if args.report_mode == 'typed' else 'api')
    assert args.steps == 60 and not args.stop_saturated
