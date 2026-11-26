"""Review round 30's minor follow-ups (notes section 77): a probe stage's failure says what its summary says, and a
model download is retried once within its limit. The gate's worst case is in test_gates.py, the runner's pids file in
the relaunch and termination tests."""
import json
import os
import subprocess
import sys
from pathlib import Path

from rewardhack import probe_rules

ROOT = Path(__file__).resolve().parents[1]


def test_a_failed_stage_says_whether_it_was_stopped_at_its_limit_and_what_it_kept(tmp_path):
    summary = tmp_path/'summary.json'
    assert 'ended before writing one' in probe_rules.stage(summary)
    summary.write_text(json.dumps(dict(status='running', argv=[])))
    assert 'rows written is unknown' in probe_rules.stage(summary)
    summary.write_text(json.dumps(dict(status='failed', rows_written=240, error='HardDeadline: hard deadline')))
    assert probe_rules.stage(summary) == 'status=failed rows_written=240 stopped at its --max-seconds: HardDeadline: hard deadline'
    summary.write_text(json.dumps(dict(status='failed', rows_written=0, error='RuntimeError: CUDA out of memory')))
    assert 'ended by an error: RuntimeError' in probe_rules.stage(summary)
    summary.write_text(json.dumps(dict(status='completed', rows_written=320)))
    assert probe_rules.stage(summary).endswith('no error recorded')
    done = subprocess.run([sys.executable, '-m', 'rewardhack.probe_rules', 'stage', str(summary)], capture_output=True, text=True, check=True,
                          env=dict(os.environ, PYTHONPATH=str(ROOT/'src')))
    assert done.stdout.strip() == probe_rules.stage(summary)


def fetch(tmp_path, hf_body, wait):
    """fetch_model from ops/pod_lib.sh with a stub `hf`; returns (exit status, the job's log lines)."""
    hf = tmp_path/'hf'
    hf.write_text('#!/bin/bash\n' + hf_body)
    hf.chmod(0o755)
    (tmp_path/'out').mkdir(exist_ok=True)
    script = f'source ops/pod_lib.sh; log() {{ echo "$*" >> {tmp_path}/steps.log; }}; HF={hf}; DOWNLOAD_WAIT={wait} fetch_model org/model rev'
    done = subprocess.run(['bash', '-c', script], cwd=ROOT, env=dict(os.environ, HOME=str(tmp_path)), capture_output=True, text=True)
    steps = (tmp_path/'steps.log').read_text().splitlines() if (tmp_path/'steps.log').exists() else []
    return done.returncode, steps


def test_a_failed_download_is_retried_once_within_its_limit(tmp_path):
    count = tmp_path/'attempts'
    flaky = f'echo attempt >> {count}\n[[ $(wc -l < {count}) -ge 2 ]] || {{ echo "File reconstruction error"; exit 1; }}\n'
    code, steps = fetch(tmp_path, flaky, 60)
    assert code == 0 and len(count.read_text().splitlines()) == 2
    assert len(steps) == 1 and steps[0].startswith('retry download org/model')
    assert 'File reconstruction error' in (tmp_path/'out'/'download-model.log').read_text()  # the first attempt's log is kept
    count.unlink()
    code, _ = fetch(tmp_path, f'echo attempt >> {count}\nexit 1\n', 60)
    assert code == 1 and len(count.read_text().splitlines()) == 2  # once, not more


def test_a_download_that_used_its_whole_limit_is_not_retried(tmp_path):
    count = tmp_path/'attempts'
    code, steps = fetch(tmp_path, f'echo attempt >> {count}\nsleep 30\n', 2)
    assert code != 0 and len(count.read_text().splitlines()) == 1
    assert steps == ['download org/model used its whole limit of 2 s (see ' + str(tmp_path) + '/out/download-model.log)']
