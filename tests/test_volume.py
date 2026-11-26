"""The network volume (notes section 78): the upload of a results folder with its checksum check, and the driver's
settings for jobs that write there and for the Qwen3-32B job."""
import shutil
import subprocess
import sys

import pytest
from conftest import PROJECT
from test_driver_refusals import drive

API = dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none',
           SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0', SMOKE_VLLM_ROLLOUTS='1', SMOKE_TOKEN_LIMIT='2048', SMOKE_FINAL_EVAL_SECONDS='600')
# The Qwen3-32B measuring pass (notes section 78): 6 steps, 2,520 s for the run; 54 minutes of setup + (2,520 + 810) s = 110 min.
QWEN_PASS = dict(API, SMOKE_PART_ONE='4', SMOKE_STEPS='6', SMOKE_TRAIN_SECONDS='2520', SMOKE_VOLUME='1')


def manifest(*args):
    return subprocess.run([sys.executable, str(PROJECT/'ops'/'sha256_manifest.py'), *map(str, args)], capture_output=True, text=True)


def test_the_manifest_names_a_changed_or_missing_file(tmp_path):
    run = tmp_path/'run'
    (run/'train-a'/'trainer').mkdir(parents=True)
    (run/'train-a'/'steps.jsonl').write_text('{"step": 1}\n')
    (run/'train-a'/'trainer'/'optimizer.pt').write_bytes(b'\x00' * 4096)
    assert manifest('write', run, tmp_path/'SUMS').returncode == 0
    assert manifest('verify', run, tmp_path/'SUMS').returncode == 0
    (run/'train-a'/'steps.jsonl').write_text('{"step": 2}\n')
    (run/'train-a'/'trainer'/'optimizer.pt').unlink()
    checked = manifest('verify', run, tmp_path/'SUMS')
    assert checked.returncode == 1 and 'DIFFERS train-a/steps.jsonl' in checked.stdout and 'MISSING train-a/trainer/optimizer.pt' in checked.stdout


def test_an_empty_manifest_does_not_pass(tmp_path):
    (tmp_path/'run').mkdir()
    (tmp_path/'SUMS').write_text('')
    assert manifest('verify', tmp_path/'run', tmp_path/'SUMS').returncode == 1


def upload(tmp_path, mounted=True):
    """ops/volume_upload.sh against this machine: ssh runs the command here (the pod's timer and removal are recorded, not
    run), rsync is real, and the volume is a folder under tmp_path."""
    ops = tmp_path/'copy'/'ops'
    shutil.copytree(PROJECT/'ops', ops)
    volume = tmp_path/'volume'
    pins = (ops/'pod_pins.env').read_text().replace('VOLUME_MOUNT=/runs-volume', f'VOLUME_MOUNT={volume}')
    (ops/'pod_pins.env').write_text(pins)
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    stubs = {'ssh': f'''while [[ $1 == -* ]]; do case $1 in -p|-o) shift 2;; *) shift;; esac; done
shift
case "$*" in *--arm-only*|*--remove-now*) echo "$*" >> {tmp_path}/pod-calls; cat > /dev/null; exit 0;; esac
exec bash -c "$*"
''', 'mountpoint': f'exit {0 if mounted else 1}\n', 'python3': f'exec "{sys.executable}" "$@"\n'}
    for name, body in stubs.items():
        (bin_dir/name).write_text('#!/bin/bash\n'+body)
        (bin_dir/name).chmod(0o755)
    source = tmp_path/'runpod-part1-test'
    (source/'train-a'/'adapter-step8').mkdir(parents=True)
    (source/'train-a'/'adapter-step8'/'adapter_model.safetensors').write_bytes(b'\x01' * 100000)
    (source/'train-a'/'summary.json').write_text('{}')
    (source/'known_hosts').write_text('pod key\n')
    import os
    result = subprocess.run(['bash', str(ops/'volume_upload.sh'), 'example.invalid', '22', str(source)],
                            env=dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}"), capture_output=True, text=True)
    calls = (tmp_path/'pod-calls').read_text() if (tmp_path/'pod-calls').exists() else ''
    return result.returncode, result.stdout+result.stderr, calls, volume/'runs'/'runpod-part1-test'


def test_an_upload_keeps_the_layout_verifies_every_file_and_removes_the_pod(tmp_path):
    code, output, calls, dest = upload(tmp_path)
    assert code == 0, output
    assert 'verified' in output and '2 of 2 files match' in output
    assert (dest/'train-a'/'adapter-step8'/'adapter_model.safetensors').stat().st_size == 100000
    assert (dest/'SHA256SUMS').exists() and not (dest/'known_hosts').exists()
    assert calls.index('--arm-only') < calls.index('--remove-now')


def test_an_upload_to_a_pod_without_the_volume_sends_nothing_and_removes_the_pod(tmp_path):
    code, output, calls, dest = upload(tmp_path, mounted=False)
    assert code == 1 and 'has no volume' in output and not dest.exists()
    assert '--arm-only' in calls and '--remove-now' in calls


@pytest.mark.parametrize('settings, said', [
    (dict(SMOKE_VOLUME='2'), 'SMOKE_VOLUME must be 0 or 1'),
    (dict(SMOKE_PART_ONE='5'), "SMOKE_PART_ONE must index a job in ops/pod_pins.env's PART_ONE_RUNS (0 to 4)"),
    # The gpt-oss-120b probe needs the one card after a one-card job; Qwen3-32B takes two.
    (dict(QWEN_PASS, SMOKE_PROBE_LARGE='1'), 'SMOKE_PROBE_LARGE=1 follows a one-card job'),
    (dict(QWEN_PASS, deadline='109'), 'SMOKE_PART_ONE=4 needs a deadline of at least 110 minutes'),
])
def test_volume_and_qwen_settings_are_refused_before_the_pod(tmp_path, settings, said):
    settings = dict(settings)
    deadline = settings.pop('deadline', '240')
    code, output, reached = drive(tmp_path, deadline=deadline, **{'LEAN_ISOLATION': 'none', **settings})
    assert code == 2 and said in output and not reached


@pytest.mark.parametrize('settings', [QWEN_PASS, dict(QWEN_PASS, SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='12000'),
                                      dict(API, SMOKE_PART_ONE='2', SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='7680', SMOKE_VOLUME='1')])
def test_volume_launches_pass_every_check(tmp_path, settings):
    """The measuring pass at 110 minutes; the 60-step Qwen3-32B run (54 + (12,000 + 810)/60 = 268 of 270 minutes); a
    gpt-oss-20b job on the volume."""
    code, output, reached = drive(tmp_path, deadline='270' if settings['SMOKE_STEPS'] == '60' and settings['SMOKE_PART_ONE'] == '4' else '240', **settings)
    assert code != 2 and 'invalid' not in output and 'does not read' not in output and reached
