"""The run archive on S3 (ops/s3_archive.sh, notes section 78) against a stand-in `aws` that keeps the bucket in a folder,
and the driver's settings for the Qwen3-32B job."""
import os
import subprocess
import sys

import pytest
from conftest import PROJECT
from test_driver_refusals import drive

# The stand-in: `s3 cp --recursive` copies files (minus --exclude names at the top), `s3api list-objects-v2` prints key and size,
# `s3 presign` prints a URL. FAKE_S3 is the folder holding the bucket.
FAKE_AWS = r'''
import os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ['--region', 'us-east-1'], args
args = args[2:]
root = Path(os.environ['FAKE_S3'])
def path(uri):
    return root / uri.removeprefix('s3://')
if args[:3] == ['s3', 'cp', '--recursive']:
    source, target = Path(args[3]), path(args[4])
    assert '--storage-class' in args and args[args.index('--storage-class')+1] == 'STANDARD_IA'
    assert args[args.index('--checksum-algorithm')+1] == 'SHA256'
    excluded = {args[i+1] for i, a in enumerate(args) if a == '--exclude'}
    for file in source.rglob('*'):
        name = file.relative_to(source).as_posix()
        if file.is_file() and name not in excluded and not os.environ.get('FAKE_S3_DROP') == name:
            (target/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, target/name)
elif args[:2] == ['s3api', 'list-objects-v2']:
    bucket, prefix = args[args.index('--bucket')+1], args[args.index('--prefix')+1]
    files = sorted(f for f in (root/bucket).rglob('*') if f.is_file() and f.relative_to(root/bucket).as_posix().startswith(prefix))
    print('\n'.join(f'{f.relative_to(root/bucket).as_posix()}\t{f.stat().st_size}' for f in files) or 'None')
elif args[:2] == ['s3', 'presign']:
    print(f'https://example.invalid/{args[2].removeprefix("s3://")}?expires={args[args.index("--expires-in")+1]}')
else:
    sys.exit(f'unexpected aws call {args}')
'''


def archive(tmp_path, *args, **env):
    bin_dir = tmp_path/'bin'
    if not bin_dir.exists():
        bin_dir.mkdir()
        (bin_dir/'fake_aws.py').write_text(FAKE_AWS)
        (bin_dir/'aws').write_text(f'#!/bin/bash\nexec "{sys.executable}" "{bin_dir}/fake_aws.py" "$@"\n')
        (bin_dir/'python3').write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
        for name in ('aws', 'python3'):
            (bin_dir/name).chmod(0o755)
    result = subprocess.run(['bash', str(PROJECT/'ops'/'s3_archive.sh'), *map(str, args)], capture_output=True, text=True,
                            env=dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE_S3=str(tmp_path/'s3'), **env))
    return result.returncode, result.stdout + result.stderr


@pytest.fixture
def run(tmp_path):
    run = tmp_path/'runpod-part1-test'
    for folder in ('train-a/adapter-step8', 'train-a/adapter', 'train-a/trainer/checkpoint-60'):
        (run/folder).mkdir(parents=True)
        (run/folder/'weights.bin').write_bytes(b'\x01' * 1000)
    (run/'train-a'/'steps.jsonl').write_text('{"step": 1}\n')
    (run/'known_hosts').write_text('pod key\n')
    return run


def stored(tmp_path):
    return tmp_path/'s3'/'rewardhack-run-archive'/'results'/'runpod-part1-test'


def test_an_upload_keeps_the_layout_and_checks_every_file(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run)
    assert code == 0, output
    assert 'all 5 files are in' in output and (run/'ARCHIVED').exists()
    assert (stored(tmp_path)/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').exists()
    assert (stored(tmp_path)/'SHA256SUMS').exists() and not (stored(tmp_path)/'known_hosts').exists()
    # A second upload (more files since) rewrites the manifest and passes again: the manifest is never listed in itself.
    (run/'train-a'/'late.json').write_text('{}')
    code, output = archive(tmp_path, 'upload', run)
    assert code == 0 and 'all 6 files are in' in output, output


def test_a_file_that_did_not_arrive_fails_the_upload_and_nothing_is_marked(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run, FAKE_S3_DROP='train-a/steps.jsonl')
    assert code == 1 and 'train-a/steps.jsonl' in output and not (run/'ARCHIVED').exists()


def test_prune_needs_an_archived_run_and_keeps_the_results(tmp_path, run):
    code, output = archive(tmp_path, 'prune', run)
    assert code == 2 and 'upload it first' in output and (run/'train-a'/'adapter-step8').exists()
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, output = archive(tmp_path, 'prune', run)
    assert code == 0, output
    assert not (run/'train-a'/'adapter-step8').exists() and not (run/'train-a'/'trainer'/'checkpoint-60').exists()
    assert (run/'train-a'/'adapter'/'weights.bin').exists() and (run/'train-a'/'steps.jsonl').exists()


def test_prune_refuses_when_the_bucket_lost_a_file(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    (stored(tmp_path)/'train-a'/'adapter-step8'/'weights.bin').unlink()
    code, output = archive(tmp_path, 'prune', run)
    assert code == 1 and 'adapter-step8/weights.bin' in output and (run/'train-a'/'adapter-step8').exists()


def test_a_symbolic_link_is_refused(tmp_path, run):
    (run/'link').symlink_to(run/'train-a'/'steps.jsonl')
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and 'symbolic links' in output


def test_presign_names_the_run_and_its_expiry(tmp_path):
    code, output = archive(tmp_path, 'presign', 'runpod-part1-test', 'train-a/trainer/checkpoint-60/weights.bin')
    assert code == 0 and 'results/runpod-part1-test/train-a/trainer/checkpoint-60/weights.bin?expires=3600' in output


def manifest(*args):
    return subprocess.run([sys.executable, str(PROJECT/'ops'/'sha256_manifest.py'), *map(str, args)], capture_output=True, text=True)


def test_the_manifest_names_a_changed_or_missing_file(tmp_path, run):
    assert manifest('write', run, run/'SHA256SUMS').returncode == 0
    assert manifest('verify', run, run/'SHA256SUMS').returncode == 0
    (run/'train-a'/'steps.jsonl').write_text('{"step": 2}\n')
    (run/'train-a'/'adapter'/'weights.bin').unlink()
    checked = manifest('verify', run, run/'SHA256SUMS')
    assert checked.returncode == 1 and 'DIFFERS train-a/steps.jsonl' in checked.stdout and 'MISSING train-a/adapter/weights.bin' in checked.stdout


API = dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none',
           SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0', SMOKE_VLLM_ROLLOUTS='1', SMOKE_TOKEN_LIMIT='2048', SMOKE_FINAL_EVAL_SECONDS='600')
# The Qwen3-32B measuring pass (notes section 78): 6 steps, 2,520 s; 71 minutes of setup + (2,520 + 810) s = 126.5 min.
QWEN_PASS = dict(API, SMOKE_PART_ONE='4', SMOKE_STEPS='6', SMOKE_TRAIN_SECONDS='2520')


@pytest.mark.parametrize('settings, deadline, said', [
    (dict(SMOKE_PART_ONE='5'), '240', "SMOKE_PART_ONE must index a job in ops/pod_pins.env's PART_ONE_RUNS (0 to 4)"),
    (dict(QWEN_PASS, SMOKE_PROBE_LARGE='1'), '240', 'SMOKE_PROBE_LARGE=1 follows a one-card job'),
    (QWEN_PASS, '126', 'SMOKE_PART_ONE=4 needs a deadline of at least 127 minutes'),
    (dict(LEAN_ISOLATION='none', SMOKE_STEPS='4'), '169', 'the preflight run (SMOKE_PART_ONE unset or 0) needs a deadline of at least 170'),
])
def test_qwen_and_preflight_settings_are_refused_before_the_pod(tmp_path, settings, deadline, said):
    code, output, reached = drive(tmp_path, deadline=deadline, **{'LEAN_ISOLATION': 'none', **settings})
    assert code == 2 and said in output and not reached


@pytest.mark.parametrize('settings, deadline', [(QWEN_PASS, '127'), (dict(QWEN_PASS, SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='12000'), '285')])
def test_qwen_launches_pass_every_check(tmp_path, settings, deadline):
    """The measuring pass at 127 minutes; the 60-step run, 71 + (12,000 + 810)/60 = 284.5 of 285 minutes."""
    code, output, reached = drive(tmp_path, deadline=deadline, **settings)
    assert code != 2 and 'invalid' not in output and 'does not read' not in output and reached


def test_prune_refuses_a_local_copy_changed_since_the_upload(tmp_path, run):
    """Round 33: a file rewritten at the same size passes a listing by size; the local hashes are checked first."""
    assert archive(tmp_path, 'upload', run)[0] == 0
    (run/'train-a'/'adapter-step8'/'weights.bin').write_bytes(b'\x02' * 1000)
    code, output = archive(tmp_path, 'prune', run)
    assert code == 1 and 'changed since it was archived' in output and (run/'train-a'/'adapter-step8').exists()


def test_prune_never_deletes_the_run_folder_itself(tmp_path):
    """Round 33: a run folder named like a snapshot matched find's own starting point."""
    run = tmp_path/'adapter-step40'
    (run/'adapter-step8').mkdir(parents=True)
    (run/'adapter-step8'/'w.bin').write_bytes(b'\x01')
    (run/'steps.jsonl').write_text('{}')
    assert archive(tmp_path, 'upload', run)[0] == 0
    assert archive(tmp_path, 'prune', run)[0] == 0
    assert (run/'steps.jsonl').exists() and not (run/'adapter-step8').exists()
