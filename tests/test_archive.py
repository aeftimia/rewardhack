"""The run archive on S3 (ops/run_archive.py, notes section 78) against a stand-in `aws` that keeps the bucket in a folder,
and the driver's settings for the Qwen3-32B job."""
import base64
import hashlib
import os
import shutil
import subprocess
import sys

import pytest
from conftest import PROJECT
from test_driver_refusals import drive

# The stand-in keeps each object as a file and its SHA-256 beside it, as S3 stores the checksum a PutObject carried. Like S3,
# put-object refuses a body whose SHA-256 is not the one given. FAKE_S3_CHANGE_ON_PUT names a file whose bytes change while it
# is read (S3 then refuses it); every put is logged to <FAKE_S3>.sent.
FAKE_AWS = r'''
import base64, hashlib, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ['--region', 'us-east-1'], args
args = args[2:]
root = Path(os.environ['FAKE_S3'])
def opt(name):
    return args[args.index(name)+1]
def path(bucket, key):
    return root/bucket/key
if args[:2] == ['s3api', 'put-object']:
    assert opt('--storage-class') == 'STANDARD_IA'
    target, body = path(opt('--bucket'), opt('--key')), Path(opt('--body')).read_bytes()
    if os.environ.get('FAKE_S3_CHANGE_ON_PUT') and opt('--body').endswith(os.environ['FAKE_S3_CHANGE_ON_PUT']):
        body = body[::-1] + b'x'
    if base64.b64encode(hashlib.sha256(body).digest()).decode() != opt('--checksum-sha256'):
        sys.exit('An error occurred (BadDigest) when calling the PutObject operation')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    Path(str(target)+'.sha256').write_text(opt('--checksum-sha256'))
    with open(str(root)+'.sent', 'a') as log:
        log.write(opt('--key').split('/', 2)[2] + '\n')
elif args[:2] == ['s3api', 'head-object']:
    assert opt('--checksum-mode') == 'ENABLED'
    checksum = Path(str(path(opt('--bucket'), opt('--key')))+'.sha256')
    if not checksum.exists():
        sys.exit('An error occurred (404) when calling the HeadObject operation: Not Found')
    print(checksum.read_text())
elif args[:2] == ['s3', 'presign']:
    print(f'https://example.invalid/{args[2].removeprefix("s3://")}?expires={opt("--expires-in")}')
else:
    sys.exit(f'unexpected aws call {args}')
'''


def archive(tmp_path, *args, **env):
    bin_dir = tmp_path/'bin'
    if not bin_dir.exists():
        bin_dir.mkdir()
        (bin_dir/'fake_aws.py').write_text(FAKE_AWS)
        (bin_dir/'aws').write_text(f'#!/bin/bash\nexec "{sys.executable}" "{bin_dir}/fake_aws.py" "$@"\n')
        (bin_dir/'aws').chmod(0o755)
    result = subprocess.run([sys.executable, str(PROJECT/'ops'/'run_archive.py'), *map(str, args)], capture_output=True, text=True,
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


def stored(tmp_path, name='runpod-part1-test'):
    return tmp_path/'s3'/'rewardhack-run-archive'/'results'/name


def sent(tmp_path):
    log = tmp_path/'s3.sent'
    return sorted(log.read_text().split()) if log.exists() else []


def test_an_upload_keeps_the_layout_and_reads_back_every_checksum(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run)
    assert code == 0, output
    assert 'all 4 files are in S3' in output and (run/'ARCHIVED').exists()
    assert (stored(tmp_path)/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').exists()
    assert not (stored(tmp_path)/'known_hosts').exists()


def test_a_relaunch_that_appends_to_logs_sends_only_what_changed(tmp_path, run):
    """Round 38: a relaunch rewrites the driver's logs; that is not another run, and archived bytes are not sent again."""
    (run/'driver.log').write_text('first launch\n')
    assert archive(tmp_path, 'upload', run)[0] == 0
    (tmp_path/'s3.sent').unlink()
    (run/'driver.log').write_text('first launch\nrelaunch\n')
    code, output = archive(tmp_path, 'upload', run)
    assert code == 0, output
    assert sent(tmp_path) == ['driver.log'] and (stored(tmp_path)/'driver.log').read_text().endswith('relaunch\n')


def test_a_file_changing_while_it_is_read_is_refused_by_s3_and_nothing_is_marked(tmp_path, run):
    """Round 38: S3 checks the body against the SHA-256 the request carries."""
    code, output = archive(tmp_path, 'upload', run, FAKE_S3_CHANGE_ON_PUT='checkpoint-60/weights.bin')
    assert code != 0 and 'BadDigest' in output and not (run/'ARCHIVED').exists()
    assert not (stored(tmp_path)/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').exists()


def test_another_run_of_the_same_name_is_refused_before_it_sends_anything(tmp_path, run):
    """Rounds 35-37: a same-named folder or a reused path must not replace archived checkpoints."""
    assert archive(tmp_path, 'upload', run)[0] == 0 and archive(tmp_path, 'prune', run)[0] == 0
    (tmp_path/'s3.sent').unlink()
    shutil.rmtree(run)
    (run/'train-a'/'adapter-step8').mkdir(parents=True)
    (run/'train-a'/'adapter-step8'/'weights.bin').write_bytes(b'\x09' * 1000)
    code, output = archive(tmp_path, 'upload', run)
    assert code == 1 and 'another run of the same name' in output and sent(tmp_path) == []
    assert (stored(tmp_path)/'train-a'/'adapter-step8'/'weights.bin').read_bytes() == b'\x01' * 1000


def test_a_moved_folder_is_the_same_run(tmp_path, run):
    """Round 37: a run is identified by its files, not its path."""
    assert archive(tmp_path, 'upload', run)[0] == 0
    moved = tmp_path/'main-checkout'/'results'/run.name
    moved.parent.mkdir(parents=True)
    run.rename(moved)
    assert archive(tmp_path, 'upload', moved)[0] == 0
    assert archive(tmp_path, 'prune', moved)[0] == 0 and not (moved/'train-a'/'adapter-step8').exists()


def test_prune_deletes_only_the_trainers_snapshots_and_checkpoints(tmp_path, run):
    code, output = archive(tmp_path, 'prune', run)
    assert code == 1 and 'upload it first' in output and (run/'train-a'/'adapter-step8').exists()
    (run/'train-a'/'notes'/'adapter-step9').mkdir(parents=True)  # not where the trainer writes snapshots
    (run/'train-a'/'notes'/'adapter-step9'/'n.txt').write_text('keep')
    (run/'train-a'/'adapter-step8-final').mkdir()  # not digits only
    (run/'train-a'/'adapter-step8-final'/'w.bin').write_bytes(b'\x02')
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, output = archive(tmp_path, 'prune', run)
    assert code == 0, output
    assert not (run/'train-a'/'adapter-step8').exists() and not (run/'train-a'/'trainer'/'checkpoint-60').exists()
    for kept in ('adapter/weights.bin', 'steps.jsonl', 'notes/adapter-step9/n.txt', 'adapter-step8-final/w.bin'):
        assert (run/'train-a'/kept).exists(), kept


def test_prune_keeps_what_s3_does_not_hold_exactly(tmp_path, run):
    """Rounds 33-38: whatever replaced an object, prune compares S3's stored SHA-256 with the file here, folder by folder."""
    assert archive(tmp_path, 'upload', run)[0] == 0
    stored_file = stored(tmp_path)/'train-a'/'adapter-step8'/'weights.bin'
    stored_file.write_bytes(b'\x07' * 1000)
    (stored_file.parent/'weights.bin.sha256').write_text(base64.b64encode(hashlib.sha256(b'\x07' * 1000).digest()).decode())
    code, output = archive(tmp_path, 'prune', run)
    assert code == 1 and 'adapter-step8 is kept' in output and (run/'train-a'/'adapter-step8'/'weights.bin').exists()


def test_prune_keeps_a_file_written_after_the_upload(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    (run/'train-a'/'trainer'/'checkpoint-60'/'late.bin').write_bytes(b'\x05')
    code, output = archive(tmp_path, 'prune', run)
    assert code == 1 and 'checkpoint-60/late.bin' in output and (run/'train-a'/'trainer'/'checkpoint-60'/'late.bin').exists()


def test_prune_does_not_follow_a_link_out_of_the_run(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    elsewhere = tmp_path/'elsewhere'
    (run/'train-a').rename(elsewhere)
    (run/'train-a').symlink_to(elsewhere)
    code, output = archive(tmp_path, 'prune', run)
    assert code == 1 and 'symbolic link' in output and (elsewhere/'adapter-step8').exists()


def test_a_run_kept_under_a_checkpoint_path_keeps_its_results(tmp_path):
    run = tmp_path/'trainer'/'checkpoint-3'/'myrun'
    (run/'adapter').mkdir(parents=True)
    (run/'adapter'/'w.bin').write_bytes(b'\x01')
    (run/'steps.jsonl').write_text('{}')
    assert archive(tmp_path, 'upload', run)[0] == 0 and archive(tmp_path, 'prune', run)[0] == 0
    assert (run/'steps.jsonl').exists() and (run/'adapter'/'w.bin').exists()


def test_a_symbolic_link_in_the_run_is_refused(tmp_path, run):
    (run/'link').symlink_to(run/'train-a'/'steps.jsonl')
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and 'symbolic links' in output


def test_presign_names_the_run_and_its_expiry(tmp_path):
    code, output = archive(tmp_path, 'presign', 'runpod-part1-test', 'train-a/trainer/checkpoint-60/weights.bin')
    assert code == 0 and 'results/runpod-part1-test/train-a/trainer/checkpoint-60/weights.bin?expires=3600' in output


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


@pytest.mark.parametrize('settings, deadline', [(QWEN_PASS, '128'), (dict(QWEN_PASS, SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='12000'), '285')])
def test_qwen_launches_pass_every_check(tmp_path, settings, deadline):
    """The measuring pass at its 128-minute deadline; the 60-step run, 71 + (12,000 + 810)/60 = 284.5 of 285 minutes."""
    code, output, reached = drive(tmp_path, deadline=deadline, **settings)
    assert code != 2 and 'invalid' not in output and 'does not read' not in output and reached
