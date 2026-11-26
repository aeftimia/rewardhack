"""The run archive on S3 (ops/run_archive.py: upload and checksum check only, owner 26 Nov, notes section 78) against a stand-in `aws` that keeps the bucket in a folder,
and the driver's settings for the Qwen3-32B job."""
import json
import os
import shutil
import subprocess
import sys

import pytest
from conftest import PROJECT
from test_driver_refusals import drive

# The stand-in keeps each object as a file and its SHA-256 beside it, as S3 stores the checksum a PutObject carried. Like S3,
# put-object refuses a body whose SHA-256 is not the one given. FAKE_S3_CHANGE_ON_PUT names a file whose bytes change while it
# is read (S3 then refuses it).
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
    if opt('--if-none-match') == '*' and path(opt('--bucket'), opt('--key')).exists():
        sys.exit('An error occurred (PreconditionFailed) when calling the PutObject operation')
    target, body = path(opt('--bucket'), opt('--key')), Path(opt('--body')).read_bytes()
    if os.environ.get('FAKE_S3_CHANGE_ON_PUT') and opt('--body').endswith(os.environ['FAKE_S3_CHANGE_ON_PUT']):
        body = body[::-1] + b'x'
    if base64.b64encode(hashlib.sha256(body).digest()).decode() != opt('--checksum-sha256'):
        sys.exit('An error occurred (BadDigest) when calling the PutObject operation')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    Path(str(target)+'.sha256').write_text(opt('--checksum-sha256'))
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


def archives(run):
    return json.loads((run/'ARCHIVES.json').read_text())


def stored(tmp_path, entry):
    return tmp_path/'s3'/entry['uri'].removeprefix('s3://')


def test_an_upload_keeps_the_layout_under_its_own_id_and_records_every_hash(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run)
    assert code == 0, output
    [entry] = archives(run)
    assert entry['uri'] == f"s3://rewardhack-run-archive/results/runpod-part1-test/{entry['id']}/"
    assert sorted(entry['sha256']) == ['train-a/adapter-step8/weights.bin', 'train-a/adapter/weights.bin',
                                       'train-a/steps.jsonl', 'train-a/trainer/checkpoint-60/weights.bin']
    assert (stored(tmp_path, entry)/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').read_bytes() == b'\x01' * 1000
    assert not (stored(tmp_path, entry)/'known_hosts').exists()
    assert all(p.exists() for p in (run/'train-a'/'adapter-step8', run/'train-a'/'trainer'/'checkpoint-60'))  # nothing deleted


def test_a_later_upload_of_a_same_named_folder_never_touches_an_earlier_archive(tmp_path, run):
    """Owner, 26 Nov: each upload is keyed by its own id."""
    assert archive(tmp_path, 'upload', run)[0] == 0
    first = archives(run)[0]
    other = tmp_path/'elsewhere'/run.name
    shutil.copytree(run, other)
    (other/'ARCHIVES.json').unlink()
    (other/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').write_bytes(b'\x09' * 1000)
    assert archive(tmp_path, 'upload', other)[0] == 0
    assert archives(other)[0]['id'] != first['id']
    assert (stored(tmp_path, first)/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').read_bytes() == b'\x01' * 1000
    assert archive(tmp_path, 'upload', run)[0] == 0 and len(archives(run)) == 2


def test_a_file_changing_while_it_is_read_is_refused_by_s3_and_nothing_is_recorded(tmp_path, run):
    """S3 checks each body against the SHA-256 its request carries."""
    code, output = archive(tmp_path, 'upload', run, FAKE_S3_CHANGE_ON_PUT='checkpoint-60/weights.bin')
    assert code != 0 and 'BadDigest' in output and 'not recorded' in output and not (run/'ARCHIVES.json').exists()


def test_a_symbolic_link_in_the_run_is_refused(tmp_path, run):
    (run/'link').symlink_to(run/'train-a'/'steps.jsonl')
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and 'symbolic links' in output and not (run/'ARCHIVES.json').exists()


def test_presign_reads_the_latest_archive_and_refuses_unknown_files(tmp_path, run):
    code, output = archive(tmp_path, 'presign', run, 'train-a/steps.jsonl')
    assert code != 0 and 'upload it first' in output
    assert archive(tmp_path, 'upload', run)[0] == 0
    entry = archives(run)[-1]
    code, output = archive(tmp_path, 'presign', run, 'train-a/trainer/checkpoint-60/weights.bin')
    assert code == 0 and f"{entry['id']}/train-a/trainer/checkpoint-60/weights.bin?expires=3600" in output
    code, output = archive(tmp_path, 'presign', run, 'train-a/nothing.bin')
    assert code != 0 and 'not in archive' in output


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
