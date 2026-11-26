"""The run archive on S3 (ops/run_archive.py: upload and checksum check only, owner 26 Nov, notes section 78) against a stand-in `aws` that keeps the bucket in a folder,
and the driver's settings for the Qwen3-32B job."""
import json
import os
import re
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
    if os.environ.get('FAKE_ADD_FILE'):
        Path(os.environ['FAKE_ADD_FILE']).write_text('late')
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
elif args[:2] == ['configure', 'export-credentials']:
    import datetime, json
    left = int(os.environ.get('FAKE_SESSION_SECONDS', '28800'))
    expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=left)
    print(json.dumps({'Version': 1, 'Expiration': expiry.isoformat().replace('+00:00', 'Z')}))
elif args[:2] == ['s3api', 'get-object']:
    source = path(opt('--bucket'), opt('--key'))
    if not source.exists():
        sys.exit('An error occurred (NoSuchKey) when calling the GetObject operation')
    Path(args[-1]).write_bytes(source.read_bytes())
elif args[:2] == ['s3', 'presign']:
    if os.environ.get('FAKE_PRESIGN_FAIL') and args[2].endswith(os.environ['FAKE_PRESIGN_FAIL']):
        sys.exit('presign failed')
    print(f'https://example.invalid/{args[2].removeprefix("s3://")}?expires={opt("--expires-in")}')
else:
    sys.exit(f'unexpected aws call {args}')
'''


COMMIT = 'a' * 40


def archive(tmp_path, *args, **env):
    bin_dir = tmp_path/'bin'
    if not bin_dir.exists():
        bin_dir.mkdir()
        (bin_dir/'fake_aws.py').write_text(FAKE_AWS)
        (bin_dir/'aws').write_text(f'#!/bin/bash\nexec "{sys.executable}" "{bin_dir}/fake_aws.py" "$@"\n')
        (bin_dir/'aws').chmod(0o755)
        (bin_dir/'python3').write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
        (bin_dir/'python3').chmod(0o755)
        # The stand-in's URLs name its bucket folder: this curl copies from there.
        (bin_dir/'curl').write_text(f"""#!/bin/bash
out=; url=; while [[ $# -gt 0 ]]; do case $1 in -o) out=$2; shift 2;; --max-time) shift 2;; -*) shift;; *) url=$1; shift;; esac; done
src={tmp_path}/s3/${{url#https://example.invalid/}}; cp "${{src%%\\?*}}" "$out"
""")
        (bin_dir/'curl').chmod(0o755)
    result = subprocess.run([sys.executable, str(PROJECT/'ops'/'run_archive.py'), *map(str, args)], capture_output=True, text=True,
                            env=dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE_S3=str(tmp_path/'s3'), **env))
    return result.returncode, result.stdout + result.stderr


def fetch(tmp_path, destination, lines):
    env = dict(os.environ, PATH=f"{tmp_path/'bin'}:{os.environ['PATH']}")
    done = subprocess.run(['bash', str(PROJECT/'ops'/'fetch_checkpoint.sh'), str(destination)], input=lines, text=True,
                          capture_output=True, env=env)
    return done.returncode, done.stdout + done.stderr


@pytest.fixture
def run(tmp_path):
    run = tmp_path/'runpod-part1-test'
    for folder in ('train-a/adapter-step8', 'train-a/adapter', 'train-a/trainer/checkpoint-60'):
        (run/folder).mkdir(parents=True)
        (run/folder/'weights.bin').write_bytes(b'\x01' * 1000)
    for part in ('optimizer.pt', 'adapter_model.safetensors', 'scheduler.pt', 'trainer_state.json'):
        (run/'train-a'/'trainer'/'checkpoint-60'/part).write_bytes(b'\x03' * 1000)
    (run/'train-a'/'steps.jsonl').write_text('{"step": 1}\n')
    (run/'known_hosts').write_text('pod key\n')
    (run/'launch-commit.txt').write_text(COMMIT + '\n')
    return run


def records(run):
    return json.loads((run/'archive.json').read_text())


def stored(tmp_path, entry):
    return tmp_path/'s3'/entry['uri'].removeprefix('s3://')


def test_an_upload_gets_its_own_run_id_and_a_committed_record(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run)
    assert code == 0, output
    [entry] = records(run)
    assert re.fullmatch(r'\d{8}T\d{6}Z-runpod-part1-test-[0-9a-f]{8}', entry['run_id'])
    assert entry['uri'] == f"s3://rewardhack-run-archive/{entry['run_id']}/"
    assert entry['storage_class'] == 'STANDARD_IA' and entry['commit'] == COMMIT
    assert 'launch-commit.txt' in entry['sha256'] and 'known_hosts' not in entry['sha256']
    assert (stored(tmp_path, entry)/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').read_bytes() == b'\x01' * 1000
    assert json.loads((stored(tmp_path, entry)/'archive.json').read_text()) == entry
    assert (run/'train-a'/'adapter-step8').exists()  # nothing deleted


def test_uploads_never_share_a_prefix(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    other = tmp_path/'elsewhere'/run.name
    shutil.copytree(run, other)
    (other/'archive.json').unlink()
    (other/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').write_bytes(b'\x09' * 1000)
    assert archive(tmp_path, 'upload', other)[0] == 0 and archive(tmp_path, 'upload', run)[0] == 0
    ids = [e['run_id'] for e in records(run)] + [records(other)[0]['run_id']]
    assert len(set(ids)) == 3
    assert (stored(tmp_path, records(run)[0])/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').read_bytes() == b'\x01' * 1000


def test_an_empty_commit_record_is_refused(tmp_path, run):
    (run/'launch-commit.txt').write_text('')
    code, output = archive(tmp_path, 'upload', run, '--commit', 'b' * 40)
    assert code != 0 and 'does not hold a full commit' in output


def test_the_commit_is_required(tmp_path, run):
    (run/'launch-commit.txt').unlink()
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and '--commit' in output and not (tmp_path/'s3').exists()
    assert archive(tmp_path, 'upload', run, '--commit', 'b' * 40)[0] == 0 and records(run)[0]['commit'] == 'b' * 40


def test_a_file_changing_while_it_is_read_is_refused_by_s3_and_nothing_is_recorded(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run, FAKE_S3_CHANGE_ON_PUT='checkpoint-60/weights.bin')
    assert code != 0 and 'BadDigest' in output and not (run/'archive.json').exists()


def test_a_file_added_during_the_upload_stops_the_record(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run, FAKE_ADD_FILE=str(run/'train-a'/'late.json'))
    assert code != 0 and 'changed while it was archived' in output and not (run/'archive.json').exists()


@pytest.mark.parametrize('odd', ['link', 'spaced name.txt'])
def test_links_and_spaced_names_are_refused(tmp_path, run, odd):
    if odd == 'link':
        (run/odd).symlink_to(run/'train-a'/'steps.jsonl')
    else:
        (run/odd).write_text('x')
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and not (run/'archive.json').exists()


def test_a_bad_record_stops_an_upload_before_anything_is_sent(tmp_path, run):
    (run/'archive.json').write_text('{}')
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and 'not a list of archives' in output and not (tmp_path/'s3').exists()


def test_resume_by_run_id_or_record_presigns_only_the_chosen_checkpoint(tmp_path, run):
    """Owner, 26 Nov: Standard-IA charges per GB read, so a resume pulls one checkpoint, never snapshots or other checkpoints."""
    for step in ('56', '600'):
        (run/'train-a'/'trainer'/f'checkpoint-{step}').mkdir()
        (run/'train-a'/'trainer'/f'checkpoint-{step}'/'weights.bin').write_bytes(b'\x02')
    assert archive(tmp_path, 'upload', run)[0] == 0
    entry = records(run)[-1]
    for source in (entry['run_id'], run/'archive.json'):
        code, output = archive(tmp_path, 'resume', source, 'train-a/trainer/checkpoint-60')
        assert code == 0, output
        lines = output.strip().splitlines()
        assert [line.split()[1] for line in lines[:-1]] == [f'train-a/trainer/checkpoint-60/{part}' for part in (
            'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'weights.bin')]
        assert lines[-1] == 'END 5' and all(f"{entry['run_id']}/" in line.split()[2] for line in lines[:-1])
    # Round 44: checkpoint-56 lacks the files of a finished save (its run was killed mid-save, say).
    code, output = archive(tmp_path, 'resume', entry['run_id'], 'train-a/trainer/checkpoint-56')
    assert code != 0 and 'not a finished save' in output
    for refused in ('train-a/adapter-step8', 'train-a/trainer', 'train-a/trainer/checkpoint-7'):
        assert archive(tmp_path, 'resume', entry['run_id'], refused)[0] != 0, refused
    assert archive(tmp_path, 'resume', '20261126T000000Z-nothing-0123abcd', 'train-a/trainer/checkpoint-60')[0] != 0


def test_resume_prints_nothing_when_one_url_fails(tmp_path, run):
    """Round 42: all lines or none, so a cut-short list never reaches a pod."""
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, output = archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60', FAKE_PRESIGN_FAIL='weights.bin')
    assert code != 0 and 'https://' not in output


def test_resume_refuses_when_the_role_credentials_end_before_the_urls(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, output = archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60', FAKE_SESSION_SECONDS='600')
    assert code != 0 and 'aws sso logout' in output and 'https://' not in output


def test_the_pod_fetches_the_whole_checkpoint_and_checks_every_hash(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, lines = archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60')
    assert code == 0
    code, output = fetch(tmp_path, tmp_path/'pod', lines)
    assert code == 0, output
    assert (tmp_path/'pod'/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').read_bytes() == b'\x01' * 1000
    assert not (tmp_path/'pod'/'train-a'/'adapter-step8').exists()
    # Fetched again into the same place: refused, so two fetches never mix.
    assert fetch(tmp_path, tmp_path/'pod', lines)[0] == 2
    # A cut-short list, and a list with a file of another checkpoint, are refused before any download.
    body = lines.strip().splitlines()
    assert fetch(tmp_path, tmp_path/'pod2', '\n'.join(body[:1]) + '\n')[0] == 2
    other = body[0].replace('checkpoint-60', 'checkpoint-56')
    assert fetch(tmp_path, tmp_path/'pod3', '\n'.join([body[0], other, 'END 2']) + '\n')[0] == 2
    # Two lists pasted together are refused (round 43).
    assert fetch(tmp_path, tmp_path/'pod5', lines + lines)[0] == 2
    # A stored file with other bytes is never kept.
    (stored(tmp_path, records(run)[-1])/'train-a'/'trainer'/'checkpoint-60'/'weights.bin').write_bytes(b'\x00')
    code, output = fetch(tmp_path, tmp_path/'pod4', lines)
    assert code == 1 and 'SHA-256 differs' in output
    # Round 43: nothing of a failed fetch stays, not the files that did arrive, nor the temporary folder.
    assert list((tmp_path/'pod4').iterdir()) == []


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
