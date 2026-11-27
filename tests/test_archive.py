"""The run archive (ops/run_archive.py upload/resume/fetch; notes section 80) against stand-in
rclone, aws and curl that keep the bucket in a folder, and the driver's settings for the Qwen3-32B job."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from conftest import PROJECT
from test_driver_refusals import drive
from rewardhack import hf_grpo

# Stand-ins: the bucket is the folder FAKE_S3/<bucket>/. rclone copies (honouring the anchored --exclude filters and
# --immutable), checks, hashes and cats; FAKE_RCLONE_CHECK_FAILS makes `check` report a difference. aws presigns and reports
# the credentials' expiry (FAKE_SESSION_SECONDS from now).
FAKE_RCLONE = r"""
import hashlib, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ['FAKE_S3'])
assert os.environ['RCLONE_CONFIG_S3_ENV_AUTH'] == 'true'
def local_files(folder):
    excluded = {args[i+1].lstrip('/') for i, a in enumerate(args) if a == '--exclude'}
    return sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob('*') if p.is_file()
                  and p.relative_to(folder).as_posix() not in excluded)
def remote(spec):
    assert spec.startswith('s3:'), spec
    return root/spec[3:]
command = args[0]
if command == 'version':
    print(os.environ.get('FAKE_RCLONE_VERSION', 'rclone v1.76.0'))
elif command == 'hashsum':
    assert args[1] == 'sha256'
    if os.environ.get('FAKE_ADD_FILE') and (root/'copied').exists():  # a file that arrived while the copy ran
        Path(os.environ['FAKE_ADD_FILE']).write_text('late')
    for name in local_files(args[2]):
        print(hashlib.sha256((Path(args[2])/name).read_bytes()).hexdigest() + '  ' + name)
elif command in ('copy', 'copyto'):
    assert args[args.index('--s3-storage-class')+1] == 'STANDARD_IA' and '--immutable' in args
    pairs = [(Path(args[1])/n, remote(args[2])/n) for n in local_files(args[1])] if command == 'copy' else [(Path(args[1]), remote(args[2]))]
    for source, target in pairs:
        if target.exists():
            sys.exit(f'ERROR: {target}: immutable file modified')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    root.mkdir(exist_ok=True); (root/'copied').write_text('')
elif command == 'check':
    assert '--one-way' in args
    differ = [n for n in local_files(args[1]) if not (remote(args[2])/n).exists()
              or (remote(args[2])/n).read_bytes() != (Path(args[1])/n).read_bytes()]
    if differ or os.environ.get('FAKE_RCLONE_CHECK_FAILS'):
        sys.exit(f'ERROR: {len(differ) or 1} differences found')
elif command == 'cat':
    target = remote(args[1])
    if target.exists():  # like rclone, a missing object prints nothing and succeeds
        sys.stdout.write(target.read_text())
else:
    sys.exit(f'unexpected rclone call {args}')
"""
FAKE_AWS = r"""
import datetime, json, os, sys
args = sys.argv[1:]
if args[:2] == ['configure', 'export-credentials']:
    left = int(os.environ.get('FAKE_SESSION_SECONDS', '3600'))
    expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=left)
    print(json.dumps({'Version': 1, 'Expiration': expiry.isoformat()}))
elif args[2:4] == ['s3', 'presign']:
    print(f'https://example.invalid/{args[4].removeprefix("s3://")}?expires={args[args.index("--expires-in")+1]}')
else:
    sys.exit(f'unexpected aws call {args}')
"""
COMMIT = 'a' * 40


def tools(tmp_path):
    bin_dir = tmp_path/'bin'
    if not bin_dir.exists():
        bin_dir.mkdir()
        scripts = {'rclone': FAKE_RCLONE, 'aws': FAKE_AWS}
        for name, source in scripts.items():
            (bin_dir/f'{name}.py').write_text(source)
            (bin_dir/name).write_text(f'#!/bin/bash\nexec "{sys.executable}" "{bin_dir}/{name}.py" "$@"\n')
        (bin_dir/'curl').write_text(f"""#!/bin/bash
out=; url=; while [[ $# -gt 0 ]]; do case $1 in -o) out=$2; shift 2;; --max-time) shift 2;; -*) shift;; *) url=$1; shift;; esac; done
src={tmp_path}/s3/${{url#https://example.invalid/}}; cp "${{src%%\\?*}}" "$out"
""")
        for name in ('rclone', 'aws', 'curl'):
            (bin_dir/name).chmod(0o755)
    return dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE_S3=str(tmp_path/'s3'), PYTHONPATH=str(PROJECT/'src'))


def archive(tmp_path, *args, **env):
    done = subprocess.run([sys.executable, str(PROJECT/'ops'/'run_archive.py'), *map(str, args)], capture_output=True, text=True,
                          env={**tools(tmp_path), **env})
    return done.returncode, done.stdout + done.stderr


def fetch(tmp_path, destination, lines):
    done = subprocess.run([sys.executable, str(PROJECT/'ops'/'run_archive.py'), 'fetch', str(destination)], input=lines, text=True,
                          capture_output=True, env=tools(tmp_path))
    return done.returncode, done.stdout + done.stderr


CHECKPOINT_PARTS = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json')


@pytest.fixture
def run(tmp_path):
    run = tmp_path/'runpod-part1-test'
    for step in (56, 60):
        (run/'train-a'/'trainer'/f'checkpoint-{step}').mkdir(parents=True)
        for part in CHECKPOINT_PARTS:
            (run/'train-a'/'trainer'/f'checkpoint-{step}'/part).write_bytes(bytes([step]) * 100)
    (run/'train-a'/'adapter-step8').mkdir()
    (run/'train-a'/'adapter-step8'/'adapter_model.safetensors').write_bytes(b'\x01' * 100)
    for name in hf_grpo.RESUME_RUN_FILES:
        (run/'train-a'/name).write_text('{}\n')
    (run/'known_hosts').write_text('pod key\n')
    (run/'launch-commit.txt').write_text(COMMIT + '\n')
    return run


def records(run):
    return json.loads((run/'archive.json').read_text())


def test_upload_copies_checks_and_records_under_a_new_run_id(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run)
    assert code == 0, output
    [entry] = records(run)
    assert re.fullmatch(r'\d{8}T\d{6}Z-runpod-part1-test-[0-9a-f]{8}', entry['run_id'])
    assert entry['uri'] == f"s3://rewardhack-run-archive/{entry['run_id']}/"
    assert entry['storage_class'] == 'STANDARD_IA' and entry['commit'] == COMMIT
    assert 'known_hosts' not in entry['sha256'] and len(entry['sha256']['train-a/steps.jsonl']) == 64
    stored = tmp_path/'s3'/'rewardhack-run-archive'/entry['run_id']
    assert (stored/'train-a'/'trainer'/'checkpoint-60'/'optimizer.pt').exists() and not (stored/'known_hosts').exists()
    assert json.loads((stored/'archive.json').read_text()) == entry
    assert archive(tmp_path, 'upload', run)[0] == 0 and len({e['run_id'] for e in records(run)}) == 2


@pytest.mark.parametrize('env, said', [(dict(FAKE_RCLONE_CHECK_FAILS='1'), 'differences found'),
                                       (dict(FAKE_RCLONE_VERSION='rclone v1.70.0'), 'pins v1.76.0')])
def test_a_failed_check_or_another_rclone_records_nothing(tmp_path, run, env, said):
    code, output = archive(tmp_path, 'upload', run, **env)
    assert code != 0 and said in output and not (run/'archive.json').exists()


def test_a_folder_changing_during_the_upload_records_nothing(tmp_path, run):
    code, output = archive(tmp_path, 'upload', run, FAKE_ADD_FILE=str(run/'train-a'/'late.json'))
    assert code != 0 and 'changed while it was archived' in output and not (run/'archive.json').exists()


def test_a_symbolic_link_is_refused(tmp_path, run):
    (run/'link').symlink_to(run/'train-a'/'steps.jsonl')
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and 'symbolic links' in output and not (tmp_path/'s3').exists()


def test_the_commit_is_required_and_must_be_whole(tmp_path, run):
    (run/'launch-commit.txt').write_text('')
    assert archive(tmp_path, 'upload', run, '--commit', 'b' * 40)[0] != 0
    (run/'launch-commit.txt').unlink()
    code, output = archive(tmp_path, 'upload', run)
    assert code != 0 and '--commit' in output
    assert archive(tmp_path, 'upload', run, '--commit', 'b' * 40)[0] == 0 and records(run)[0]['commit'] == 'b' * 40


def test_resume_names_one_checkpoint_and_the_run_files_and_the_pod_checks_them(tmp_path, run):
    """Owner, 26 Nov: Standard-IA charges per GB read, so a resume reads one checkpoint and the run files the trainer's
    resume needs (round 45), never snapshots or other checkpoints; the pod keeps them only when every SHA-256 matches."""
    assert archive(tmp_path, 'upload', run)[0] == 0
    entry = records(run)[0]
    code, lines = archive(tmp_path, 'resume', entry['run_id'], 'train-a/trainer/checkpoint-60')
    assert code == 0, lines
    names = [name for _, name, _ in json.loads(lines)]
    assert names == sorted([f'train-a/trainer/checkpoint-60/{part}' for part in CHECKPOINT_PARTS]
                           + [f'train-a/{name}' for name in hf_grpo.RESUME_RUN_FILES])
    assert archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60')[1] == lines
    code, output = fetch(tmp_path, tmp_path/'pod', lines)
    assert code == 0, output
    assert (tmp_path/'pod'/'train-a'/'trainer'/'checkpoint-60'/'optimizer.pt').read_bytes() == bytes([60]) * 100
    assert (tmp_path/'pod'/'train-a'/'steps.jsonl').exists() and not (tmp_path/'pod'/'train-a'/'adapter-step8').exists()
    assert not (tmp_path/'pod'/'train-a'/'trainer'/'checkpoint-56').exists()
    assert fetch(tmp_path, tmp_path/'pod', lines)[0] == 2  # never on top of an earlier fetch
    assert fetch(tmp_path, tmp_path/'pod2', lines[:len(lines)//2])[0] == 2  # cut short
    assert fetch(tmp_path, tmp_path/'pod3', lines + lines)[0] == 2  # two lists
    stored = tmp_path/'s3'/'rewardhack-run-archive'/entry['run_id']/'train-a'/'trainer'/'checkpoint-60'/'optimizer.pt'
    stored.write_bytes(b'\x00')
    code, output = fetch(tmp_path, tmp_path/'pod4', lines)
    assert code == 2 and 'nothing kept' in output and list((tmp_path/'pod4').iterdir()) == []


def test_resume_refuses_what_it_cannot_resume(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    run_id = records(run)[0]['run_id']
    for checkpoint in ('train-a/adapter-step8', 'train-a/trainer', 'train-a/trainer/checkpoint-7'):
        assert archive(tmp_path, 'resume', run_id, checkpoint)[0] != 0, checkpoint
    (run/'train-a'/'trainer'/'checkpoint-56'/'rng_state.pth').unlink()  # a save cut short, uploaded as it was
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, output = archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-56')
    assert code != 0 and 'rng_state.pth' in output
    code, output = archive(tmp_path, 'resume', '20261126T000000Z-runpod-part1-test-0123abcd', 'train-a/trainer/checkpoint-60')
    assert code != 0 and 'no archive' in output
    code, output = archive(tmp_path, 'resume', run_id, 'train-a/trainer/checkpoint-60', FAKE_SESSION_SECONDS='600')
    assert code != 0 and 'aws sso logout' in output and 'https://' not in output


def test_fetched_checkpoint_passes_the_trainers_resume_checks(tmp_path, run, monkeypatch):
    """Archive selection must satisfy the trainer's actual reader, not just its list of filenames."""
    original = run/'train-a'
    args = SimpleNamespace(**dict.fromkeys(hf_grpo.RESUME_SETTINGS))
    vars(args).update(steps=61, num_generations=1, prompts_per_step=1, lr_schedule='constant',
                      resume_from=tmp_path/'pod'/'train-a'/'trainer'/'checkpoint-60', out=tmp_path/'continued')
    monkeypatch.setattr(hf_grpo.importlib.metadata, 'version', lambda name: 'test-version')
    (original/hf_grpo.RUN_SETTINGS).write_text(json.dumps(hf_grpo.run_settings(args)))
    (original/'summary.json').write_text(json.dumps({'status': 'completed', 'global_step': 60}))
    (original/'trainer'/'checkpoint-60'/'trainer_state.json').write_text(json.dumps({'global_step': 60, 'epoch': 1.0}))
    (original/'training-rollouts.jsonl').write_text('{}\n' * 60)
    (original/'steps.jsonl').write_text(''.join(json.dumps({'step': step}) + '\n' for step in range(1, 61)))
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, document = archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60')
    assert code == 0, document
    code, output = fetch(tmp_path, tmp_path/'pod', document)
    assert code == 0, output
    hf_grpo.check_resume(args)


@pytest.mark.parametrize('missing', ['summary.json', 'before-evaluation.jsonl'])
def test_archive_without_required_run_metadata_cannot_resume(tmp_path, run, missing):
    (run/'train-a'/missing).unlink()
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, output = archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60')
    assert code != 0 and missing in output


def test_existing_s3_record_and_objects_are_read_without_rewriting(tmp_path, run):
    """The archive format written before the JSON resume protocol; no new uploader creates this fixture."""
    run_id = '20261126T000000Z-runpod-part1-test-0123abcd'
    entry = dict(run_id=run_id, uri=f's3://rewardhack-run-archive/{run_id}/', storage_class='STANDARD_IA',
                 commit=COMMIT, uploaded_at='2026-11-26T00:00:00+00:00',
                 sha256={p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in run.rglob('*') if p.is_file() and p.name != 'known_hosts'})
    stored = tmp_path/'s3'/'rewardhack-run-archive'/run_id
    shutil.copytree(run, stored)
    (stored/'archive.json').write_text(json.dumps(entry, indent=1) + '\n')
    (run/'archive.json').write_text(json.dumps([entry], indent=1) + '\n')
    before = {p: p.read_bytes() for p in stored.rglob('*') if p.is_file()}
    local_before = (run/'archive.json').read_bytes()
    code, document = archive(tmp_path, 'resume', run_id, 'train-a/trainer/checkpoint-60')
    assert code == 0, document
    assert archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60') == (0, document)
    assert fetch(tmp_path, tmp_path/'pod', document)[0] == 0
    assert {p: p.read_bytes() for p in stored.rglob('*') if p.is_file()} == before
    assert (run/'archive.json').read_bytes() == local_before
    assert archive(tmp_path, 'upload', run)[0] == 0
    assert records(run)[0] == entry and len(records(run)) == 2
    assert {p: p.read_bytes() for p in stored.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('field, value', [('run_id', None), ('commit', 'short'), ('sha256', []),
                                          ('sha256', {'train-a/steps.jsonl': 'bad'}), ('extra', True)])
def test_record_schema_refuses_bad_fields_before_upload(tmp_path, run, field, value):
    entry = dict(run_id='20261126T000000Z-runpod-part1-test-0123abcd',
                 uri='s3://rewardhack-run-archive/20261126T000000Z-runpod-part1-test-0123abcd/',
                 storage_class='STANDARD_IA', commit=COMMIT, uploaded_at='2026-11-26T00:00:00+00:00',
                 sha256={'train-a/steps.jsonl': 'a' * 64})
    entry[field] = value
    (run/'archive.json').write_text(json.dumps([entry]))
    code, output = archive(tmp_path, 'upload', run)
    assert code == 2 and 'validation error' in output
    assert not (tmp_path/'s3').exists()


GOOD_ROW = ['a' * 64, 'train-a/steps.jsonl', 'https://example.invalid/steps.jsonl']


@pytest.mark.parametrize('rows', [[], [GOOD_ROW, GOOD_ROW], [GOOD_ROW, ['bad', 'train-a/x', GOOD_ROW[2]]],
    [GOOD_ROW, ['b' * 64, 'train-a/x', None]], [GOOD_ROW, ['b' * 64, 'train-a/x', 'file:///etc/passwd']],
    *[[GOOD_ROW, ['b' * 64, name, GOOD_ROW[2]]]
      for name in ('/absolute', 'train-a/../escape', 'train-b/x', 'train-a//x', 'train-a/./x')]])
def test_bad_resume_document_is_refused_before_any_download(tmp_path, rows):
    destination = tmp_path/'pod'
    code, output = fetch(tmp_path, destination, json.dumps(rows))
    assert code == 2, output
    assert not destination.exists()  # even the valid first row must not be downloaded


def test_download_failure_removes_the_whole_temporary_run(tmp_path, run):
    assert archive(tmp_path, 'upload', run)[0] == 0
    code, document = archive(tmp_path, 'resume', run/'archive.json', 'train-a/trainer/checkpoint-60')
    assert code == 0, document
    entry = records(run)[0]
    (tmp_path/'s3'/'rewardhack-run-archive'/entry['run_id']/'train-a'/'trainer'/'checkpoint-60'/'optimizer.pt').unlink()
    code, output = fetch(tmp_path, tmp_path/'pod', document)
    assert code != 0 and 'curl' in output and list((tmp_path/'pod').iterdir()) == []


API = dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none',
           SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0', SMOKE_VLLM_ROLLOUTS='1', SMOKE_TOKEN_LIMIT='2048', SMOKE_FINAL_EVAL_SECONDS='600')
# The Qwen3-32B measuring pass (notes section 80): 6 steps, 2,520 s; 71 minutes of setup + (2,520 + 810) s = 126.5 min.
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
