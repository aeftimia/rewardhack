"""The run archive on S3: a thin wrapper over rclone (owner, 26 Nov; notes section 80), run on the laptop with the project's
venv and the owner's AWS SSO session (rclone's env_auth reads it). Pods hold no AWS credentials: they get presigned URLs.

  .tools/venv/bin/python ops/run_archive.py upload <results folder> [--commit <sha>]
  .tools/venv/bin/python ops/run_archive.py resume <run id | archive.json> <run>/trainer/checkpoint-<step>

upload: a new run id, <UTC time>-<folder name>-<8 random hex digits>, never derived from anything that can repeat;
`rclone copy` to s3:<bucket>/<run id>/ in the storage class, `--immutable`; `rclone check` with checksums, failing on any
difference; then the record, appended to <results folder>/archive.json (committed with the run's results) and copied beside
the archive: run id, URI, storage class, the git commit the run used, upload time, and every file's SHA-256
(`rclone hashsum sha256`). Everything else about the run is read from its own committed files.

resume: for one finished checkpoint, the lines `sha256 path url` for its files and for the run files a resume reads
(hf_grpo.RESUME_RUN_FILES), then `END <count>`, all or nothing, for ops/fetch_checkpoint.sh on the pod. Standard-IA charges
per GB read, so a resume reads that checkpoint alone, never snapshots or other checkpoints. Nothing here deletes anything.
"""
import argparse
import json
import os
import re
import secrets
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rewardhack.hf_grpo import CHECKPOINT_FILES, RESUME_RUN_FILES

ENV = Path(__file__).with_name('archive.env')
SETTINGS = {name.strip(): value.strip() for name, value in
            (line.split('=', 1) for line in ENV.read_text().splitlines() if line.strip() and not line.startswith('#'))}
BUCKET, REGION, STORAGE, RCLONE_VERSION = (SETTINGS[k] for k in ('ARCHIVE_BUCKET', 'ARCHIVE_REGION', 'ARCHIVE_STORAGE_CLASS', 'RCLONE_VERSION'))
URL_SECONDS = int(SETTINGS['ARCHIVE_URL_SECONDS'])
# cap: 1.5x estimated 4 h (a 60-step run's 12 GB at 1.3 MB/s, the slowest uplink worth allowing for)
TRANSFER_TIMEOUT = 21600
# cap: 1.5x estimated 40 s (one metadata request, a presign or a credentials read, with the tool's own retries)
REQUEST_TIMEOUT = 60
RECORD = 'archive.json'
COMMIT_FILE = 'launch-commit.txt'  # written by ops/drive_runpod.sh: the commit the pod ran
# rclone filters anchored at the folder's top: the record, and the laptop's SSH trust for the pod, are not the run.
EXCLUDES = ['--exclude', f'/{RECORD}', '--exclude', '/known_hosts']
# rclone's S3 remote, from the environment: credentials from the AWS SSO profile (env_auth), no config file.
REMOTE = {'RCLONE_CONFIG_S3_TYPE': 's3', 'RCLONE_CONFIG_S3_PROVIDER': 'AWS', 'RCLONE_CONFIG_S3_ENV_AUTH': 'true',
          'RCLONE_CONFIG_S3_REGION': REGION}
FOLDER_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
RUN_ID = re.compile(r'\d{8}T\d{6}Z-[A-Za-z0-9][A-Za-z0-9._-]*-[0-9a-f]{8}')
COMMIT = re.compile(r'[0-9a-f]{40}')
CHECKPOINT = re.compile(r'([^/]+)/trainer/checkpoint-\d+')
ENTRY_KEYS = {'run_id', 'uri', 'storage_class', 'commit', 'uploaded_at', 'sha256'}


def run(*command, timeout):
    """A command's output; a failure or a timeout stops with its message."""
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, env={**os.environ, **REMOTE})
    if done.returncode != 0:
        raise SystemExit(f'{" ".join(command[:3])} failed:\n{done.stderr.strip()}')
    return done.stdout


def sha256_manifest(folder):
    """Every file's SHA-256 (hex), from rclone's hashsum of the local folder."""
    lines = run('rclone', 'hashsum', 'sha256', str(folder), *EXCLUDES, timeout=TRANSFER_TIMEOUT).splitlines()
    return dict(sorted((name, digest) for digest, name in (line.split('  ', 1) for line in lines)))


def read_record(path):
    try:
        archives = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f'{path} is not valid JSON ({error}): repair it first') from error
    if not isinstance(archives, list) or not all(isinstance(a, dict) and a.keys() == ENTRY_KEYS and RUN_ID.fullmatch(a['run_id'])
                                                 and a['uri'] == f's3://{BUCKET}/{a["run_id"]}/' for a in archives):
        raise SystemExit(f'{path} is not a list of archives this tool writes: repair it first')
    return archives


def launch_commit(folder, given):
    """The commit the run used: the driver's record of it, or, for runs that predate that record, the one the owner names."""
    recorded = (folder/COMMIT_FILE).read_text().strip() if (folder/COMMIT_FILE).exists() else None
    if recorded is not None and not COMMIT.fullmatch(recorded):
        raise SystemExit(f'{folder/COMMIT_FILE} does not hold a full commit: {recorded!r}')
    if recorded and given and recorded != given:
        raise SystemExit(f'--commit {given} differs from {folder/COMMIT_FILE}: {recorded}')
    commit = recorded or given
    if not commit or not COMMIT.fullmatch(commit):
        raise SystemExit(f'{folder} has no {COMMIT_FILE}: name the full commit the run used with --commit (its driver log or notes say)')
    return commit


def upload(path, given_commit):
    folder = Path(path).resolve(strict=True)
    if not folder.is_dir() or not FOLDER_NAME.fullmatch(folder.name):
        raise SystemExit(f"{path}: a results folder whose name is letters, digits, '.', '_' or '-'")
    # rclone skips symbolic links with only a notice, so a linked file would be left out of the archive (round 46).
    links = [p for p in folder.rglob('*') if p.is_symlink()]
    if links:
        raise SystemExit(f'{folder} holds symbolic links, which rclone would leave out: {links[:3]}')
    earlier = read_record(folder/RECORD) if (folder/RECORD).exists() else []
    commit = launch_commit(folder, given_commit)
    version = run('rclone', 'version', timeout=REQUEST_TIMEOUT).split()[1]
    if version != RCLONE_VERSION:
        raise SystemExit(f'rclone {version} here; ops/archive.env pins {RCLONE_VERSION}')
    before = sha256_manifest(folder)
    if not before or any(re.search(r'\s', name) for name in before):
        raise SystemExit(f'{folder}: nothing to archive, or file names with whitespace (the resume lines are space-separated)')
    run_id = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{folder.name}-{secrets.token_hex(4)}'
    remote = f's3:{BUCKET}/{run_id}'
    print(f'archiving {len(before)} files as {run_id}')
    run('rclone', 'copy', str(folder), remote, '--s3-storage-class', STORAGE, '--immutable', *EXCLUDES, timeout=TRANSFER_TIMEOUT)
    run('rclone', 'check', str(folder), remote, '--one-way', *EXCLUDES, timeout=TRANSFER_TIMEOUT)
    # The hashes recorded are of the files rclone checked: unchanged since the copy began (review rounds 38-43).
    if sha256_manifest(folder) != before:
        raise SystemExit(f'{folder} changed while it was archived; not recorded: run it again')
    entry = {'run_id': run_id, 'uri': f's3://{BUCKET}/{run_id}/', 'storage_class': STORAGE, 'commit': commit,
             'uploaded_at': datetime.now(timezone.utc).isoformat(), 'sha256': before}
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch)/RECORD).write_text(json.dumps(entry, indent=1) + '\n')
        run('rclone', 'copyto', str(Path(scratch)/RECORD), f'{remote}/{RECORD}', '--s3-storage-class', STORAGE, '--immutable',
            timeout=REQUEST_TIMEOUT)
    temporary = folder/(RECORD + '.writing')
    temporary.write_text(json.dumps([*earlier, entry], indent=1) + '\n')
    os.replace(temporary, folder/RECORD)
    print(f'verified: all {len(before)} files are in {entry["uri"]} (rclone check); recorded in {folder/RECORD} (commit it)')


def archive_entry(source):
    """A run id (its record read from the bucket) or a local archive.json (its newest entry: give the run id for another)."""
    if RUN_ID.fullmatch(source):
        text = run('rclone', 'cat', f's3:{BUCKET}/{source}/{RECORD}', timeout=REQUEST_TIMEOUT)
        if not text.strip():  # rclone cat of a missing object prints nothing and succeeds (round 46)
            raise SystemExit(f'no archive s3:{BUCKET}/{source}/{RECORD}')
        with tempfile.TemporaryDirectory() as scratch:
            (Path(scratch)/'list.json').write_text(f'[{text}]')
            [entry] = read_record(Path(scratch)/'list.json')
        if entry['run_id'] != source:
            raise SystemExit(f'{source}/{RECORD} names another archive, {entry["run_id"]}')
        return entry
    path = Path(source)
    if path.name != RECORD or not path.is_file():
        raise SystemExit(f"{source}: a run id (<UTC time>-<folder>-<8 hex digits>) or a results folder's {RECORD}")
    archives = read_record(path)
    if not archives:
        raise SystemExit(f'{path} lists no archive')
    return archives[-1]


def resume(source, checkpoint):
    match = CHECKPOINT.fullmatch(checkpoint)
    if not match:
        raise SystemExit(f'{checkpoint}: a trainer checkpoint of a run, <run>/trainer/checkpoint-<step>')
    entry = archive_entry(source)
    files = entry['sha256']
    # A finished save holds every file the trainer resumes from, trainer_state.json last (HF writes it after the rest).
    needed = [f'{checkpoint}/{name}' for name in (*CHECKPOINT_FILES, 'trainer_state.json')] + \
             [f'{match.group(1)}/{name}' for name in RESUME_RUN_FILES]
    missing = [name for name in needed if name not in files]
    if missing:
        raise SystemExit(f'archive {entry["run_id"]} cannot resume {checkpoint}: it lacks {missing}')
    names = sorted({*needed, *(name for name in files if name.startswith(checkpoint + '/'))})
    # A presigned URL dies with the SSO role credentials that signed it, whatever its own expiry (review rounds 41-42).
    expiration = json.loads(run('aws', 'configure', 'export-credentials', '--format', 'process', timeout=REQUEST_TIMEOUT))['Expiration']
    left = (datetime.fromisoformat(expiration.replace('Z', '+00:00')) - datetime.now(timezone.utc)).total_seconds()
    if left < URL_SECONDS:
        raise SystemExit(f'the AWS role credentials end in {left / 60:.0f} minutes, before the URLs would '
                         f'({URL_SECONDS / 60:.0f}): run `aws sso logout && aws sso login` first')
    urls = [run('aws', '--region', REGION, 's3', 'presign', entry['uri'] + name, '--expires-in', str(URL_SECONDS),
                timeout=REQUEST_TIMEOUT).strip() for name in names]
    for name, url in zip(names, urls, strict=True):  # all lines or none, then the count the pod requires (round 42)
        print(files[name], name, url)
    print('END', len(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    up = commands.add_parser('upload')
    up.add_argument('folder')
    up.add_argument('--commit', help='the full commit the run used, for a results folder without launch-commit.txt')
    back = commands.add_parser('resume')
    back.add_argument('source', help="a run id, or a results folder's archive.json")
    back.add_argument('checkpoint', help='<run>/trainer/checkpoint-<step>')
    args = parser.parse_args()
    handlers = {'upload': lambda: upload(args.folder, args.commit), 'resume': lambda: resume(args.source, args.checkpoint)}
    handlers[args.command]()


if __name__ == '__main__':
    main()
