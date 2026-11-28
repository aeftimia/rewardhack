"""Upload and verify runs with rclone; resume emits one JSON array of [sha256, path, URL] rows.
On a pod, fetch reads that array from stdin and publishes the run only after every hash matches.
Existing S3 records and paths are unchanged. See docs/notes-2026-11-21.md section 80.
"""
import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, TypedDict

from pydantic import Field, TypeAdapter

from rewardhack._archive_settings import ENV, SETTINGS
from rewardhack.hf_grpo import CHECKPOINT_FILES, RESUME_RUN_FILES
from rewardhack.json_objects import STRICT, unique_object

BUCKET, REGION, STORAGE, RCLONE_VERSION = (SETTINGS[k] for k in ('ARCHIVE_BUCKET', 'ARCHIVE_REGION', 'ARCHIVE_STORAGE_CLASS', 'RCLONE_VERSION'))
URL_SECONDS = int(SETTINGS['ARCHIVE_URL_SECONDS'])
# cap: 1.5x estimated 4 h (a 60-step run's 12 GB at 1.3 MB/s, the slowest uplink worth allowing for)
TRANSFER_TIMEOUT = 21600
# cap: 1.5x estimated 40 s (one metadata request, a presign or a credentials read, with the tool's own retries)
REQUEST_TIMEOUT = 60
# cap: 1.5x estimated 13 min (a 1.5 GB optimizer state at 3 MB/s)
FETCH_TIMEOUT = 1200
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
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


class Archive(TypedDict):
    __pydantic_config__ = STRICT
    run_id: Annotated[str, Field(pattern=f'^{RUN_ID.pattern}$')]
    uri: str
    storage_class: str
    commit: Annotated[str, Field(pattern=f'^{COMMIT.pattern}$')]
    uploaded_at: str
    sha256: Annotated[dict[str, Digest], Field(min_length=1)]


ARCHIVES = TypeAdapter(list[Archive])
DOWNLOADS = TypeAdapter(Annotated[list[tuple[Digest, str, Annotated[str, Field(pattern=r'^https://\S+$')]]], Field(min_length=1)], config=STRICT)


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


def read_record(text):
    archives = ARCHIVES.validate_python(json.loads(text, object_pairs_hook=unique_object))
    if any(a['uri'] != f's3://{BUCKET}/{a["run_id"]}/' for a in archives):
        raise ValueError('archive URI does not match its run id and bucket')
    return archives


def launch_commit(folder, given):
    """The commit the run used: the driver's record of it, or, for runs that predate that record, the one the owner names."""
    recorded = (folder/COMMIT_FILE).read_text().strip() if (folder/COMMIT_FILE).exists() else None
    if recorded is not None and given is not None and recorded != given:
        raise SystemExit(f'--commit {given} differs from {folder/COMMIT_FILE}: {recorded}')
    commit = recorded if recorded is not None else given
    if not commit or not COMMIT.fullmatch(commit):
        raise SystemExit(f'{folder}: need a full commit in {COMMIT_FILE} or --commit (the driver log or notes say)')
    return commit


def upload(path, given_commit):
    folder = Path(path).resolve(strict=True)
    if not folder.is_dir() or not FOLDER_NAME.fullmatch(folder.name):
        raise SystemExit(f"{path}: a results folder whose name is letters, digits, '.', '_' or '-'")
    # rclone skips symbolic links with only a notice, so a linked file would be left out of the archive (round 46).
    links = [p for p in folder.rglob('*') if p.is_symlink()]
    if links:
        raise SystemExit(f'{folder} holds symbolic links, which rclone would leave out: {links[:3]}')
    earlier = read_record((folder/RECORD).read_text()) if (folder/RECORD).exists() else []
    commit = launch_commit(folder, given_commit)
    version = run('rclone', 'version', timeout=REQUEST_TIMEOUT).split()[1]
    if version != RCLONE_VERSION:
        raise SystemExit(f'rclone {version} here; {ENV.name} pins {RCLONE_VERSION}')
    before = sha256_manifest(folder)
    if not before or any(re.search(r'\s', name) for name in before):
        raise SystemExit(f'{folder}: nothing to archive, or file names with whitespace')
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
        [entry] = read_record(f'[{text}]')
        if entry['run_id'] != source:
            raise SystemExit(f'{source}/{RECORD} names another archive, {entry["run_id"]}')
        return entry
    path = Path(source)
    if path.name != RECORD or not path.is_file():
        raise SystemExit(f"{source}: a run id (<UTC time>-<folder>-<8 hex digits>) or a results folder's {RECORD}")
    archives = read_record(path.read_text())
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
    print(json.dumps([(files[name], name, url) for name, url in zip(names, urls, strict=True)]))


def fetch(destination):
    rows = DOWNLOADS.validate_json(sys.stdin.read())  # one complete document, validated before any download
    names = [name for _, name, _ in rows]
    run_name = names[0].split('/')[0]
    if (not FOLDER_NAME.fullmatch(run_name) or len(set(names)) != len(names)
            or any(not name.startswith(run_name + '/') or PurePosixPath(name).as_posix() != name
                   or '..' in PurePosixPath(name).parts for name in names)):
        raise ValueError('expected distinct relative file paths within one run')
    target = Path(destination)/run_name
    if target.exists() or target.is_symlink():
        raise ValueError(f'{target} already exists: fetch into a destination without it')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='.fetching.') as scratch:
        for digest, name, url in rows:
            path = Path(scratch)/name
            path.parent.mkdir(parents=True, exist_ok=True)
            run('curl', '-fsS', '--max-time', str(FETCH_TIMEOUT), '-o', str(path), url, timeout=FETCH_TIMEOUT)
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
                    raise ValueError(f'{name} differs from its archived SHA-256: nothing kept')
        (Path(scratch)/run_name).rename(target)
    print(f'fetched all {len(rows)} files of {run_name}, each with its archived SHA-256')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    up = commands.add_parser('upload')
    up.add_argument('folder')
    up.add_argument('--commit', help='the full commit the run used, for a results folder without launch-commit.txt')
    back = commands.add_parser('resume')
    back.add_argument('source', help="a run id, or a results folder's archive.json")
    back.add_argument('checkpoint', help='<run>/trainer/checkpoint-<step>')
    commands.add_parser('fetch', help='read resume JSON from stdin on the pod').add_argument('destination')
    args = parser.parse_args()
    handlers = {'upload': lambda: upload(args.folder, args.commit), 'resume': lambda: resume(args.source, args.checkpoint),
                'fetch': lambda: fetch(args.destination)}
    try:
        handlers[args.command]()
    except ValueError as error:
        parser.exit(2, f'{error}\n')
