"""The run archive on S3 (ops/archive.env; notes section 78), run on the laptop with the owner's AWS SSO session. Pods hold no
AWS credentials; they get presigned URLs for what they must read.

  python3 ops/run_archive.py upload <results folder>            # a new archive of the folder, every file checked
  python3 ops/run_archive.py resume <results folder> <run>/trainer/checkpoint-<step>
      # 'sha256 path url' per file of that one checkpoint in the latest archive (GET URLs an hour long), for a pod's resume

Each upload gets its own id when it starts (UTC time and a random suffix) and goes to
s3://<bucket>/<prefix>/<folder name>/<id>/, so no upload can collide with or overwrite another (owner, 26 Nov). Each file is
sent with one PutObject carrying its SHA-256, so S3 refuses bytes that differ from the hash (a file changing while it is read
cannot land) and stores that checksum; every file's stored checksum is then read back. Only when all match is the archive
added to ARCHIVES.json in the folder, with every file's hash. Nothing here deletes anything: removing local files after a
verified upload is the owner's own step (owner, 26 Nov).
"""
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ENV = Path(__file__).with_name('archive.env')
SETTINGS = {name.strip(): value.strip() for name, value in
            (line.split('=', 1) for line in ENV.read_text().splitlines() if line.strip() and not line.startswith('#'))}
BUCKET, REGION, STORAGE, PREFIX = (SETTINGS[k] for k in ('ARCHIVE_BUCKET', 'ARCHIVE_REGION', 'ARCHIVE_STORAGE_CLASS', 'ARCHIVE_PREFIX'))
URL_SECONDS = int(SETTINGS['ARCHIVE_URL_SECONDS'])
WORKERS = 8  # parallel hashing and S3 requests; S3 serves many streams, and a laptop's disk and uplink are not slowed by eight
# One PutObject takes at most 5 GB; the largest file of a run so far is a 1.5 GB optimizer state.
PUT_LIMIT_BYTES = 5 * 10**9
# cap: 1.5x estimated 17 min (one 5 GB object at 5 MB/s, the laptop uplink estimate)
PUT_TIMEOUT = 1500
# cap: 1.5x estimated 40 s (one metadata request, with the CLI's own retries)
HEAD_TIMEOUT = 60
RECORD = 'ARCHIVES.json'  # in the results folder: every verified archive of it, newest last
# Top-level files of a results folder that are not part of the run: the record, and the laptop's SSH trust for the pod.
NOT_THE_RUN = {RECORD, RECORD + '.writing', 'known_hosts'}
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
CHECKPOINT = re.compile(r'[^/]+/trainer/checkpoint-\d+')  # where the trainer writes a resumable checkpoint


class ArchiveError(RuntimeError):
    """A request to S3 that failed, or an object stored without the checksum this tool sends."""


def aws(*args, timeout):
    return subprocess.run(['aws', '--region', REGION, *args], capture_output=True, text=True, timeout=timeout, check=False)


def sha256(path):
    with open(path, 'rb') as handle:
        return base64.b64encode(hashlib.file_digest(handle, 'sha256').digest()).decode()


def put(key, path, digest):
    done = aws('s3api', 'put-object', '--bucket', BUCKET, '--key', key, '--body', str(path), '--storage-class', STORAGE,
               '--checksum-sha256', digest, '--if-none-match', '*', timeout=PUT_TIMEOUT)  # S3 refuses to replace an object
    if done.returncode != 0:
        raise ArchiveError(f'put-object {key}: {done.stderr.strip()}')


def stored_sha256(key):
    done = aws('s3api', 'head-object', '--bucket', BUCKET, '--key', key, '--checksum-mode', 'ENABLED',
               '--query', 'ChecksumSHA256', '--output', 'text', timeout=HEAD_TIMEOUT)
    if done.returncode != 0:
        raise ArchiveError(f'head-object {key}: {done.stderr.strip()}')
    value = done.stdout.strip()
    if value in ('', 'None'):
        raise ArchiveError(f'{key} is stored without a SHA-256 checksum')
    return value


def results_folder(path):
    folder = Path(path).resolve(strict=True)
    if not folder.is_dir() or not RUN_NAME.fullmatch(folder.name):
        raise SystemExit(f"{path}: a results folder whose name is letters, digits, '.', '_' or '-'")
    return folder


def run_files(folder):
    """The run's files as names under folder, sorted; a symbolic link anywhere is refused (it would be archived unchecked)."""
    def unreadable(error):
        raise SystemExit(f'cannot read {error.filename}: {error.strerror}')  # a folder skipped would be missing from the archive
    paths = []
    for top, folders, files in os.walk(folder, onerror=unreadable):
        paths += [Path(top)/name for name in folders + files]
    odd = [p for p in paths if p.is_symlink() or not (p.is_dir() or p.is_file())]
    if odd:
        raise SystemExit(f'{folder} holds symbolic links or special files, which the archive cannot check: {odd[:3]}')
    names = sorted(p.relative_to(folder).as_posix() for p in paths if p.is_file())
    return [name for name in names if name not in NOT_THE_RUN]


def in_parallel(function, items):
    """function over items, in input order; every item's failure is reported, and any one stops the command."""
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(function, item) for item in items]
    results, failures = [], []
    for item, future in zip(items, futures, strict=True):
        try:
            results.append(future.result())
        except (ArchiveError, OSError, subprocess.TimeoutExpired) as error:
            failures.append(f'{item}: {error}')
    if failures:
        raise SystemExit('\n'.join(['failed (the archive is not recorded):', *failures[:20]]))
    return results


def records(folder):
    path = folder/RECORD
    if not path.exists():
        return []
    try:
        archives = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f'{path} is not valid JSON ({error}): repair it before archiving or resuming') from error
    if not isinstance(archives, list) or not all(isinstance(a, dict) and {'id', 'uri', 'at', 'sha256'} <= a.keys() for a in archives):
        raise SystemExit(f'{path} is not a list of archives this tool writes: repair it before archiving or resuming')
    return archives


def record(folder, archives):
    """ARCHIVES.json replaced whole: a crash leaves the old file or the new one, never a truncated one."""
    temporary = folder/(RECORD + '.writing')
    temporary.write_text(json.dumps(archives, indent=1) + '\n')
    os.replace(temporary, folder/RECORD)


def upload(path):
    folder = results_folder(path)
    earlier = records(folder)  # read first: an unreadable record stops this before anything is sent
    names = run_files(folder)
    if not names:
        raise SystemExit(f'{folder} holds no files of a run: nothing to archive')
    big = [name for name in names if (folder/name).stat().st_size > PUT_LIMIT_BYTES]
    if big:
        raise SystemExit(f'files over the single-request limit of {PUT_LIMIT_BYTES} bytes: {big[:5]}')
    archive_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)
    root = f'{PREFIX}/{folder.name}/{archive_id}'
    print(f'archiving {len(names)} files to s3://{BUCKET}/{root}/')
    local = in_parallel(lambda name: sha256(folder/name), names)
    in_parallel(lambda item: put(f'{root}/{item[0]}', folder/item[0], item[1]), list(zip(names, local, strict=True)))
    stored = in_parallel(lambda name: stored_sha256(f'{root}/{name}'), names)
    wrong = [name for name, here, there in zip(names, local, stored, strict=True) if here != there]
    if wrong:
        raise SystemExit(f'S3 does not hold these as they were sent; the archive is not recorded: {wrong[:10]}')
    # A file added or removed while the upload ran (a copy from a pod still going) would be missing from the archive.
    if run_files(folder) != names:
        raise SystemExit(f'{folder} changed while it was archived (files added or removed); the archive is not recorded: run it again')
    entry = {'id': archive_id, 'uri': f's3://{BUCKET}/{root}/', 'at': datetime.now(timezone.utc).isoformat(),
             'sha256': dict(zip(names, local, strict=True))}
    record(folder, [*earlier, entry])
    print(f'verified: all {len(names)} files are in {entry["uri"]} with their SHA-256 (recorded in {folder/RECORD})')


def resume(path, checkpoint):
    """For a pod resuming a run: 'sha256 path url' for each file of one trainer checkpoint in the folder's latest archive, and
    nothing else. Standard-IA charges per GB read, so a resume reads one checkpoint (about 2.3 GB for gpt-oss-20b), never
    the run's snapshots or other checkpoints (owner, 26 Nov)."""
    if not CHECKPOINT.fullmatch(checkpoint):
        raise SystemExit(f'{checkpoint}: a trainer checkpoint of a run, <run>/trainer/checkpoint-<step>')
    folder = results_folder(path)
    archives = records(folder)
    if not archives:
        raise SystemExit(f'{folder} has no verified archive ({RECORD}): upload it first')
    latest = archives[-1]
    # Only a URI this tool writes for this folder: an edited record must not send a pod elsewhere.
    if not re.fullmatch(f's3://{re.escape(BUCKET)}/{re.escape(PREFIX)}/{re.escape(folder.name)}/[0-9TZ]+-[0-9a-f]{{8}}/', latest['uri']):
        raise SystemExit(f'{folder/RECORD} names {latest["uri"]}, not an archive of this folder')
    names = sorted(name for name in latest['sha256'] if name.startswith(checkpoint + '/'))
    if not names:
        raise SystemExit(f'no {checkpoint} in archive {latest["id"]}')
    odd = [name for name in names if any(part in ('', '.', '..') for part in name.split('/'))]
    if odd:
        raise SystemExit(f'{folder/RECORD} lists names this tool never writes: {odd[:3]}')
    # A presigned URL dies with the SSO session that signed it, whatever its own expiry (review round 41).
    done = aws('configure', 'export-credentials', '--format', 'process', timeout=HEAD_TIMEOUT)
    if done.returncode != 0:
        raise SystemExit(f'cannot read the AWS session: {done.stderr.strip()}')
    expiration = json.loads(done.stdout)['Expiration']
    left = (datetime.fromisoformat(expiration.replace('Z', '+00:00')) - datetime.now(timezone.utc)).total_seconds()
    if left < URL_SECONDS:
        raise SystemExit(f'the AWS session ends in {left / 60:.0f} minutes, before the URLs would ({URL_SECONDS / 60:.0f}): '
                         'run `aws sso login` again first')
    for name in names:
        done = aws('s3', 'presign', latest['uri'] + name, '--expires-in', str(URL_SECONDS), timeout=HEAD_TIMEOUT)
        if done.returncode != 0:
            raise SystemExit(f'presign {name}: {done.stderr.strip()}')
        print(latest['sha256'][name], name, done.stdout.strip())


# Each command with the number of arguments it takes.
COMMANDS = {'upload': (upload, 1), 'resume': (resume, 2)}

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in COMMANDS or COMMANDS[sys.argv[1]][1] != len(sys.argv) - 2:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]][0](*sys.argv[2:])
