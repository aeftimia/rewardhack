"""The run archive on S3 (ops/archive.env; notes section 78), run on the laptop with the owner's AWS SSO session. Pods hold no
AWS credentials; they get presigned URLs for what they must read.

  python3 ops/run_archive.py upload <results folder>            # a new archive of the folder, every file checked
  python3 ops/run_archive.py presign <results folder> <path>...  # GET URLs into its latest archive, for a pod's resume

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
import re
import secrets
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ENV = Path(__file__).with_name('archive.env')
SETTINGS = dict(line.split('=', 1) for line in ENV.read_text().splitlines() if line and not line.startswith('#'))
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
NOT_THE_RUN = {RECORD, 'known_hosts'}
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')


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
    links = [p for p in folder.rglob('*') if p.is_symlink()]
    if links:
        raise SystemExit(f'{folder} holds symbolic links, which the archive cannot check: {links[:3]}')
    names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file())
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
    return json.loads(path.read_text()) if path.exists() else []


def upload(path):
    folder = results_folder(path)
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
    entry = {'id': archive_id, 'uri': f's3://{BUCKET}/{root}/', 'at': datetime.now(timezone.utc).isoformat(),
             'sha256': dict(zip(names, local, strict=True))}
    (folder/RECORD).write_text(json.dumps([*records(folder), entry], indent=1) + '\n')
    print(f'verified: all {len(names)} files are in {entry["uri"]} with their SHA-256 (recorded in {folder/RECORD})')


def presign(path, *names):
    if not names:
        raise SystemExit('presign <results folder> <path in the run>...')
    folder = results_folder(path)
    archives = records(folder)
    if not archives:
        raise SystemExit(f'{folder} has no verified archive ({RECORD}): upload it first')
    latest = archives[-1]
    unknown = [name for name in names if name not in latest['sha256']]
    if unknown:
        raise SystemExit(f'not in archive {latest["id"]}: {unknown}')
    for name in names:
        done = aws('s3', 'presign', latest['uri'] + name, '--expires-in', str(URL_SECONDS), timeout=HEAD_TIMEOUT)
        if done.returncode != 0:
            raise SystemExit(f'presign {name}: {done.stderr.strip()}')
        print(done.stdout.strip())


# Each command with the number of arguments it takes (None: one or more).
COMMANDS = {'upload': (upload, 1), 'presign': (presign, None)}

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in COMMANDS or COMMANDS[sys.argv[1]][1] not in (None, len(sys.argv) - 2):
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]][0](*sys.argv[2:])
