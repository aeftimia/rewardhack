"""The run archive on S3 (ops/archive.env; notes section 78), run on the laptop with the owner's AWS SSO session. Pods hold no
AWS credentials; they get presigned URLs for what they must read.

  python3 ops/run_archive.py upload <results folder>   # every file S3 lacks with its local SHA-256 is sent, then every
                                                      # file's stored SHA-256 is read back; ARCHIVED only when all match
  python3 ops/run_archive.py prune <results folder>    # deletes <run>/*/adapter-stepN and <run>/*/trainer/checkpoint-N,
                                                      # folder by folder, each only when S3 holds every file's exact bytes
  python3 ops/run_archive.py presign <run name> <path>...  # GET URLs, ARCHIVE_URL_SECONDS long, for a pod's resume

What S3 holds is checked against S3 itself (review round 38): each file is sent with one PutObject carrying its SHA-256, so
S3 refuses bytes that differ from the hash (a file changing while it is read cannot land), and stores that checksum, which
HeadObject returns. Nothing relies on sizes, times, folder paths or a manifest kept beside the files. A snapshot or checkpoint
file S3 already holds with other bytes is never replaced (its archived copy may be the only one once pruned); other files
(logs a relaunch appends to) are. Only the owner runs prune, one run at a time (the laptop has about 50 GB free).
"""
import base64
import hashlib
import json
import re
import shutil
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
# Top-level files of a results folder that are not part of the run: the archive's mark, and the laptop's SSH trust for the pod.
NOT_THE_RUN = {'ARCHIVED', 'known_hosts'}
# Where the trainer writes what prune may delete (hf_grpo: adapter-step{global_step}; the trainer's checkpoint-{step}).
PRUNABLE = re.compile(r'[^/]+/(?:adapter-step\d+|trainer/checkpoint-\d+)')
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')


class ArchiveError(RuntimeError):
    """A request to S3 that failed, or an object this tool did not write."""


def aws(*args, timeout):
    return subprocess.run(['aws', '--region', REGION, *args], capture_output=True, text=True, timeout=timeout, check=False)


def sha256(path):
    with open(path, 'rb') as handle:
        return base64.b64encode(hashlib.file_digest(handle, 'sha256').digest()).decode()


def key(run, name):
    return f'{PREFIX}/{run}/{name}'


def prunable_folder(name):
    """The snapshot or checkpoint folder a file lies in, or None."""
    for depth in (2, 3):
        head = '/'.join(name.split('/')[:depth])
        if PRUNABLE.fullmatch(head):
            return head
    return None


def stored_sha256(run, name):
    """The SHA-256 S3 stores for an object, or None when there is no such object. Any other failure raises."""
    done = aws('s3api', 'head-object', '--bucket', BUCKET, '--key', key(run, name), '--checksum-mode', 'ENABLED',
               '--query', 'ChecksumSHA256', '--output', 'text', timeout=HEAD_TIMEOUT)
    if done.returncode != 0:
        if '(404)' in done.stderr or 'Not Found' in done.stderr:
            return None
        raise ArchiveError(f'head-object {name}: {done.stderr.strip()}')
    value = done.stdout.strip()
    if value in ('', 'None'):
        raise ArchiveError(f'{name} is stored without a SHA-256 checksum: it was not sent by this tool')
    return value


def put(run, folder, name, digest):
    done = aws('s3api', 'put-object', '--bucket', BUCKET, '--key', key(run, name), '--body', str(folder/name),
               '--storage-class', STORAGE, '--checksum-sha256', digest, timeout=PUT_TIMEOUT)
    if done.returncode != 0:
        raise ArchiveError(f'put-object {name}: {done.stderr.strip()}')


def run_folder(path):
    """The resolved folder and its run name; the name is the run's key in the bucket."""
    folder = Path(path).resolve(strict=True)
    if not folder.is_dir() or not RUN_NAME.fullmatch(folder.name):
        raise SystemExit(f"{path}: a results folder whose name is letters, digits, '.', '_' or '-'")
    return folder, folder.name


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
        raise SystemExit('\n'.join(['failed:', *failures[:20]]))
    return results


def upload(path):
    folder, run = run_folder(path)
    (folder/'ARCHIVED').unlink(missing_ok=True)  # only a finished upload marks the run (round 34)
    names = run_files(folder)
    if not names:
        raise SystemExit(f'{folder} holds no files of a run: nothing to archive')
    big = [name for name in names if (folder/name).stat().st_size > PUT_LIMIT_BYTES]
    if big:
        raise SystemExit(f'files over the single-request limit of {PUT_LIMIT_BYTES} bytes: {big[:5]}')
    local = dict(zip(names, in_parallel(lambda name: sha256(folder/name), names), strict=True))
    stored = dict(zip(names, in_parallel(lambda name: stored_sha256(run, name), names), strict=True))
    clashes = [n for n in names if stored[n] not in (None, local[n]) and prunable_folder(n)]
    if clashes:
        raise SystemExit(f's3://{BUCKET}/{PREFIX}/{run} holds these snapshot or checkpoint files with other bytes, so it is '
                         f'another run of the same name: rename the folder. {clashes[:10]}')
    sending = [name for name in names if stored[name] != local[name]]
    print(f'sending {len(sending)} of {len(names)} files to s3://{BUCKET}/{PREFIX}/{run}/')
    in_parallel(lambda name: put(run, folder, name, local[name]), sending)
    # Read back: every file's stored SHA-256 is its local one (another upload of the same name may have raced this one).
    after = dict(zip(names, in_parallel(lambda name: stored_sha256(run, name), names), strict=True))
    wrong = [name for name in names if after[name] != local[name]]
    if wrong:
        raise SystemExit(f'S3 does not hold these files as they are here: {wrong[:10]}')
    (folder/'ARCHIVED').write_text(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'uri': f's3://{BUCKET}/{PREFIX}/{run}/',
                                               'files': len(names)}) + '\n')
    print(f'archived: all {len(names)} files are in S3 with their SHA-256')


def prune(path):
    folder, run = run_folder(path)
    if not (folder/'ARCHIVED').exists():
        raise SystemExit(f'{folder} has not been archived (no ARCHIVED): upload it first')
    # Candidates by the trainer's own layout, one run folder down. The folder is resolved, so a candidate whose resolved path
    # differs is reached through a symbolic link, and might lie outside the run.
    candidates = sorted({p for pattern in ('*/adapter-step*', '*/trainer/checkpoint-*') for p in folder.glob(pattern)})
    removed = 0
    for candidate in candidates:
        name = candidate.relative_to(folder).as_posix()
        if not PRUNABLE.fullmatch(name) or not candidate.is_dir():
            continue
        if candidate.resolve() != candidate:
            raise SystemExit(f'{candidate} is reached through a symbolic link: nothing more pruned')
        # Hashed and checked against S3 now, then deleted, one folder at a time: a file written meanwhile is not lost.
        files = sorted(p for p in candidate.rglob('*') if not p.is_dir())
        if any(p.is_symlink() for p in files):
            raise SystemExit(f'{candidate} holds a symbolic link: nothing more pruned')
        names = [p.relative_to(folder).as_posix() for p in files]
        local = in_parallel(lambda n: sha256(folder/n), names)
        stored = in_parallel(lambda n: stored_sha256(run, n), names)
        missing = [n for n, here, there in zip(names, local, stored, strict=True) if here != there]
        if missing:
            raise SystemExit(f'S3 does not hold these as they are here, so {name} is kept (upload again first): {missing[:10]}')
        if sorted(p for p in candidate.rglob('*') if not p.is_dir()) != files:
            raise SystemExit(f'{name} changed while it was checked: kept')
        shutil.rmtree(candidate)
        removed += 1
        print(f'pruned {name} ({len(names)} files, each held in S3 with its SHA-256)')
    print(f'{removed} snapshot and checkpoint folders pruned')


def presign(run, *names):
    if not RUN_NAME.fullmatch(run):
        raise SystemExit(f'not a run name: {run}')
    for name in names:
        done = aws('s3', 'presign', f's3://{BUCKET}/{key(run, name)}', '--expires-in', str(URL_SECONDS), timeout=HEAD_TIMEOUT)
        if done.returncode != 0:
            raise SystemExit(f'presign {name}: {done.stderr.strip()}')
        print(done.stdout.strip())


# Each command with the number of arguments it takes (None: one or more).
COMMANDS = {'upload': (upload, 1), 'prune': (prune, 1), 'presign': (presign, None)}

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in COMMANDS or COMMANDS[sys.argv[1]][1] not in (None, len(sys.argv) - 2):
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]][0](*sys.argv[2:])
