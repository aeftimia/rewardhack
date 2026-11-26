"""The run archive on S3 (ops/archive.env; notes section 78), run on the laptop with the owner's AWS SSO session. Pods hold no
AWS credentials; they get presigned URLs for what they must read.

  python3 ops/run_archive.py upload <results folder> [--commit <sha>]
      # archives the folder under a new run id, s3://<bucket>/<run id>/, every file's SHA-256 checked by S3, and appends
      # the archive to <results folder>/archive.json, the small record committed with the run's results
  python3 ops/run_archive.py resume <run id | archive.json> <run>/trainer/checkpoint-<step>
      # 'sha256 path url' per file of that one checkpoint, then 'END <count>' (GET URLs ARCHIVE_URL_SECONDS long),
      # for ops/fetch_checkpoint.sh on a pod

A run id is <UTC time>-<results folder name>-<8 random hex digits>: made when the upload starts, never derived from anything
that can repeat (owner, 26 Nov). archive.json holds, per archive: the run id, its URI, the storage class, the git commit the
run used, the upload time and every file's SHA-256. Everything else about the run (settings, model, λ) is read from its own
committed files (summary.json, run-settings.json), not repeated here. The same record is stored as <run id>/archive.json in the
bucket, so a run id alone is enough to resume. Nothing here deletes anything (owner, 26 Nov).

Each file goes up in one PutObject carrying its SHA-256 and If-None-Match: *, so S3 refuses bytes that differ from the hash (a
file changing while it is read cannot land) and refuses to replace any object; every stored checksum is read back before the
archive is recorded.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ENV = Path(__file__).with_name('archive.env')
SETTINGS = {name.strip(): value.strip() for name, value in
            (line.split('=', 1) for line in ENV.read_text().splitlines() if line.strip() and not line.startswith('#'))}
BUCKET, REGION, STORAGE = (SETTINGS[k] for k in ('ARCHIVE_BUCKET', 'ARCHIVE_REGION', 'ARCHIVE_STORAGE_CLASS'))
URL_SECONDS = int(SETTINGS['ARCHIVE_URL_SECONDS'])
WORKERS = 8  # parallel hashing and S3 requests; S3 serves many streams, and a laptop's disk and uplink are not slowed by eight
# One PutObject takes at most 5 GB; the largest file of a run so far is a 1.5 GB optimizer state.
PUT_LIMIT_BYTES = 5 * 10**9
# cap: 1.5x estimated 17 min (one 5 GB object at 5 MB/s, the laptop uplink estimate)
PUT_TIMEOUT = 1500
# cap: 1.5x estimated 40 s (one metadata request, a presign or a credentials read, with the CLI's own retries)
HEAD_TIMEOUT = 60
RECORD = 'archive.json'  # in the results folder, committed with the run's results; also <run id>/archive.json in the bucket
COMMIT_FILE = 'launch-commit.txt'  # written by ops/drive_runpod.sh: the commit the pod ran
# Top-level files of a results folder that are not part of the archived run: the record, and the laptop's SSH trust for the pod.
NOT_THE_RUN = {RECORD, RECORD + '.writing', 'known_hosts'}
FOLDER_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
RUN_ID = re.compile(r'\d{8}T\d{6}Z-[A-Za-z0-9][A-Za-z0-9._-]*-[0-9a-f]{8}')
COMMIT = re.compile(r'[0-9a-f]{40}')
CHECKPOINT = re.compile(r'[^/]+/trainer/checkpoint-\d+')  # where the trainer writes a resumable checkpoint
ENTRY_KEYS = {'run_id', 'uri', 'storage_class', 'commit', 'uploaded_at', 'sha256'}


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


def run_files(folder):
    """The run's files as names under folder, sorted. An unreadable folder, a symbolic link or a special file stops it: each
    would leave the archive short of the folder while it reported every file."""
    def unreadable(error):
        raise SystemExit(f'cannot read {error.filename}: {error.strerror}')
    paths = []
    for top, folders, files in os.walk(folder, onerror=unreadable):
        paths += [Path(top)/name for name in folders + files]
    odd = [p for p in paths if p.is_symlink() or not (p.is_dir() or p.is_file())]
    if odd:
        raise SystemExit(f'{folder} holds symbolic links or special files, which the archive cannot check: {odd[:3]}')
    names = sorted(p.relative_to(folder).as_posix() for p in paths if p.is_file())
    spaced = [name for name in names if re.search(r'\s', name)]
    if spaced:
        raise SystemExit(f'file names with spaces cannot be resumed (the resume lines are space-separated): {spaced[:3]}')
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


def read_record(path):
    """The archives a record lists, each checked to be one this tool writes."""
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
    earlier = read_record(folder/RECORD) if (folder/RECORD).exists() else []  # read first: a bad record stops this before sending
    commit = launch_commit(folder, given_commit)
    names = run_files(folder)
    if not names:
        raise SystemExit(f'{folder} holds no files of a run: nothing to archive')
    big = [name for name in names if (folder/name).stat().st_size > PUT_LIMIT_BYTES]
    if big:
        raise SystemExit(f'files over the single-request limit of {PUT_LIMIT_BYTES} bytes: {big[:5]}')
    run_id = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{folder.name}-{secrets.token_hex(4)}'
    print(f'archiving {len(names)} files as {run_id}')
    local = in_parallel(lambda name: sha256(folder/name), names)
    in_parallel(lambda item: put(f'{run_id}/{item[0]}', folder/item[0], item[1]), list(zip(names, local, strict=True)))
    stored = in_parallel(lambda name: stored_sha256(f'{run_id}/{name}'), names)
    wrong = [name for name, here, there in zip(names, local, stored, strict=True) if here != there]
    if wrong:
        raise SystemExit(f'S3 does not hold these as they were sent; the archive is not recorded: {wrong[:10]}')
    # A file added or removed while the upload ran (a copy from a pod still going) would be missing from the archive.
    if run_files(folder) != names:
        raise SystemExit(f'{folder} changed while it was archived (files added or removed); not recorded: run it again')
    entry = {'run_id': run_id, 'uri': f's3://{BUCKET}/{run_id}/', 'storage_class': STORAGE, 'commit': commit,
             'uploaded_at': datetime.now(timezone.utc).isoformat(), 'sha256': dict(zip(names, local, strict=True))}
    # The bucket's copy first: if it fails, nothing local claims an archive. Then the local record, replaced whole.
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch)/RECORD).write_text(json.dumps(entry, indent=1) + '\n')
        put(f'{run_id}/{RECORD}', Path(scratch)/RECORD, sha256(Path(scratch)/RECORD))
    temporary = folder/(RECORD + '.writing')
    temporary.write_text(json.dumps([*earlier, entry], indent=1) + '\n')
    os.replace(temporary, folder/RECORD)
    print(f'verified: all {len(names)} files are in {entry["uri"]} with their SHA-256; recorded in {folder/RECORD} (commit it)')


def archive_entry(source):
    """The archive a resume reads: a run id (its record fetched from the bucket), or a local archive.json (its newest entry)."""
    if RUN_ID.fullmatch(source):
        with tempfile.TemporaryDirectory() as scratch:
            done = aws('s3api', 'get-object', '--bucket', BUCKET, '--key', f'{source}/{RECORD}', str(Path(scratch)/RECORD),
                       timeout=HEAD_TIMEOUT)
            if done.returncode != 0:
                raise SystemExit(f'no archive {source}: {done.stderr.strip()}')
            [entry] = read_record_entry(Path(scratch)/RECORD)
        if entry['run_id'] != source:
            raise SystemExit(f'{source}/{RECORD} names another archive, {entry["run_id"]}')
        return entry
    path = Path(source)
    if path.name != RECORD or not path.is_file():
        raise SystemExit(f'{source}: a run id (<UTC time>-<folder>-<8 hex digits>) or a results folder\'s {RECORD}')
    archives = read_record(path)
    if not archives:
        raise SystemExit(f'{path} lists no archive')
    return archives[-1]


def read_record_entry(path):
    """The bucket's record of one archive, as a one-item list so read_record's checks apply."""
    try:
        entry = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f'{path} is not valid JSON ({error})') from error
    scratch = Path(path).with_suffix('.list')
    scratch.write_text(json.dumps([entry]))
    return read_record(scratch)


def resume(source, checkpoint):
    """For a pod resuming a run: 'sha256 path url' for each file of one trainer checkpoint, and nothing else. Standard-IA
    charges per GB read, so a resume reads one checkpoint (about 2.3 GB for gpt-oss-20b), never the run's snapshots or other
    checkpoints (owner, 26 Nov)."""
    if not CHECKPOINT.fullmatch(checkpoint):
        raise SystemExit(f'{checkpoint}: a trainer checkpoint of a run, <run>/trainer/checkpoint-<step>')
    entry = archive_entry(source)
    names = sorted(name for name in entry['sha256'] if name.startswith(checkpoint + '/'))
    if not names:
        raise SystemExit(f'no {checkpoint} in archive {entry["run_id"]}')
    odd = [name for name in names if any(part in ('', '.', '..') for part in name.split('/'))]
    if odd:
        raise SystemExit(f'the record of {entry["run_id"]} lists names this tool never writes: {odd[:3]}')
    # A presigned URL dies with the SSO session that signed it, whatever its own expiry (review round 41).
    done = aws('configure', 'export-credentials', '--format', 'process', timeout=HEAD_TIMEOUT)
    if done.returncode != 0:
        raise SystemExit(f'cannot read the AWS session: {done.stderr.strip()}')
    expiration = json.loads(done.stdout)['Expiration']
    left = (datetime.fromisoformat(expiration.replace('Z', '+00:00')) - datetime.now(timezone.utc)).total_seconds()
    if left < URL_SECONDS:
        # `aws sso login` alone keeps the cached role credentials until they expire; logging out clears them (round 42).
        raise SystemExit(f'the AWS role credentials end in {left / 60:.0f} minutes, before the URLs would '
                         f'({URL_SECONDS / 60:.0f}): run `aws sso logout && aws sso login` first')
    urls = []
    for name in names:
        done = aws('s3', 'presign', entry['uri'] + name, '--expires-in', str(URL_SECONDS), timeout=HEAD_TIMEOUT)
        if done.returncode != 0:
            raise SystemExit(f'presign {name}: {done.stderr.strip()}')
        urls.append(done.stdout.strip())
    # All lines or none, then a count the pod requires, so a cut-short list is never taken for a whole checkpoint (round 42).
    for name, url in zip(names, urls, strict=True):
        print(entry['sha256'][name], name, url)
    print('END', len(names))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    up = commands.add_parser('upload')
    up.add_argument('folder')
    up.add_argument('--commit', help='the full commit the run used, for a results folder without launch-commit.txt')
    back = commands.add_parser('resume')
    back.add_argument('source', help='a run id, or a results folder\'s archive.json')
    back.add_argument('checkpoint', help='<run>/trainer/checkpoint-<step>')
    args = parser.parse_args()
    handlers = {'upload': lambda: upload(args.folder, args.commit), 'resume': lambda: resume(args.source, args.checkpoint)}
    handlers[args.command]()


if __name__ == '__main__':
    main()
