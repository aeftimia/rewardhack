"""Sign immutable pod uploads and record S3-verified receipts for the registry."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionClosedError, ReadTimeoutError

from archive_pod_snapshot import sign, verify, DESCRIBE_SECONDS, URL_SECONDS, CLIENT_OPTIONS
from rewardhack import archive
from rewardhack.state_files import atomic_json


def copy_snapshot(client, source, key, row):
    """A lost acknowledgement may leave a complete immutable destination."""
    started = time.monotonic()
    try:
        response = client.copy_object(Bucket=archive.BUCKET, Key=key, CopySource={'Bucket': archive.BUCKET, 'Key': source},
                                      StorageClass=archive.STORAGE, ChecksumAlgorithm='SHA256', IfNoneMatch='*')
        outcome = f"acknowledged after {response['ResponseMetadata']['RetryAttempts']} SDK retries"
    except ClientError as error:
        if error.response['Error']['Code'] != 'PreconditionFailed' or error.response['ResponseMetadata']['HTTPStatusCode'] != 412:
            raise
        outcome = 'destination already exists'
    except (ReadTimeoutError, ConnectionClosedError) as error:
        outcome = type(error).__name__
    print(f"S3 copy {row['path']} ({row['size']} bytes): {outcome} in {time.monotonic()-started:.1f}s; checking destination", flush=True)
    verify(client, archive.BUCKET, key, row)


def final_attempt(result, rows, candidate):
    """Share a destination across retries of the same immutable output inventory."""
    inventory = sorted((row['path'], row['size'], row['sha256']) for row in rows)
    # dry-ok: ops/pod_snapshot_upload.py::describe | This local filename fingerprints the ordered list of inventory entries. Its encoding and algorithm may change independently of the SHA-256 of each object's bytes that S3 verifies.
    digest = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
    path = result.with_suffix(f'.direct-attempt-{digest}.json')
    try:
        atomic_json(path, candidate)
    except FileExistsError:
        saved = json.loads(path.read_text())
        if not archive.RUN_ID.fullmatch(saved) or f'-{result.name}-outputs-' not in saved:
            raise ValueError(f'invalid archive attempt in {path}') from None
        return saved, True
    return candidate, False


def transfer(mode, result, command, client):
    config = json.loads(Path('.tools/experiment.json').read_text())
    if config['run_id'] != result.name:
        raise ValueError('archive destination differs from experiment ID')
    def remote(action, data=None):
        response = subprocess.run([*command, action], input=json.dumps(data), stdout=subprocess.PIPE,
                                  text=True, check=True, timeout=URL_SECONDS + DESCRIBE_SECONDS)
        return json.loads(response.stdout)

    verified_snapshots = {}
    def publish(label, rows, snapshot):
        run_id = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{result.name}-{label}-{secrets.token_hex(4)}'
        resumed = False
        if not snapshot:
            run_id, resumed = final_attempt(result, rows, run_id)
        def send(row):
            path = Path(row['path'])
            if path.is_absolute() or '..' in path.parts or (snapshot and path.parts[0] != config['output_arm']):
                raise ValueError('unexpected pod artifact path')
            name = path.name if snapshot else path.as_posix()
            key = f'{run_id}/{name}'
            if resumed:
                try:
                    verify(client, archive.BUCKET, key, row)
                except ClientError as error:
                    if error.response['ResponseMetadata']['HTTPStatusCode'] != 404:
                        raise
                else:
                    return name, row['sha256']
            previous = verified_snapshots.get((path.as_posix(), row['sha256']))
            if previous is None:
                request = sign(client, archive.BUCKET, key, row)
                remote('upload', request)
                verify(client, archive.BUCKET, key, row)
            else:
                copy_snapshot(client, previous, key, row)
            return name, row['sha256']
        with ThreadPoolExecutor(4) as pool:
            hashes = dict(pool.map(send, rows))
        return dict(run_id=run_id, sha256=hashes)

    directory = result.resolve().parents[2]
    for label, files in remote('snapshots').items():
        if not re.fullmatch(r'adapter-step[1-9][0-9]*', label) or not files or any(Path(n).name != n for n in files):
            raise ValueError('invalid adapter snapshot inventory')
        folder = directory / 'snapshots' / f'{result.name}-{label}'
        receipt = folder / archive.RECORD
        if receipt.exists():
            # dry-ok: src/rewardhack/experiment_results.py::registered_checkpoint | This producer checks its own per-step completion receipt to reuse verified S3 objects. When the file is absent it uploads a new snapshot. The reader can instead fetch an existing full-run receipt by its registered archive ID.
            entry = archive.archive_entry(str(receipt))
            for name, digest in entry['sha256'].items():
                verified_snapshots[(f"{config['output_arm']}/{label}/{name}", digest)] = f"{entry['run_id']}/{name}"
            continue
        names = [f"{config['output_arm']}/{label}/{name}" for name in files]
        rows = remote('describe', names)
        if [row['path'] for row in rows] != names:
            raise ValueError('snapshot inventory changed')
        entry = publish(label, rows, True)
        entry.update(uri=f"s3://{archive.BUCKET}/{entry['run_id']}/", storage_class=archive.STORAGE,
                     commit=(result / archive.COMMIT_FILE).read_text().strip(),
                     uploaded_at=datetime.now(timezone.utc).isoformat())
        client.put_object(Bucket=archive.BUCKET, Key=f"{entry['run_id']}/archive.json",
                          Body=json.dumps(entry).encode(), IfNoneMatch='*', StorageClass=archive.STORAGE)
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt, [entry])
        for name, digest in entry['sha256'].items():
            verified_snapshots[(f"{config['output_arm']}/{label}/{name}", digest)] = f"{entry['run_id']}/{name}"
    if mode == 'final':
        direct = publish('outputs', remote('bulk'), False)
        atomic_json(result.with_suffix('.direct.json'), direct, exclusive=False)


if __name__ == '__main__':
    client = boto3.client('s3', region_name=archive.REGION, config=Config(**CLIENT_OPTIONS))
    transfer(sys.argv[1], Path(sys.argv[2]), sys.argv[3:], client)
