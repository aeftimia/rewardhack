"""Coordinate one immutable GPT snapshot upload using the installed AWS CLI Python.

Run with awscli's bundled interpreter. No AWS credentials or signed URLs are
written into the experiment output. A receipt is published only after S3 checks.
"""
import argparse
import json
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from awscli.botocore import session
from awscli.botocore.config import Config

from pod_snapshot_upload import s3_checksum, verify_s3_checksum

# The AWS CLI interpreter runs this script directly from the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rewardhack._archive_settings import SETTINGS

URL_SECONDS = 900
S3_READ_TIMEOUT_SECONDS = 60
DESCRIBE_SECONDS = 300  # 1.5x estimated 200 seconds to hash a complete 11 GB run
REMOTE_HELPER = 'pilot-repo/ops/pod_snapshot_upload.py'


def sign(client, bucket, key, row):
    checksum = s3_checksum(row)
    headers = {'Content-Length': str(row['size']), 'If-None-Match': '*',
               'x-amz-checksum-sha256': checksum,
               'x-amz-storage-class': SETTINGS['ARCHIVE_STORAGE_CLASS']}
    url = client.generate_presigned_url('put_object', Params={
        'Bucket': bucket, 'Key': key, 'ContentLength': row['size'],
        'ChecksumSHA256': checksum, 'IfNoneMatch': '*',
        'StorageClass': SETTINGS['ARCHIVE_STORAGE_CLASS']}, ExpiresIn=URL_SECONDS, HttpMethod='PUT')
    signed = set(parse_qs(urlsplit(url).query)['X-Amz-SignedHeaders'][0].split(';'))
    if not {name.lower() for name in headers} <= signed:
        raise ValueError('presign omitted a required immutable-upload constraint')
    return dict(row, url=url, headers=headers)


def verify(client, bucket, key, row):
    response = client.head_object(Bucket=bucket, Key=key, ChecksumMode='ENABLED')
    verify_s3_checksum(row, response['ChecksumSHA256'])
    if response['ContentLength'] != row['size']:
        raise ValueError('S3 object size differs from the snapshot size')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    parser.add_argument('--run', required=True, choices=['train-baseline', 'train-lambda0', 'train-lambda05'])
    # dry-ok: entrypoints/pilot.py::main | This CLI needs a checkpoint index or a completed-run selector to determine which files the pod must upload. The pilot CLI selects a frozen spec or offline example and has no snapshot or S3 input. Adding a default spec to pilot would leave snapshot selection compulsory.
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--step', type=int)
    selection.add_argument('--completed', action='store_true')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    args = parser.parse_args()
    if (args.step is not None and args.step < 1) or len(args.commit) != 40 or any(c not in '0123456789abcdef' for c in args.commit):
        parser.error('positive step and full launch commit required')
    # Only a literal IP is accepted: no shell syntax or accidental SSH options.
    import ipaddress
    ipaddress.ip_address(args.host)
    if not 1 <= args.port <= 65535:
        parser.error('invalid SSH port')
    ssh = ['ssh', '-p', str(args.port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', '-o', 'StrictHostKeyChecking=yes',
           '-o', f'UserKnownHostsFile={args.known_hosts.resolve()}', f'root@{args.host}',
           'python3', REMOTE_HELPER]
    names = [f'{args.run}/adapter-step{args.step}/{name}'
             for name in ('adapter_model.safetensors', 'adapter_config.json', 'README.md')]
    names += [f'{args.run}/{name}' for name in ('run-settings.json', 'tasks.json')]
    command = ['describe-completed', args.run] if args.completed else ['describe', *names]
    description = subprocess.run([*ssh, *command], check=True, capture_output=True,
                                 text=True, timeout=DESCRIBE_SECONDS)
    rows = json.loads(description.stdout)
    if args.completed:
        names = [row['path'] for row in rows]
        if not names or len(set(names)) != len(names) or any(not name.startswith(args.run + '/') for name in names):
            raise ValueError('pod returned unexpected completed-run paths')
    elif [row['path'] for row in rows] != names:
        raise ValueError('pod returned unexpected snapshot paths')
    label = 'completed' if args.completed else f'step-{args.step}'
    run_id = f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-gpt-{args.run}-{label}-{secrets.token_hex(4)}'
    bucket = SETTINGS['ARCHIVE_BUCKET']
    client = session.get_session().create_client('s3', region_name=SETTINGS['ARCHIVE_REGION'],
        config=Config(signature_version='s3v4', connect_timeout=15, read_timeout=S3_READ_TIMEOUT_SECONDS,
                      retries={'max_attempts': 2, 'mode': 'standard'}))
    for row in rows:
        key = f'{run_id}/{row["path"]}'
        request = sign(client, bucket, key, row)
        # Capture errors: CalledProcessError exposes argv, which contains no URL;
        # never echo stdin, where the signed bearer URL is carried.
        subprocess.run([*ssh, 'upload'], input=json.dumps([request]), capture_output=True,
                       text=True, check=True, timeout=URL_SECONDS + S3_READ_TIMEOUT_SECONDS)
        verify(client, bucket, key, row)
        print(f'verified {row["path"]}: {row["size"]} bytes', flush=True)
    entry = dict(run_id=run_id, uri=f's3://{bucket}/{run_id}/',
                 storage_class=SETTINGS['ARCHIVE_STORAGE_CLASS'], commit=args.commit,
                 uploaded_at=datetime.now(timezone.utc).isoformat(),
                 sha256={row['path']: row['sha256'] for row in rows})
    body = (json.dumps(entry, indent=1) + '\n').encode()
    client.put_object(Bucket=bucket, Key=f'{run_id}/archive.json', Body=body,
                      IfNoneMatch='*', StorageClass=SETTINGS['ARCHIVE_STORAGE_CLASS'])
    args.receipts.mkdir(parents=True, exist_ok=True)
    receipt = args.receipts / f'{label}.json'
    with receipt.open('x') as stream:
        json.dump([entry], stream, indent=1)
        stream.write('\n')
    print(f'verified {label}: {entry["uri"]}', flush=True)


if __name__ == '__main__':
    main()
