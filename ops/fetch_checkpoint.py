"""Restore S3 objects before the pod installs trainer dependencies."""
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from pod_snapshot_upload import SOCKET_TIMEOUT


def download(folder, row):
    digest, name, signed = row
    url = urlsplit(signed)
    if url.scheme != 'https' or not url.hostname or not url.hostname.endswith('.amazonaws.com'):
        raise ValueError('invalid S3 host')
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = http.client.HTTPSConnection(url.hostname, timeout=SOCKET_TIMEOUT)
    hasher = hashlib.sha256()
    try:
        connection.request('GET', url.path + '?' + url.query)
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f'{name}: S3 returned HTTP {response.status}')
        with path.open('xb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                hasher.update(chunk)
    finally:
        connection.close()
    if hasher.hexdigest() != digest:
        raise ValueError(f'{name}: archived SHA-256 mismatch')
    print(f'verified {name} ({path.stat().st_size} bytes)', flush=True)


def fetch(destination, rows):
    if not rows:
        raise ValueError('empty resume manifest')
    for _, name, _ in rows:
        if set(name.split('/')) & {'', '.', '..'} or '/' not in name:
            raise ValueError('invalid checkpoint member')
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        stage = Path(scratch) / 'parent'
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda row: download(stage, row), rows))
        stage.rename(destination)


if __name__ == '__main__':
    fetch(Path(sys.argv[1]), json.load(sys.stdin))
