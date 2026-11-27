"""Pod side of the GPT resume backup: hash an immutable snapshot, then PUT signed objects.

Credentials stay on the coordinator. Upload requests arrive only on stdin. This
one-off operational helper deliberately does not change the recorded trainer code.
"""
import argparse
import base64
import hashlib
import http.client
import json
import signal
import struct
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path.home() / 'out'
MAX_BYTES = 5_000_000_000
UPLOAD_SECONDS = 900
# Mirrored from the trainer because this helper runs with the pod's stdlib Python;
# test_resume_requirements_match_trainer guards the copy without importing torch here.
CHECKPOINT_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
RESUME_RUN_FILES = ('run-settings.json', 'training-rollouts.jsonl', 'steps.jsonl', 'summary.json', 'before-evaluation.jsonl')


def owned_file(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name:
        raise ValueError('invalid archive path')
    path = ROOT / name
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('archive input is missing or symlinked')
    if ROOT.resolve() not in path.resolve().parents:
        raise ValueError('archive input escapes output directory')
    return path


def describe(name):
    path = owned_file(name)
    before = path.stat()
    if not 0 < before.st_size <= MAX_BYTES:
        raise ValueError('object exceeds single-PUT size bounds')
    if path.suffix == '.safetensors':
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            if header_size > min(before.st_size - 8, 10_000_000):
                raise ValueError('incomplete safetensors header')
            header = json.loads(stream.read(header_size))
        expected = 8 + header_size + max(value['data_offsets'][1]
                                        for key, value in header.items() if key != '__metadata__')
        if before.st_size != expected:
            raise ValueError('incomplete safetensors payload')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('archive input changed during hashing')
    return dict(path=name, size=before.st_size, sha256=digest)


def expired(_signum, _frame):
    raise TimeoutError('object upload exceeded its elapsed-time bound')


def completed_files(run):
    if run not in ('train-baseline', 'train-lambda0', 'train-lambda05'):
        raise ValueError('unexpected training directory')
    if (ROOT / 'EXIT_CODE').read_text().strip() != '0':
        raise ValueError('job has not completed successfully')
    summary = json.loads(owned_file(f'{run}/summary.json').read_text())
    if summary['status'] != 'completed':
        raise ValueError('training has not completed successfully')
    checkpoint = ROOT / run / 'trainer' / f'checkpoint-{summary["global_step"]}'
    for name in (*CHECKPOINT_FILES, 'trainer_state.json'):
        owned_file((checkpoint / name).relative_to(ROOT).as_posix())
    for name in RESUME_RUN_FILES:
        owned_file(f'{run}/{name}')
    paths = sorted((ROOT / run).rglob('*'), key=lambda path: (not path.is_relative_to(checkpoint), str(path)))
    return [describe(path.relative_to(ROOT).as_posix())
            for path in paths if path.is_file() or path.is_symlink()]


def upload(request):
    actual = describe(request['path'])
    if actual != {key: request[key] for key in ('path', 'size', 'sha256')}:
        raise ValueError('archive input differs from signed request')
    url = urlsplit(request['url'])
    if (url.scheme != 'https' or not url.hostname or not url.hostname.endswith('.amazonaws.com')
            or url.username or url.password or url.port not in (None, 443) or url.fragment):
        raise ValueError('unexpected upload destination')
    headers = request['headers']
    expected = base64.b64encode(bytes.fromhex(actual['sha256'])).decode()
    if (headers['x-amz-checksum-sha256'] != expected
            or headers['Content-Length'] != str(actual['size'])
            or headers['If-None-Match'] != '*'):
        raise ValueError('signed headers do not bind the input')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(UPLOAD_SECONDS)
    connection = http.client.HTTPSConnection(url.hostname, timeout=60)
    try:
        connection.putrequest('PUT', url.path + '?' + url.query)
        for key, value in headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        with owned_file(actual['path']).open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        # Never print the signed URL or an S3 response body. A 412 is only a
        # candidate existing object; the coordinator must verify its checksum.
        if response.status not in (200, 412):
            raise RuntimeError(f'S3 upload returned HTTP {response.status}')
        status = response.status
    finally:
        connection.close()
        signal.alarm(0)
    if describe(actual['path']) != actual:
        raise ValueError('archive input changed while uploading')
    return dict(actual, http_status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['describe', 'describe-completed', 'upload'])
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args()
    if args.mode == 'describe':
        print(json.dumps([describe(name) for name in args.paths]))
    elif args.mode == 'describe-completed':
        if len(args.paths) != 1:
            parser.error('describe-completed needs one training directory')
        print(json.dumps(completed_files(args.paths[0])))
    else:
        import sys
        requests = json.load(sys.stdin)
        for request in requests:
            print(json.dumps(upload(request)), flush=True)


if __name__ == '__main__':
    main()
