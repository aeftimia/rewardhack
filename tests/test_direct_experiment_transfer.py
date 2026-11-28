"""Exercise the pod's GET and PUT paths with an in-memory S3 transport."""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError


def test_driver_routes_resume_and_final_archive(tmp_path, monkeypatch):
    from conftest import driver_script
    from test_round2_regressions import drive
    root = Path(driver_script(tmp_path)).parents[1]
    staged = root / '.tools'
    staged.mkdir()
    (staged / 'parent-download.json').write_text('[]')
    interpreter = tmp_path / 'coordinator'
    interpreter.write_text('#!/bin/sh\n[ "$2" = final ] && echo receipt > "$3.direct.json"\n')
    interpreter.chmod(0o755)
    monkeypatch.setenv('EXPERIMENT_PYTHON', str(interpreter))
    monkeypatch.setenv('SMOKE_EXPERIMENT', '1')
    code, calls, result = drive(tmp_path)
    assert code == 0, result.stdout + result.stderr
    assert any('fetch_checkpoint.py' in cmd for cmd in calls)
    assert not any(cmd.startswith('rsync') and 'experiment-parent' in cmd for cmd in calls)
    pulls = [cmd for cmd in calls if cmd.startswith('rsync') and ':./' in cmd]
    assert pulls and all('--exclude=adapter*/' in cmd and '--exclude=trainer/checkpoint-*/' in cmd for cmd in pulls)
    assert (tmp_path / 'results.direct.json').is_file()


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'ops'))
    import direct_experiment_archive
    import fetch_checkpoint
    import pod_snapshot_upload
    return direct_experiment_archive, fetch_checkpoint, pod_snapshot_upload


@pytest.mark.parametrize('outcome', ['ok', '412', 'timeout', 'wrong-hash', 'wrong-size', 'missing', 'denied'])
def test_slow_copy_only_accepts_verified_destination(modules, outcome):
    driver, _, pod = modules
    row = dict(path='train-a/adapter-step2/weights', sha256='12'*32, size=37)
    heads = []
    def error(status, code):
        return ClientError({'Error': {'Code': code}, 'ResponseMetadata': {'HTTPStatusCode': status}}, 'CopyObject')
    def copy(**args):
        assert args['IfNoneMatch'] == '*' and args['CopySource']['Key'] == 'snapshot/weights'
        if outcome == 'ok':
            return {'ResponseMetadata': {'RetryAttempts': 0}}
        if outcome == 'timeout':
            raise ReadTimeoutError(endpoint_url='https://example.invalid')
        if outcome == 'denied':
            raise error(403, 'AccessDenied')
        raise error(412, 'PreconditionFailed')
    def head(**args):
        heads.append(args)
        if outcome == 'missing':
            raise error(404, 'NotFound')
        return {'ChecksumSHA256': 'different' if outcome == 'wrong-hash' else pod.s3_checksum(row),
                'ContentLength': 0 if outcome == 'wrong-size' else row['size']}
    client = SimpleNamespace(copy_object=copy, head_object=head)
    if outcome in ('denied', 'missing', 'wrong-size', 'wrong-hash'):
        with pytest.raises((ClientError, ValueError)):
            driver.copy_snapshot(client, 'snapshot/weights', 'outputs/weights', row)
    else:
        driver.copy_snapshot(client, 'snapshot/weights', 'outputs/weights', row)
    assert heads == ([] if outcome == 'denied' else [
        {'Bucket': driver.archive.BUCKET, 'Key': 'outputs/weights', 'ChecksumMode': 'ENABLED'}])


def test_archive_retry_reuses_prefix_only_for_identical_inventory(modules, tmp_path):
    driver, _, _ = modules
    result = tmp_path / 'experiment'
    first = '20261128T043709Z-experiment-outputs-abcdef12'
    second = '20261128T044109Z-experiment-outputs-34567890'
    rows = [dict(path='a', size=1, sha256='a'*64), dict(path='b', size=2, sha256='b'*64)]
    assert driver.final_attempt(result, rows, first) == (first, False)
    assert driver.final_attempt(result, rows[::-1], second) == (first, True)
    rows[0]['sha256'] = 'c'*64
    assert driver.final_attempt(result, rows, second) == (second, False)


@pytest.mark.parametrize('corrupt', [False, True])
def test_verified_roundtrip_without_local_weights(tmp_path, monkeypatch, modules, corrupt):
    driver, fetcher, pod = modules
    checkout = tmp_path / 'registry/id/checkout'
    result = checkout / 'results/id'
    result.mkdir(parents=True)
    (checkout / '.tools').mkdir()
    (checkout / '.tools/experiment.json').write_text(json.dumps(dict(run_id='id', output_arm='train-a')))
    monkeypatch.chdir(checkout)
    disk = tmp_path / 'pod'
    name = 'train-a/trainer/checkpoint-1/optimizer.pt'
    (disk / name).parent.mkdir(parents=True)
    payload = b'optimizer-state'
    (disk / name).write_bytes(payload)
    monkeypatch.setattr(pod, 'ROOT', disk)
    bucket = {}

    class Transport:
        def __init__(self, host, timeout):
            self.body = bytearray()
        def putrequest(self, method, path):
            self.key = path.split('?')[0].lstrip('/')
        def putheader(self, key, value): pass
        def endheaders(self): pass
        def send(self, data): self.body.extend(data)
        def request(self, method, path): self.putrequest(method, path)
        def close(self): pass
        def getresponse(self):
            if self.body:
                bucket[self.key] = bytes(self.body)
            body = io.BytesIO(bucket[self.key])
            return SimpleNamespace(status=200, read=body.read)

    monkeypatch.setattr(pod.http.client, 'HTTPSConnection', Transport)
    monkeypatch.setattr(pod.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(pod.signal, 'alarm', lambda *args: None)
    class S3:
        def generate_presigned_url(self, operation, Params, **kwargs):
            return f"https://bucket.s3.amazonaws.com/{Params['Key']}?X-Amz-SignedHeaders=content-length;if-none-match;x-amz-checksum-sha256;x-amz-storage-class"
        def head_object(self, Bucket, Key, ChecksumMode):
            data = bucket[Key]
            digest = hashlib.sha256(data).digest()
            return dict(ContentLength=len(data), ChecksumSHA256=base64.b64encode(digest).decode())

    actions = []
    def ssh(args, input, **kwargs):
        actions.append(args[-1])
        modes = {'snapshots': lambda: {}, 'bulk': lambda: [pod.describe(name)],
                 'upload': lambda: pod.upload(json.loads(input))}
        return SimpleNamespace(stdout=json.dumps(modes[args[-1]]()))
    monkeypatch.setattr(driver.subprocess, 'run', ssh)
    driver.transfer('final', result, ['ssh-pod'], S3())
    receipt = json.loads(result.with_suffix('.direct.json').read_text())
    actions.clear()
    driver.transfer('final', result, ['ssh-pod'], S3())
    assert 'upload' not in actions
    assert json.loads(result.with_suffix('.direct.json').read_text()) == receipt
    assert not (result / name).exists()
    key = f"{receipt['run_id']}/{name}"
    if corrupt:
        bucket[key] = b'corrupt'
    restored = tmp_path / 'next-pod'
    rows = [[receipt['sha256'][name], name, 'https://bucket.s3.amazonaws.com/' + key]]
    if corrupt:
        with pytest.raises(ValueError, match='SHA-256 mismatch'):
            fetcher.fetch(restored, rows)
        assert not restored.exists()
    else:
        fetcher.fetch(restored, rows)
        assert (restored / name).read_bytes() == payload
