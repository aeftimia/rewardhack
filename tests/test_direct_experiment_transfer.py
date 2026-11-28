"""Exercise the pod's GET and PUT paths with an in-memory S3 transport."""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


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

    def ssh(args, input, **kwargs):
        modes = {'snapshots': lambda: {}, 'bulk': lambda: [pod.describe(name)],
                 'upload': lambda: pod.upload(json.loads(input))}
        return SimpleNamespace(stdout=json.dumps(modes[args[-1]]()))
    monkeypatch.setattr(driver.subprocess, 'run', ssh)
    driver.transfer('final', result, ['ssh-pod'], S3())
    receipt = json.loads(result.with_suffix('.direct.json').read_text())
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
