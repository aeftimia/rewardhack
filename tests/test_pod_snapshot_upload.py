"""Direct snapshot uploads must bind actual bytes and fail before publishing receipts."""
import importlib.util
import base64
import json
import struct
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('pod_snapshot_upload',
    Path(__file__).resolve().parents[1] / 'ops/pod_snapshot_upload.py')
pod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pod)


def test_resume_requirements_match_trainer():
    from rewardhack import hf_grpo
    assert pod.CHECKPOINT_FILES == hf_grpo.CHECKPOINT_FILES
    assert pod.RESUME_RUN_FILES == hf_grpo.RESUME_RUN_FILES


def test_owned_files_reject_escape_symlink_and_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, 'ROOT', tmp_path)
    (tmp_path / 'empty').touch()
    (tmp_path / 'data').write_bytes(b'content')
    (tmp_path / 'link').symlink_to(tmp_path / 'data')
    for name in ('../data', '/etc/passwd', 'link', 'empty'):
        with pytest.raises(ValueError):
            pod.describe(name)


def test_partial_safetensors_not_treated_as_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, 'ROOT', tmp_path)
    header = json.dumps({'tensor': {'data_offsets': [0, 4]}}).encode()
    path = tmp_path / 'adapter_model.safetensors'
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'12')
    with pytest.raises(ValueError, match='payload'):
        pod.describe(path.name)
    path.write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
    assert pod.describe(path.name)['size'] == path.stat().st_size


def test_changed_input_rejected_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, 'ROOT', tmp_path)
    path = tmp_path / 'data'
    path.write_bytes(b'original')
    request = pod.describe(path.name)
    path.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='differs'):
        pod.upload(request)


def test_unexpected_destination_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, 'ROOT', tmp_path)
    (tmp_path / 'data').write_bytes(b'content')
    request = dict(pod.describe('data'), url='https://attacker.invalid/upload')
    with pytest.raises(ValueError, match='destination'):
        pod.upload(request)


def test_completed_archive_requires_finished_job_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(pod, 'ROOT', tmp_path)
    (tmp_path / 'EXIT_CODE').write_text('1')
    with pytest.raises(ValueError, match='job'):
        pod.completed_files('train-baseline')
    (tmp_path / 'EXIT_CODE').write_text('0')
    run = tmp_path / 'train-baseline'
    run.mkdir()
    (run / 'summary.json').write_text(json.dumps(dict(status='running', global_step=6)))
    with pytest.raises(ValueError, match='training'):
        pod.completed_files('train-baseline')
    (run / 'summary.json').write_text(json.dumps(dict(status='completed', global_step=6)))
    with pytest.raises(ValueError, match='missing'):
        pod.completed_files('train-baseline')


@pytest.mark.parametrize('status', [200, 412, 403])
def test_upload_requires_success_or_verified_existing_candidate(tmp_path, monkeypatch, status):
    from types import SimpleNamespace
    monkeypatch.setattr(pod, 'ROOT', tmp_path)
    (tmp_path / 'data').write_bytes(b'content')
    row = pod.describe('data')
    request = dict(row, url='https://bucket.s3.amazonaws.com/key?signature=test', headers={
        'Content-Length': str(row['size']), 'If-None-Match': '*',
        'x-amz-checksum-sha256': base64.b64encode(bytes.fromhex(row['sha256'])).decode()})
    sent = bytearray()

    class Connection:
        def putrequest(self, *args): pass
        def putheader(self, *args): pass
        def endheaders(self): pass
        def send(self, data): sent.extend(data)
        def getresponse(self): return SimpleNamespace(status=status)
        def close(self): pass

    monkeypatch.setattr(pod.http.client, 'HTTPSConnection', lambda *args, **kwargs: Connection())
    if status == 403:
        with pytest.raises(RuntimeError, match='403'):
            pod.upload(request)
    else:
        assert pod.upload(request) == dict(row, http_status=status)
    assert sent == b'content'
