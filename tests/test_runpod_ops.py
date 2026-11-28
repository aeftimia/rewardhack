"""Exercise billing transitions with a fake service, never a paid test pod."""
import json
import os
import subprocess
import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests import Timeout
from runpod.error import QueryError

from rewardhack import runpod_ops as ops

POD = {'id': 'test-pod', 'name': 'test-run', 'status': 'RUNNING', 'cost': 2.69, 'disk': 250, 'mounts': {},
       'gpu': {'id': 'test-gpu', 'count': 1}, 'dataCenterId': 'test-dc',
       'ssh': {'direct': {'host': '192.0.2.1', 'port': 12345, 'username': 'root'}},
       'env': {'SECRET': 'must-not-print'}}
CONFIG = {'name': 'test-run', 'image': 'runpod/pytorch:test-tag', 'gpu': POD['gpu'],
          'disk': 250, 'ports': ['22/tcp'], 'cloud': 'SECURE'}


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    (tmp_path / '.tools').mkdir()
    record = {'record_id': 'test-record', 'settings': {}, 'remote': 'ops/remote_test.sh',
              'deadline_minutes': 60, 'hourly_usd': 2.725, 'worst_case_usd': 5}
    (tmp_path / ops.gates.RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(ops.gates, 'record_problems', lambda *a, **k: [])
    monkeypatch.setattr(ops.gates, 'pod_pin', lambda *a: 1)
    monkeypatch.setattr(ops.gates, 'mark_launched', lambda *a, **kw: None)
    monkeypatch.setattr(ops.time, 'sleep', lambda *a: None)

    class Service:
        def __init__(self):
            self.calls = []
            self.pod_value = deepcopy(POD)
            self.balance = 10
            self.removed = []

        def request(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            return {('GET', 'billing/balance'): {'balanceAmount': self.balance},
                    ('GET', 'pods'): {'pods': []}, ('POST', 'pods'): self.pod_value}[(method, path)]

        def pod(self, pod_id):
            return self.pod_value

        def read(self, path, **params):
            return self.request('GET', path)

        def collection(self, path, field, **params):
            return self.request('GET', path)[field]

        def terminate(self, pod_id):
            self.removed.append(pod_id)

    return tmp_path, deepcopy(CONFIG), Service()


def test_sdk_transport_uses_explicit_key_and_bounded_calls(monkeypatch):
    monkeypatch.setenv('RUNPOD_API_KEY', 'test-only')
    monkeypatch.delenv('RUNPOD_API_BASE_URL', raising=False)
    client = ops.RunPod()
    calls = []
    monkeypatch.setattr(client, 'transport', lambda *a, **k: calls.append((a, k)))
    client.request('POST', 'pods', json=CONFIG)
    args, kwargs = calls[0]
    assert args == ('POST', '/v2/pods')
    assert kwargs['api_key'] == 'test-only'
    assert kwargs['timeout'] == ops.API_SECONDS
    assert kwargs['max_retries'] == 0
    monkeypatch.setenv('RUNPOD_API_BASE_URL', 'https://untrusted.invalid')
    with pytest.raises(ValueError, match='must be unset'):
        ops.RunPod()


def test_status_does_not_expose_environment(deployment):
    _, _, client = deployment
    original = client.request
    client.request = lambda method, path: {'pods': [POD]} if path == 'pods' else original(method, path)
    result = ops.status(client)
    assert result['estimated_pod_hourly_usd'] == 2.725
    assert 'must-not-print' not in json.dumps(result)


def test_no_pod_created_when_approval_refused(deployment, monkeypatch):
    root, config, client = deployment
    monkeypatch.setattr(ops.gates, 'record_problems', lambda *a, **k: ['approval refused'])
    with pytest.raises(ValueError, match='approval refused'):
        ops.launch(client, root, config, 'results/test')
    assert client.calls == []


def test_insufficient_live_balance_never_creates_pod(deployment):
    root, config, client = deployment
    client.balance = 1
    with pytest.raises(ValueError, match='Live balance'):
        ops.launch(client, root, config, 'results/test')
    assert not any(method == 'POST' for method, _, _ in client.calls)


@pytest.mark.parametrize('failure', ['price', 'driver', 'endpoint', 'record'])
def test_failed_startup_removes_pod_even_if_record_write_fails(deployment, monkeypatch, failure):
    root, config, client = deployment
    if failure == 'price':
        client.pod_value['cost'] = 20
    elif failure == 'endpoint':
        client.pod_value['ssh']['direct'] = None
        client.pod_value['status'] = 'ERROR'
    else:
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(2, 'driver')
        monkeypatch.setattr(ops, 'drive', fail)
        if failure == 'record':
            monkeypatch.setattr(ops.gates, 'mark_launched', fail)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        ops.launch(client, root, config, 'results/test')
    assert client.removed == [POD['id']]


def test_success_hands_exact_endpoint_and_record_to_driver(deployment, monkeypatch):
    root, config, client = deployment
    calls = []
    monkeypatch.setattr(ops, 'drive', lambda *a: calls.append(a))
    monkeypatch.delenv('SMOKE_STEPS', raising=False)
    result = ops.launch(client, root, config, 'results/test', settings={'SMOKE_STEPS': '3'})
    assert result['driver_completed']
    assert calls[0][2:4] == (('192.0.2.1', '12345'), 'results/test')
    assert calls[0][4]['SMOKE_STEPS'] == '3' and 'SMOKE_STEPS' not in os.environ
    assert client.calls[-1][2]['json']['startSsh'] is True
    assert client.removed == []
    assert json.loads((root / 'results/test.pod.json').read_text()) == {
        'pod_id': POD['id'], 'record_id': 'test-record'}


def test_deployment_uses_the_validated_snapshot_without_rereading(deployment, monkeypatch):
    root, config, client = deployment
    read = Path.read_text
    reads = []
    def changed(path, *args, **kwargs):
        content = read(path, *args, **kwargs)
        if path == root / ops.gates.RECORD:
            reads.append(path)
            if len(reads) > 1:
                return json.dumps({**json.loads(content), 'remote': 'ops/unapproved.sh'})
        return content
    monkeypatch.setattr(Path, 'read_text', changed)
    calls = []
    monkeypatch.setattr(ops, 'drive', lambda *args: calls.append(args))
    ops.launch(client, root, config, 'results/test')
    assert len(reads) == 1
    assert calls[0][1]['remote'] == 'ops/remote_test.sh'


def test_receipt_failure_still_removes_created_pod(deployment, monkeypatch):
    root, config, client = deployment
    def unwritable(*args, **kwargs):
        raise OSError('receipt disk full')
    monkeypatch.setattr(ops, 'atomic_json', unwritable)
    with pytest.raises(OSError, match='receipt disk full'):
        ops.launch(client, root, config, 'results/test')
    assert client.removed == [POD['id']]


def test_log_write_failure_preserves_receipt_and_removes_pod(deployment):
    root, config, client = deployment
    class FullLog:
        def write(self, text):
            raise OSError('log disk full')
    with pytest.raises(OSError, match='log disk full'):
        ops.launch(client, root, config, 'results/test', log=FullLog())
    assert json.loads((root / 'results/test.pod.json').read_text())['pod_id'] == POD['id']
    assert client.removed == [POD['id']]


def test_ambiguous_create_is_not_retried_and_matching_pod_is_removed(deployment, monkeypatch):
    root, config, client = deployment
    original = client.request
    posts = []
    reads = []
    def request(method, path, **kwargs):
        if method == 'POST':
            posts.append(path)
            raise Timeout('lost response')
        if path == 'pods':
            reads.append(path)
            return {'pods': [] if len(reads) == 1 else [POD]}
        return original(method, path, **kwargs)
    monkeypatch.setattr(client, 'request', request)
    with pytest.raises(Timeout):
        ops.launch(client, root, config, 'results/test')
    assert posts == ['pods']
    assert client.removed == [POD['id']]
    assert json.loads((root / 'results/test.pod.json').read_text())['pod_id'] == POD['id']


def test_termination_retries_network_failure_and_checks_absence(monkeypatch):
    client = ops.RunPod.__new__(ops.RunPod)
    calls = []
    def request(*args):
        calls.append(args)
        if len(calls) == 1:
            raise Timeout('temporary')
    monkeypatch.setattr(client, 'request', request)
    monkeypatch.setattr(client, 'pod', lambda pod_id: None)
    monkeypatch.setattr(ops.time, 'sleep', lambda *a: None)
    assert client.terminate('test-pod')['removed']
    assert len(calls) == 2
    monkeypatch.setattr(client, 'pod', lambda pod_id: POD)
    with pytest.raises(RuntimeError, match='unconfirmed'):
        client.terminate('test-pod')


@pytest.mark.parametrize('status', [403, 404])
def test_only_actual_404_is_absence(monkeypatch, status):
    from runpod.api import ctl_commands
    client = ops.RunPod.__new__(ops.RunPod)
    client.key = 'test-only'
    def refused(method, path, *, api_key):
        assert (method, path, api_key) == ('GET', '/v2/pods/test-pod', client.key)
        raise QueryError('pod lookup failed', method, status_code=status)
    monkeypatch.setattr(ctl_commands, 'run_rest_request', refused)
    if status == 404:
        assert client.pod('test-pod') is None
    else:
        with pytest.raises(QueryError):
            client.pod('test-pod')


@pytest.mark.parametrize('status, attempts', [(403, 1), (503, 2)])
def test_read_retries_only_transient_failures(monkeypatch, status, attempts):
    client = ops.RunPod.__new__(ops.RunPod)
    calls = []
    def request(method, path, **kwargs):
        calls.append((method, path))
        if len(calls) == 1:
            raise QueryError('read failed', 'GET', status_code=status)
        return {'balanceAmount': 10}
    monkeypatch.setattr(client, 'request', request)
    monkeypatch.setattr(ops.time, 'sleep', lambda *a: None)
    if status == 403:
        with pytest.raises(QueryError):
            client.read('billing/balance')
    else:
        assert client.read('billing/balance') == {'balanceAmount': 10}
    assert calls == [('GET', 'billing/balance')] * attempts


def test_stopped_pod_retains_only_persistent_storage_charges():
    pod = deepcopy(POD)
    pod['status'] = 'EXITED'
    pod['mounts'] = {'persistent': {'size': 100, 'path': '/workspace'}}
    assert float(ops.hourly(pod)) == 0.028


def test_collection_follows_pages_and_refuses_repeated_cursor(monkeypatch):
    client = ops.RunPod.__new__(ops.RunPod)
    calls = []
    def request(method, path, **kwargs):
        calls.append(kwargs['params'])
        return {'pods': [POD], 'pagination': {'hasNextPage': len(calls) == 1, 'nextCursor': 'next'}}
    monkeypatch.setattr(client, 'request', request)
    assert len(client.collection('pods', 'pods')) == 2
    assert calls == [{}, {'cursor': 'next'}]
    monkeypatch.setattr(client, 'request', lambda *a, **k: {
        'pods': [], 'pagination': {'hasNextPage': True, 'nextCursor': 'repeated'}})
    with pytest.raises(ValueError, match='Invalid pagination'):
        client.collection('pods', 'pods')


def test_catalog_is_unpaginated_and_billing_filter_is_forwarded():
    cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'entrypoints/runpod_ops.py'))
    calls = []
    client = SimpleNamespace(read=lambda *a, **k: calls.append((a, k)))
    cli['read'](SimpleNamespace(command='gpus', count=2), client)
    assert calls[-1] == (('catalog/gpus',), {'include': 'AVAILABILITY', 'product': 'POD', 'count': 2})
    cli['read'](SimpleNamespace(command='billing', pod_id='test-pod'), client)
    assert calls[-1] == (('billing/pods',), {'podId': 'test-pod'})
