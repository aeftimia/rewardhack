import csv
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from rewardhack import budget, experiments
from rewardhack.run_registry import CONFIG_FILE, new_run
from rewardhack.state_files import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def request():
    return dict(parameters={'steps': 6}, parent=None, driver={'SMOKE_PART_ONE':'9', 'JUDGE_MAX_USD':'1'},
                pod={}, gate=dict(line='gpt-oss-20b-gpu', launch_cap=15, deadline=136, basis='test',
                                  hourly=2.725, other_usd=1))


def test_per_key_inheritance_and_identity():
    defaults = experiments.read(ROOT / 'ops/run_defaults.json')
    parent = new_run(defaults, {'seed':5, 'max_completion_length':8192})
    child = new_run(defaults, {'steps':60}, parent=parent, checkpoint_step=6, inherit={'max_completion_length':False})
    assert child['settings'] == dict(steps=60, seed=5, max_completion_length=16384)
    assert child['parent'] == {'run_id':parent['run_id'], 'checkpoint_step':6}
    assert child['run_id'] != parent['run_id']
    defaults['continuation'].update(inherit=False, inherit_parameters={'seed':True})
    configured = new_run(defaults, {}, parent=parent, checkpoint_step=6)
    assert configured['settings'] == {'seed':5, 'max_completion_length':16384}
    assert configured['inherit'] == {'seed':True, 'max_completion_length':False}
    with pytest.raises(ValueError, match='inheritance'):
        new_run(defaults, {}, inherit=True)


def test_prepare_binds_continuation_to_the_archived_parent(tmp_path, monkeypatch):
    parent = experiments.prepare(ROOT, request(), state=tmp_path)
    effective = experiments.read(parent / CONFIG_FILE)
    atomic_json(parent / 'effective.json', effective)
    arm = effective['output_arm']
    digest = experiments.gates.file_sha256(parent / 'effective.json')
    receipt = {'run_id':'immutable-archive', 'sha256':{f'{arm}/{CONFIG_FILE}':digest}}
    monkeypatch.setattr(experiments.archive, 'archive_entry', lambda source: receipt)
    spec = request()
    spec.update(parameters={'steps':60}, parent={'run_id':parent.name, 'arm':arm, 'checkpoint_step':6, 'archive':'alias'})
    directory = experiments.prepare(ROOT, spec, state=tmp_path)
    config = experiments.read(directory / CONFIG_FILE)
    assert config['parent'] == {'run_id':parent.name, 'checkpoint_step':6}
    assert config['parent_archive'] == 'immutable-archive'
    assert directory != parent


@pytest.mark.parametrize('exit_code', ['0', '1'])
def test_launch_archive_teardown_and_reconciled_ledger(tmp_path, monkeypatch, exit_code):
    directory = experiments.prepare(ROOT, request(), state=tmp_path / 'registry')
    checkout = directory / 'checkout'
    checkout.mkdir()
    root = tmp_path / 'project'
    ledger = root / budget.LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text(','.join(budget.FIELDS) + '\n')
    monkeypatch.setattr(experiments, '_checkout', lambda *args: checkout)
    config = experiments.read(directory / CONFIG_FILE)
    result = checkout / 'results' / config['run_id']
    pod_id = 'one-pod'
    terminated, archived = [], []
    client = SimpleNamespace(terminate=terminated.append, request=lambda *a, **kw: {
        'records':[{'podId':pod_id}], 'metadata':{'totals':{'totalAmount':0.3}}})
    monkeypatch.setitem(sys.modules, 'rewardhack.runpod_ops', SimpleNamespace(RunPod=lambda:client))
    def invoke(command, **kwargs):
        if 'ready' in command:
            assert 'SMOKE_TOKEN_LIMIT=16384' in command
            return
        arm = result / config['output_arm']
        arm.mkdir(parents=True)
        (result / 'EXIT_CODE').write_text(exit_code)
        atomic_json(arm / CONFIG_FILE, config)
        atomic_json(arm / 'summary.json', dict(model='test', judge_usage={'usd':0.1}))
        atomic_json(result.with_suffix('.pod.json'), {'pod_id':pod_id})
    monkeypatch.setattr(experiments.subprocess, 'run', invoke)
    monkeypatch.setattr(experiments.archive, 'upload', lambda path, commit: archived.append(path))
    with ThreadPoolExecutor(1) as uploads:
        if exit_code == '0':
            for future in experiments._execute(root, directory, uploads):
                future.result()
        else:
            with pytest.raises(RuntimeError, match='experiment failed'):
                experiments._execute(root, directory, uploads)
    experiments.reconcile_billing(root, directory, client)
    assert terminated == [pod_id] and archived == [result]
    with ledger.open() as stream:
        row, = csv.DictReader(stream)
    assert float(row['total_usd']) == pytest.approx(0.4)
    assert row['evidence'].endswith(config['run_id'])


@pytest.mark.parametrize('historical_files', [None, ['model.bin']])
def test_snapshot_waits_for_all_files_and_only_uploads_once(tmp_path, monkeypatch, historical_files):
    destination = tmp_path / 'out'
    snapshot = destination / 'train-a/adapter-step1'
    snapshot.mkdir(parents=True)
    if historical_files is None:
        atomic_json(snapshot / 'snapshot-files.json', {'model.bin':3})
    sent, scheduled = [], {}
    monkeypatch.setattr(experiments.archive, 'upload', lambda p, commit: sent.append((p/'model.bin').read_bytes()))
    with ThreadPoolExecutor(1) as pool:
        def scan():
            experiments.enqueue_snapshots(destination, tmp_path, pool, scheduled, 'commit', historical_files)
        scan()
        assert not scheduled
        (snapshot / 'model.bin').write_bytes(b'abc')
        scan()
        scan()
    assert sent == [b'abc']


def test_real_training_entrypoint_consumes_prepared_settings(tmp_path, monkeypatch):
    from rewardhack import hf_grpo
    spec = request()
    spec['driver']['SMOKE_PART_ONE'] = '8'
    directory = experiments.prepare(ROOT, spec, state=tmp_path / 'registry')
    output = tmp_path / 'train-lambda0'
    calls = []
    monkeypatch.setattr(hf_grpo, 'run', calls.append)
    script = ROOT / 'entrypoints/run.py'
    monkeypatch.setattr(sys, 'argv', [str(script), 'train', '--out', str(output), '--lean', 'lean',
                                     '--run-config', str(directory / CONFIG_FILE)])
    runpy.run_path(str(script), run_name='__main__')
    args, = calls
    assert args.max_completion_length == 16384 and args.steps == 6 and args.prompt_variant == 'refute'
    effective = experiments.read(output / CONFIG_FILE)
    assert effective['run_id'] == directory.name and effective['settings']['seed'] == 42


def test_batch_skips_unaffordable_head_and_charges_once(tmp_path, monkeypatch):
    client = SimpleNamespace(request=lambda *args: {'balanceAmount':5})
    monkeypatch.setitem(sys.modules, 'rewardhack.runpod_ops', SimpleNamespace(RunPod=lambda:client))
    a, b = tmp_path / 'large', tmp_path / 'small'
    for folder, hourly in [(a, 9), (b, 1)]:
        folder.mkdir()
        doc = request()
        doc['gate'].update(deadline=60, hourly=hourly, other_usd=0)
        atomic_json(folder / 'request.json', doc)
    monkeypatch.setattr(experiments.gates, 'overhead_minutes', lambda *args:0)
    monkeypatch.setattr(experiments.gates, 'git', lambda *args:str(tmp_path))
    monkeypatch.setattr(budget, 'read_ledger', lambda *args:[])
    ran = []
    def execute(root, directory, uploads):
        ran.append(directory)
        return []
    monkeypatch.setattr(experiments, '_execute', execute)
    with pytest.raises(ValueError, match='no queued job fits'):
        experiments.run_batch(ROOT, [a, b], parallelism=2)
    assert ran == [b]
