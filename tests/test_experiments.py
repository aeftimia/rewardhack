import csv
import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rewardhack import budget, experiments, runpod_ops
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


def test_exact_one_time_cap_cannot_be_reused_in_another_registry(tmp_path, monkeypatch):
    import json
    original_git = experiments.gates.git
    approval_id = 'a' * 32
    spec = request()
    spec['gate']['launch_cap'] = 25.07
    with pytest.raises(ValueError, match='cap exceeds'):
        experiments.prepare(ROOT, spec, state=tmp_path)
    spec['approval'] = approval_id
    approved = json.dumps(spec)
    def git(root, *args):
        if args == ('show', f'HEAD:ops/approved-runs/{approval_id}.json'):
            return approved
        if args[-1] == '--git-common-dir':
            return str(tmp_path)
        return original_git(root, *args)
    monkeypatch.setattr(experiments.gates, 'git', git)
    spec['parameters']['steps'] = 60
    with pytest.raises(ValueError, match='differs'):
        experiments.prepare(ROOT, spec, state=tmp_path/'first')
    spec['parameters']['steps'] = 6
    assert experiments.prepare(ROOT, spec, state=tmp_path/'first').name == approval_id
    with pytest.raises(FileExistsError):
        experiments.prepare(ROOT, spec, state=tmp_path/'second')


def test_staged_historical_trainer_reads_its_committed_vllm_pin(tmp_path):
    import subprocess
    (tmp_path/'.tools').mkdir()
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    for name in ('src/rewardhack/vllm_rollouts.py','entrypoints/run.py','requirements-vllm.txt'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('vllm_rollouts.py'):
            path.write_text("from pathlib import Path\ndef pinned_vllm():\n    return (Path(__file__).resolve().parents[2]/'requirements-vllm.txt').read_text()\n")
        else:
            path.write_text('vllm==fixture\n')
    subprocess.run(['git','add','.'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],
                   cwd=tmp_path,check=True)
    experiments.stage_historical_trainer(tmp_path, 'HEAD')
    assert (tmp_path/'.tools/experiment-source/requirements-vllm.txt').read_text() == 'vllm==fixture\n'


@pytest.mark.parametrize('local', [True, False])
def test_prepare_binds_continuation_to_the_archived_parent(tmp_path, monkeypatch, local):
    import json
    from rewardhack import archive
    parent = experiments.prepare(ROOT, request(), state=tmp_path)
    effective = experiments.read(parent / CONFIG_FILE)
    atomic_json(parent / 'effective.json', effective)
    arm = effective['output_arm']
    digest = experiments.gates.file_sha256(parent / 'effective.json')
    archive_id = '20261127T230000Z-parent-01234567'
    checkpoint = {f'{arm}/trainer/checkpoint-6/{f}':'a'*64 for f in (*archive.CHECKPOINT_FILES, 'trainer_state.json')}
    receipt = {'run_id':archive_id, 'commit':'b'*40, 'sha256':{f'{arm}/{CONFIG_FILE}':digest, **checkpoint,
               **{f'{arm}/{f}':'c'*64 for f in archive.RESUME_RUN_FILES}}}
    index = dict(run_id=parent.name, archive_id=archive_id, uri=f's3://{archive.BUCKET}/{archive_id}/',
                 backup_status='verified', source_commit=receipt['commit'], effective_config=f'{arm}/{CONFIG_FILE}',
                 effective_config_sha256=digest, checkpoints={'trainer':{'6':checkpoint}})
    if local:
        atomic_json(parent / 'outputs.json', index)
    else:
        effective_bytes = (parent / 'effective.json').read_bytes()
        (parent / 'effective.json').unlink()
        def fetch(_, command, remote, *args, timeout):
            if command == 'cat':
                assert remote == f's3:{archive.BUCKET}/experiments/{parent.name}/outputs.json'
                return json.dumps(index)
            assert command == 'copyto' and remote.endswith(f'/{archive_id}/{arm}/{CONFIG_FILE}')
            Path(args[0]).write_bytes(effective_bytes)
        monkeypatch.setattr(archive, 'run', fetch)
    monkeypatch.setattr(experiments.archive, 'archive_entry', lambda source: receipt)
    spec = request()
    spec.update(parameters={'steps':60}, parent={'run_id':parent.name, 'checkpoint_step':6})
    directory = experiments.prepare(ROOT, spec, state=tmp_path)
    config = experiments.read(directory / CONFIG_FILE)
    assert config['parent'] == {'run_id':parent.name, 'checkpoint_step':6}
    assert config['parent_archive'] == archive_id and config['checkpoint_arm'] == arm
    assert directory != parent
    spec['parent']['checkpoint_step'] = 7
    with pytest.raises(ValueError, match='no archived trainer checkpoint'):
        experiments.prepare(ROOT, spec, state=tmp_path)


def test_summary_only_parent_refused_before_any_archive_request(tmp_path):
    root = tmp_path / 'project'
    (root / 'ops').mkdir(parents=True)
    key = 'historical/train-baseline'
    atomic_json(root / 'ops/historical_run_settings.json', {key:{'settings':{}, 'launch_commit':None}})
    with pytest.raises(ValueError, match='original run-settings digest is missing'):
        experiments._resolve_parent(root, tmp_path, dict(run_id=key, arm='train-baseline', checkpoint_step=6, archive='unused'))


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
    client = SimpleNamespace(terminate=terminated.append, read=lambda *a, **kw: {
        'records':[{'podId':pod_id}], 'metadata':{'totals':{'totalAmount':0.3}}})
    def ready(root, remote, deadline, basis, line, hourly, other_usd, launch_cap, settings):
        assert root == checkout and settings['SMOKE_TOKEN_LIMIT'] == '16384'
        return {}, []
    monkeypatch.setattr(experiments.gates, 'ready', ready)
    def invoke(provider, root, pod, results, *, settings, log):
        assert provider is client and root / results == result
        assert settings['SMOKE_TOKEN_LIMIT'] == '16384' and pod['name'] == 'rewardhack-' + config['run_id']
        arm = result / config['output_arm']
        arm.mkdir(parents=True)
        (result / 'EXIT_CODE').write_text(exit_code)
        atomic_json(arm / CONFIG_FILE, config)
        atomic_json(arm / 'summary.json', dict(model='test', judge_usage={'usd':0.1}))
        for stage, stream in [('before', 'before-evaluation'), ('training', 'training-rollouts'), ('after', 'after-evaluation')]:
            identity = dict(run_id=directory.name, phase=stage, generation_id=f'{directory.name}/{stage}/1/task/0')
            atomic_json(arm / (stream + '.jsonl'), dict(identity, raw=stage))
            atomic_json(arm / (stage + '-generations.jsonl'), identity)
        atomic_json(result.with_suffix('.pod.json'), {'pod_id':pod_id})
    monkeypatch.setattr(runpod_ops, 'launch', invoke)
    if exit_code == '0':
        experiments._execute(root, directory, checkout, archived.append, client)
    else:
        with pytest.raises(RuntimeError, match='experiment failed'):
            experiments._execute(root, directory, checkout, archived.append, client)
    experiments.reconcile_billing(root, directory, client)
    assert terminated == [pod_id] and archived == ['outputs']
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
    sent, scheduled = [], set()
    def scan():
        experiments.enqueue_snapshots(destination, tmp_path, sent.append, scheduled, historical_files)
    scan()
    assert not scheduled
    (snapshot / 'model.bin').write_bytes(b'abc')
    scan()
    scan()
    assert sent == ['adapter-step1']
    assert (tmp_path / 'snapshots' / f'{tmp_path.name}-adapter-step1/model.bin').read_bytes() == b'abc'


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


def test_experiment_registers_config_rows_and_checkpoints_under_one_id(tmp_path, monkeypatch):
    from test_archive import tools
    from rewardhack import archive
    from rewardhack.experiment_results import experiment_rows

    env = tools(tmp_path)
    for key in ('PATH', 'FAKE_S3'):
        monkeypatch.setenv(key, env[key])
    state = tmp_path / 'registry'
    bucket = tmp_path / 's3' / archive.BUCKET
    provider = SimpleNamespace(read=lambda *args: {'balanceAmount':15}, terminate=lambda pod: None)
    monkeypatch.setattr(runpod_ops, 'RunPod', lambda:provider)
    git = experiments.gates.git
    monkeypatch.setattr(experiments.gates, 'git', lambda root, *args:
                        str(tmp_path) if '--git-common-dir' in args else git(root, *args))
    monkeypatch.setattr(budget, 'read_ledger', lambda *args: [])
    monkeypatch.setattr(experiments, 'reconcile_billing', lambda *args: None)
    def checkout(root, directory):
        target = directory / 'checkout'
        target.mkdir()
        (target / 'src').symlink_to(root / 'src', target_is_directory=True)
        return target
    monkeypatch.setattr(experiments, '_checkout', checkout)
    monkeypatch.setattr(experiments.gates, 'ready', lambda *args: ({}, []))
    def driver(client, cwd, pod, results, *, settings, log):
        config = experiments.read(cwd / '.tools/experiment.json')
        run_id = config['run_id']
        assert experiments.read(bucket / 'experiments' / run_id / 'config.json')['config'] == config
        result = cwd / 'results' / run_id
        arm = result / config['output_arm']
        arm.mkdir(parents=True)
        atomic_json(arm / CONFIG_FILE, config)
        for phase, stream in [('before','before-evaluation'), ('training','training-rollouts'), ('after','after-evaluation')]:
            identity = dict(step=1, run_id=run_id, phase=phase, generation_id=f'{run_id}/{phase}/1/task/0')
            atomic_json(arm / (stream + '.jsonl'), dict(identity, raw=phase))
            atomic_json(arm / (phase + '-generations.jsonl'), dict(identity,
                        messages=[dict(role='user', content='question')], prompt_token_ids=[1], completion_token_ids=[2]))
        snapshot = arm / 'adapter-step1'
        snapshot.mkdir()
        (snapshot / 'weights').write_bytes(b'weights')
        atomic_json(snapshot / 'snapshot-files.json', {'weights':7})
        trainer = arm / 'trainer/checkpoint-1'
        trainer.mkdir(parents=True)
        atomic_json(trainer / 'trainer_state.json', {'global_step':1})
        atomic_json(result.with_suffix('.pod.json'), {'pod_id':'fixture'})
        (result / 'EXIT_CODE').write_text('0')
    monkeypatch.setattr(runpod_ops, 'launch', driver)
    run_id = experiments.experiment(request(), root=ROOT, state=state)
    example, = experiment_rows(run_id, 'training', stream='generations', state=state)
    assert example['completion_token_ids'] == [2]
    remote = bucket / 'experiments' / run_id
    limit = time.monotonic() + 30
    jobs = [state / run_id / 'backups' / (name + '.json') for name in ('outputs', 'adapter-step1')]
    while time.monotonic() < limit and any(experiments.read(job)['status'] != 'verified' for job in jobs):
        time.sleep(.05)
    index = experiments.read(remote / 'outputs.json')
    assert index['run_id'] == run_id and index['backup_status'] == 'verified'
    assert index['effective_config_sha256'] == experiments.gates.file_sha256(state / run_id / 'effective.json')
    assert '1' in index['checkpoints']['trainer']
    assert experiments.read(remote / 'checkpoints/adapter-step1.json')['run_id'] == run_id
    assert (bucket / index['archive_id'] / index['effective_config']).is_file()


@pytest.mark.parametrize('auto_reload', [False, True])
def test_batch_skips_unaffordable_head_and_charges_once(tmp_path, monkeypatch, auto_reload):
    monkeypatch.setenv(runpod_ops.AUTO_RELOAD, str(int(auto_reload)))
    client = SimpleNamespace(read=lambda *args: {'balanceAmount':5})
    monkeypatch.setattr(runpod_ops, 'RunPod', lambda:client)
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
    def execute(root, directory, client):
        ran.append(directory)
        return []
    monkeypatch.setattr(experiments, '_run', execute)
    if auto_reload:
        experiments._run_prepared_batch(ROOT, [a, b], parallelism=2)
        assert set(ran) == {a, b}
        return
    with pytest.raises(ValueError, match='no queued job fits'):
        experiments._run_prepared_batch(ROOT, [a, b], parallelism=2)
    assert ran == [b]
