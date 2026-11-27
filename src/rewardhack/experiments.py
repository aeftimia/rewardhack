"""Execute versioned experiment requests on RunPod, with one checkout and receipt per run."""
import csv
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from . import archive, budget, gates
from .export_transcripts import atomic_text
from .run_registry import CONFIG_FILE, new_run
from .state_files import atomic_json, file_lock


def read(path):
    return json.loads(Path(path).read_text())


def run_id_directory(state, run_id):
    if not re.fullmatch('[a-f0-9]{32}', run_id):
        raise ValueError(f'invalid experiment ID {run_id!r}')
    return Path(state) / run_id


def prepare(root, state, request):
    """Resolve a request and allocate its unique run directory before any provider call."""
    root, state = Path(root), Path(state)
    if isinstance(request, (str, Path)):
        request = read(request)
    if set(request) - {'parameters', 'inherit', 'parent', 'pod', 'driver', 'gate'}:
        raise ValueError('unknown experiment request fields')
    from .hf_grpo import RESUME_SETTINGS
    unknown = set(request['parameters']) - (set(RESUME_SETTINGS) | {'steps'})
    if unknown:
        raise ValueError(f'unknown experiment parameters: {sorted(unknown)}')
    slot = request['driver']['SMOKE_PART_ONE']
    if not re.fullmatch('[1-9][0-9]*', slot):
        raise ValueError('one training arm is required per experiment ID')
    selection = subprocess.run(['bash', '-c', 'source ops/pod_pins.env; echo "${PART_ONE_RUNS[$1]}"',
                                'training-arm', slot], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    if len(selection) != 1:
        raise ValueError('the selected pod job must run exactly one training arm')
    defaults = read(root / 'ops/run_defaults.json')
    approved = budget.nonnegative_finite(request['gate']['launch_cap'], 'requested launch cap')
    if approved > defaults['max_run_usd']:
        raise ValueError(f"run cap exceeds the approved ${defaults['max_run_usd']}")
    parent = None
    if request['parent'] is not None:
        parent_id = request['parent']['run_id']
        if '/' in parent_id:
            parent = read(root / 'ops/historical_run_settings.json')[parent_id]
        else:
            parent = read(run_id_directory(state, parent_id) / 'effective.json')
    row = new_run(defaults, request['parameters'], parent=parent,
                  checkpoint_step=request['parent']['checkpoint_step'] if parent else None,
                  inherit=request.get('inherit'))
    row['default_overrides'] = defaults['continuation']['overrides'] if parent else {}
    name, arm, penalty, variant, report = selection[0].split(':')
    for key, value in dict(arm=arm, penalty=float(penalty), prompt_variant=variant, report_mode=report).items():
        if key in row['settings'] and row['settings'][key] != value:
            raise ValueError(f'{key} differs between requested condition and RunPod job {slot}')
        row['settings'][key] = value
    row['output_arm'] = 'train-' + name
    directory = run_id_directory(state, row['run_id'])
    if parent:
        receipt = archive.archive_entry(request['parent']['archive'])
        arm = request['parent']['arm']
        if not re.fullmatch(r'train-[a-zA-Z0-9_-]+', arm):
            raise ValueError('invalid parent archive arm')
        if parent_id.endswith('/' + arm):
            # Historical receipts bind the source commit and run settings.
            if parent['launch_commit'] != receipt['commit'] or parent['settings_sha256'] != receipt['sha256'][arm + '/run-settings.json']:
                raise ValueError('archive differs from the historical predecessor')
            row['historical_commit'] = receipt['commit']
        elif '/' in parent_id or gates.file_sha256(run_id_directory(state, parent_id) / 'effective.json') != receipt['sha256'][arm + '/' + CONFIG_FILE]:
            raise ValueError('archive does not identify the requested parent')
        elif 'historical_commit' in parent:
            row['historical_commit'] = parent['historical_commit']
        row['checkpoint_arm'] = arm
        row['parent_archive'] = receipt['run_id']
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / CONFIG_FILE, row)
    atomic_json(directory / 'request.json', request)
    atomic_json(directory / 'source.json', {'commit': gates.git(root, 'rev-parse', 'HEAD').strip(),
                                           'config_sha256': gates.file_sha256(directory / CONFIG_FILE),
                                           'request_sha256': gates.file_sha256(directory / 'request.json')})
    return directory


def _checkout(root, directory):
    checkout = directory / 'checkout'
    subprocess.run(['git', 'worktree', 'add', '--detach', str(checkout), read(directory / 'source.json')['commit']], cwd=root, check=True)
    return checkout


def _stage_parent(checkout, config):
    parent = config['parent']
    if parent is None:
        return
    # Presigned URLs travel through a pipe; neither process puts them in a log.
    program = [sys.executable, str(checkout / 'ops/run_archive.py')]
    env = {**os.environ, 'PYTHONPATH': str(checkout / 'src')}
    checkpoint = f"{config['checkpoint_arm']}/trainer/checkpoint-{parent['checkpoint_step']}"
    signed = subprocess.run([*program, 'resume', config['parent_archive'], checkpoint], env=env, capture_output=True, check=True)
    subprocess.run([*program, 'fetch', str(checkout / '.tools/experiment-parent')],
                   input=signed.stdout, env=env, check=True)


def book_cost(ledger, pod_id, line, run_id, model, gpu_usd, judge_usd):
    """Publish the provider's latest per-pod total idempotently under the shared ledger lock."""
    ledger = Path(ledger)
    amount = budget.nonnegative_finite(gpu_usd, 'pod billing total')
    judge = budget.nonnegative_finite(judge_usd, 'judge usage')
    evidence = f'runpod/{pod_id}/experiment/{run_id}'
    row = dict(date=datetime.now(timezone.utc).date().isoformat(), provider='runpod',
               model=model, purpose='experiment ' + run_id, resource=pod_id, hours=0,
               hourly_usd=0, other_usd=amount + judge, total_usd=amount + judge,
               status='metered', evidence=evidence, line=line)
    with file_lock(ledger.with_suffix('.lock'), wait=budget.REGISTRY_WAIT):
        rows = budget.read_ledger(ledger, include_reservations=False)
        rows = [existing for existing in rows if existing['evidence'] != evidence]
        with atomic_text(ledger) as stream:
            writer = csv.DictWriter(stream, fieldnames=budget.FIELDS)
            writer.writeheader()
            writer.writerows([*rows, row])


def enqueue_snapshots(destination, directory, uploads, scheduled, commit):
    """Rsync renames individual files atomically. Only enqueue complete immutable saves."""
    for marker in destination.glob('train-*/adapter-step*/snapshot-files.json'):
        snapshot = marker.parent
        if snapshot in scheduled:
            continue
        files = read(marker)
        if not files or any(Path(name).name != name for name in files):
            raise ValueError(f'invalid snapshot manifest: {marker}')
        if any(not (snapshot / name).is_file() or (snapshot / name).stat().st_size != size
               for name, size in files.items()):
            continue
        copy = directory / 'snapshots' / f'{directory.name}-{snapshot.name}'
        copy.mkdir(parents=True)
        for name in files:
            os.link(snapshot / name, copy / name)
        scheduled[snapshot] = uploads.submit(archive.upload, copy, commit)


def reconcile_billing(root, directory, client):
    """Refresh the same ledger row when RunPod publishes settled charges."""
    request = read(directory / 'request.json')
    config = read(directory / CONFIG_FILE)
    result = directory / 'checkout/results' / config['run_id']
    pod_id = read(result.with_suffix('.pod.json'))['pod_id']
    receipt = client.request('GET', 'billing/pods', params={'podId': pod_id})
    atomic_json(directory / 'billing.json', receipt, exclusive=False)
    if not receipt['records'] or any(r['podId'] != pod_id for r in receipt['records']):
        raise ValueError(f'{pod_id}: awaiting attributable billing records')
    summary = result / config['output_arm'] / 'summary.json'
    data = read(summary) if summary.is_file() else {}
    usage = data.get('judge_usage')
    if 'judge_usage' not in data and float(request['driver']['JUDGE_MAX_USD']) > 0:
        raise ValueError(f'{pod_id}: missing judge usage; retain billing evidence for reconciliation')
    book_cost(root / budget.LEDGER, pod_id, request['gate']['line'], config['run_id'],
              data.get('model'), receipt['metadata']['totals']['totalAmount'],
              usage['usd'] if usage is not None else 0)


def _execute(root, directory, uploads):
    from .runpod_ops import RunPod
    request = read(directory / 'request.json')
    config = read(directory / CONFIG_FILE)
    source = read(directory / 'source.json')
    if (source['config_sha256'] != gates.file_sha256(directory / CONFIG_FILE)
            or source['request_sha256'] != gates.file_sha256(directory / 'request.json')):
        raise ValueError('prepared run configuration was changed')
    checkout = _checkout(root, directory)
    tools = checkout / '.tools'
    tools.mkdir(exist_ok=True)
    atomic_json(tools / 'experiment.json', config)
    if 'historical_commit' in config:
        with tempfile.TemporaryFile() as bundle:
            subprocess.run(['git', 'archive', config['historical_commit'], 'src', 'entrypoints/run.py'],
                           cwd=checkout, stdout=bundle, check=True)
            bundle.seek(0)
            with tarfile.open(fileobj=bundle) as files:
                files.extractall(tools / 'experiment-source', filter='data')
    _stage_parent(checkout, config)
    pod = {**request['pod'], 'name': 'rewardhack-' + config['run_id']}
    atomic_json(tools / 'pod.json', pod)
    settings = {**request['driver'], 'SMOKE_EXPERIMENT':'1', 'SMOKE_SKIP_PROBE':'1',
                'SMOKE_SECOND_RUN':'0', 'SMOKE_TOKEN_LIMIT':str(config['settings']['max_completion_length'])}
    for key, setting in [('steps','SMOKE_STEPS'), ('num_generations','SMOKE_TRAIN_GENERATIONS'),
                         ('save_every','SMOKE_SAVE_EVERY'), ('backward_batch_size','SMOKE_BACKWARD_BATCH')]:
        if key in config['settings'] and config['settings'][key] is not None:
            settings[setting] = str(config['settings'][key])
    gate = request['gate']
    cli = ['bash', 'ops/gate.sh']
    command = [*cli, 'ready', '--remote', 'ops/remote_runpod_smoke.sh']
    for key in ('deadline', 'basis', 'line', 'hourly', 'other_usd', 'launch_cap'):
        command.extend(['--' + key.replace('_', '-'), str(gate[key])])
    for key, value in settings.items():
        command.extend(['--set', f'{key}={value}'])
    subprocess.run(command, cwd=checkout, check=True)
    result = Path('results') / config['run_id']
    destination = checkout / result
    env = {**os.environ, **settings, 'PYTHONPATH':str(checkout / 'src')}
    client = RunPod()
    # Separate process confines the driver's environment to this checkout.
    snapshots = {}
    commit = source['commit']
    transfers = []
    receipt = destination.with_suffix('.pod.json')
    try:
        with (directory / 'launch.log').open('x') as log, ThreadPoolExecutor(1) as driver:
            running = driver.submit(subprocess.run, [sys.executable, 'entrypoints/runpod_ops.py', 'launch',
                                    '--config', '.tools/pod.json', '--results', str(result)], env=env, cwd=checkout,
                                    stdout=log, stderr=subprocess.STDOUT, check=True)
            while not running.done():
                wait([running], timeout=30)
                enqueue_snapshots(destination, directory, uploads, snapshots, commit)
            running.result()
            enqueue_snapshots(destination, directory, uploads, snapshots, commit)
        if not receipt.is_file():
            raise ValueError('RunPod driver completed without its pod receipt')
        if (destination / 'EXIT_CODE').read_text().strip() != '0':
            raise RuntimeError(f'{destination}: experiment failed')
        effective, = destination.glob('train-*/' + CONFIG_FILE)
        atomic_json(directory / 'effective.json', read(effective))
    finally:
        try:
            if receipt.is_file():
                client.terminate(read(receipt)['pod_id'])
        finally:
            if destination.is_dir():
                transfers.append(uploads.submit(archive.upload, destination, commit))
            if receipt.is_file():
                reconcile_billing(root, directory, client)
    return [*transfers, *snapshots.values()]


def run_batch(root, directories, *, parallelism):
    """Run prepared configs concurrently within the live balance and grant line limits."""
    from .runpod_ops import RunPod
    root = Path(root).resolve()
    pending = [Path(directory).resolve() for directory in directories]
    if not pending or parallelism < 1 or len(set(pending)) != len(pending):
        raise ValueError('batch needs distinct prepared runs and positive parallelism')
    client = RunPod()
    charges = {}
    receipts = {}
    common = Path(gates.git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir').strip())
    with file_lock(common / 'experiment-batch.lock'), ThreadPoolExecutor(parallelism) as uploads, ThreadPoolExecutor(parallelism) as workers:
        while pending or charges:
            available = Decimal(str(client.request('GET', 'billing/balance')['balanceAmount']))
            for directory in list(pending):
                if len(charges) == parallelism:
                    break
                request = read(directory / 'request.json')
                gate, driver = request['gate'], request['driver']
                hours = (gate['deadline'] + gates.overhead_minutes(root, driver)) / 60
                worst, invalid = budget.check([], gate['line'], gate['hourly'], hours,
                                              gate['other_usd'], gate['launch_cap'])
                if invalid:
                    raise ValueError(f'{directory.name}: {invalid}')
                cap = Decimal(str(worst))
                reserved = sum((cost for _, cost, _ in charges.values()), Decimal(0))
                rows = budget.read_ledger(root / budget.LEDGER)
                holds = [dict(total_usd=float(cost), status='reserved', line=line) for _, cost, line in charges.values()]
                _, problems = budget.check(rows + holds, gate['line'], 0, 0, float(cap), float(cap))
                if cap + reserved > available or problems:
                    continue
                pending.remove(directory)
                charges[workers.submit(_execute, root, directory, uploads)] = directory, cap, gate['line']
            if pending and not charges:
                raise ValueError(f'no queued job fits the current balance and grant limits: {available}, queued={pending}')
            if charges:
                finished, _ = wait(charges, return_when=FIRST_COMPLETED)
                for task in finished:
                    directory, _, _ = charges.pop(task)
                    receipts[directory] = task.result()
        for transfers in receipts.values():
            for transfer in transfers:
                transfer.result()
    return list(receipts)


def run_experiment(root, directory):
    """Execute one prepared experiment with the same lifecycle as a batch."""
    return run_batch(root, [directory], parallelism=1)[0]
