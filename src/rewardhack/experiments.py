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
from ._hashing import source_sha256
from .experiment_backups import backup_queue
from .experiment_results import registered_checkpoint
from .export_transcripts import atomic_text
from .experiment_artifacts import collect_outputs, read_effective, register_config, validate_generation_links
from .run_registry import CONFIG_FILE, checked_source, new_run, read, result_directory, run_id_directory, training_keys
from .state_files import atomic_json, file_lock, pending_pod_archive, pod_receipt


def _resolve_parent(root, state, spec):
    """Bind the selected step to its parent settings, trainer source and verified archive."""
    lineage = {'historical_snapshot_files': None}
    if spec is None:
        return None, lineage
    parent_id, step = spec['run_id'], spec['checkpoint_step']
    if type(step) is not int or step < 1:
        raise ValueError('checkpoint step must be a positive integer')
    if '/' in parent_id:
        if set(spec) != {'run_id', 'checkpoint_step', 'arm', 'archive'}:
            raise ValueError('a historical continuation needs run_id, checkpoint_step, arm and archive')
        parent = read(root / 'ops/historical_run_settings.json')[parent_id]
        if 'settings_sha256' not in parent:
            raise ValueError(f'{parent_id}: original run-settings digest is missing')
        arm = spec['arm']
        if not re.fullmatch(r'train-[a-zA-Z0-9_-]+', arm) or not parent_id.endswith('/' + arm):
            raise ValueError('archive arm differs from the historical predecessor')
        receipt = archive.archive_entry(spec['archive'])
        if parent['launch_commit'] != receipt['commit'] or parent['settings_sha256'] != receipt['sha256'][arm + '/run-settings.json']:
            raise ValueError('archive differs from the historical predecessor')
        lineage['historical_commit'] = receipt['commit']
    else:
        if set(spec) != {'run_id', 'checkpoint_step'}:
            raise ValueError('a registered continuation takes only run_id and checkpoint_step')
        parent, arm, receipt = registered_checkpoint(run_id_directory(state, parent_id), step)
        if 'historical_commit' in parent:
            lineage['historical_commit'] = parent['historical_commit']
    archive.checkpoint_members(receipt, arm, f'{arm}/trainer/checkpoint-{step}')
    lineage.update(checkpoint_arm=arm, parent_archive=receipt['run_id'])
    if 'historical_commit' in lineage:
        prefix = f'{arm}/adapter-step{step}/'
        names = sorted(path.removeprefix(prefix) for path in receipt['sha256'] if path.startswith(prefix))
        if not names:
            raise ValueError('archived historical checkpoint has no matching adapter snapshot')
        lineage['historical_snapshot_files'] = names
    return parent, lineage


def prepare(root, request, *, state=None):
    """Resolve a request and allocate its unique run directory before any provider call."""
    root = Path(root)
    state = root / 'experiments' if state is None else Path(state)
    if isinstance(request, (str, Path)):
        request = read(request)
    if set(request) - {'parameters', 'inherit', 'parent', 'pod', 'driver', 'gate'}:
        raise ValueError('unknown experiment request fields')
    unknown = set(request['parameters']) - training_keys()
    if unknown:
        raise ValueError(f'unknown experiment parameters: {sorted(unknown)}')
    slot = request['driver']['SMOKE_PART_ONE']
    driver = request['driver']
    for key, value in {'SMOKE_PROBE_LARGE':'0', 'SMOKE_SECOND_RUN':'0', 'SMOKE_SKIP_PROBE':'1'}.items():
        if key in driver and driver[key] != value:
            raise ValueError(f'{key} conflicts with the single configured experiment')
    if 'SMOKE_MEMORY_LIMIT' in driver:
        raise ValueError('a separate memory measurement requires its own experiment config')
    if not re.fullmatch('[1-9][0-9]*', slot):
        raise ValueError('one training arm is required per experiment ID')
    jobs = (root / 'ops/part_one_runs.txt').read_text().splitlines()
    if int(slot) >= len(jobs):
        raise ValueError(f'unknown training job {slot}')
    selection = jobs[int(slot)].split()
    if len(selection) != 1:
        raise ValueError('the selected pod job must run exactly one training arm')
    defaults = read(root / 'ops/run_defaults.json')
    approved = budget.nonnegative_finite(request['gate']['launch_cap'], 'requested launch cap')
    if approved > defaults['max_run_usd']:
        raise ValueError(f"run cap exceeds the approved ${defaults['max_run_usd']}")
    parent, lineage = _resolve_parent(root, state, request['parent'])
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
    row.update(lineage)
    if 'historical_commit' in row and row['settings']['rollout_backend'] != 'vllm':
        raise ValueError('exact historical trainer capture requires the vLLM sampler')
    directory = run_id_directory(state, row['run_id'])
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
    if source_sha256() != source_sha256(checkout / 'src/rewardhack'):
        raise ValueError('the installed experiment backend differs from the prepared commit; run with its rewardhack package')
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
    atomic_json(checkout / '.tools/parent-download.json', json.loads(signed.stdout))


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


def enqueue_snapshots(destination, directory, submit, scheduled, historical_files=None):
    """Rsync renames individual files atomically. Only enqueue complete immutable saves."""
    for receipt in (directory / 'snapshots').glob(f'{directory.name}-adapter-step*/archive.json'):
        if receipt not in scheduled:
            submit(receipt.parent.name.removeprefix(directory.name + '-'))
            scheduled.add(receipt)
    for snapshot in destination.glob('train-*/adapter-step*'):
        if snapshot in scheduled:
            continue
        marker = snapshot / 'snapshot-files.json'
        if historical_files is not None:
            files = dict.fromkeys(historical_files)
        elif marker.is_file():
            files = read(marker)
        else:
            continue
        if not files or any(Path(name).name != name for name in files):
            raise ValueError(f'invalid snapshot manifest: {marker}')
        if any(not (snapshot / name).is_file() or (size is not None and (snapshot / name).stat().st_size != size)
               for name, size in files.items()):
            continue
        copy = directory / 'snapshots' / f'{directory.name}-{snapshot.name}'
        copy.mkdir(parents=True)
        for name in files:
            os.link(snapshot / name, copy / name)
        submit(snapshot.name)
        scheduled.add(snapshot)


def reconcile_billing(root, directory, client):
    """Refresh the same ledger row when RunPod publishes settled charges."""
    request = read(directory / 'request.json')
    config = read(directory / CONFIG_FILE)
    result = result_directory(directory)
    pod_id = read(pod_receipt(result))['pod_id']
    receipt = client.read('billing/pods', podId=pod_id)
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


def _execute(root, directory, checkout, submit, client):
    from .runpod_ops import launch
    request = read(directory / 'request.json')
    config = read(directory / CONFIG_FILE)
    tools = checkout / '.tools'
    tools.mkdir(exist_ok=True)
    atomic_json(checkout / gates.EXPERIMENT_CONFIG, config)
    if 'historical_commit' in config:
        with tempfile.TemporaryFile() as bundle:
            subprocess.run(['git', 'archive', config['historical_commit'], 'src', 'entrypoints/run.py'],
                           cwd=checkout, stdout=bundle, check=True)
            bundle.seek(0)
            with tarfile.open(fileobj=bundle) as files:
                files.extractall(tools / 'experiment-source', filter='data')
    _stage_parent(checkout, config)
    pod = {**request['pod'], 'name': 'rewardhack-' + config['run_id']}
    settings = {**request['driver'], 'SMOKE_EXPERIMENT':'1', 'SMOKE_SKIP_PROBE':'1',
                'SMOKE_SECOND_RUN':'0', 'SMOKE_TOKEN_LIMIT':str(config['settings']['max_completion_length'])}
    for key, setting in [('steps','SMOKE_STEPS'), ('num_generations','SMOKE_TRAIN_GENERATIONS'),
                         ('save_every','SMOKE_SAVE_EVERY'), ('backward_batch_size','SMOKE_BACKWARD_BATCH')]:
        if key in config['settings'] and config['settings'][key] is not None:
            settings[setting] = str(config['settings'][key])
    gate = request['gate']
    _, problems = gates.ready(checkout, 'ops/remote_runpod_smoke.sh', gate['deadline'], gate['basis'], gate['line'],
                              gate['hourly'], gate['other_usd'], gate['launch_cap'], settings)
    if problems:
        raise ValueError('\n'.join(problems))
    result = Path('results') / config['run_id']
    destination = checkout / result
    snapshots = set()
    receipt = pod_receipt(destination)
    try:
        with (directory / 'launch.log').open('x') as log, ThreadPoolExecutor(1) as driver:
            running = driver.submit(launch, client, checkout, pod, result, settings=settings, log=log)
            while True:
                done, _ = wait([running], timeout=30)
                enqueue_snapshots(destination, directory, submit, snapshots, config['historical_snapshot_files'])
                if done:
                    break
            running.result()
        if not receipt.is_file():
            raise ValueError('RunPod driver completed without its pod receipt')
        if (destination / 'EXIT_CODE').read_text().strip() != '0':
            raise RuntimeError(f'{destination}: experiment failed')
        # dry-ok: src/rewardhack/experiment_artifacts.py::read_effective | The equality check lives entirely in read_effective. This caller supplies the run's file and persists the returned document. Changing the comparison only requires editing the helper.
        effective = read_effective(destination / config['output_arm'] / CONFIG_FILE, config)
        atomic_json(directory / 'effective.json', effective)
    finally:
        try:
            if receipt.is_file() and not pending_pod_archive(destination):
                client.terminate(read(receipt)['pod_id'])
        finally:
            try:
                if destination.is_dir():
                    index = collect_outputs(directory)
                    submit('outputs')
                    if index['exit_code'] == 0:
                        validate_generation_links(directory, index)
            finally:
                if receipt.is_file():
                    reconcile_billing(root, directory, client)


def _run(root, directory, client):
    source = checked_source(directory)
    register_config(directory, source)
    checkout = _checkout(root, directory)
    try:
        with backup_queue(directory) as submit:
            _execute(root, directory, checkout, submit, client)
    finally:
        (checkout / '.tools/parent-download.json').unlink(missing_ok=True)


def _run_prepared_batch(root, directories, parallelism):
    """Run prepared configs concurrently within the live balance and grant line limits."""
    from .runpod_ops import RunPod, available_with_top_up
    root = Path(root).resolve()
    pending = [Path(directory).resolve() for directory in directories]
    if not pending or parallelism < 1 or len(set(pending)) != len(pending):
        raise ValueError('batch needs distinct prepared runs and positive parallelism')
    client = RunPod()
    charges = {}
    common = Path(gates.git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir').strip())
    with file_lock(common / 'experiment-batch.lock'), ThreadPoolExecutor(parallelism) as workers:
        while pending or charges:
            available = available_with_top_up(client.read('billing/balance')['balanceAmount'])
            for directory in list(pending):
                if len(charges) == parallelism:
                    break
                request = read(directory / 'request.json')
                gate, driver = request['gate'], request['driver']
                _, hours = gates.launch_duration(root, gate['deadline'], driver)
                worst, invalid = budget.check([], gate['line'], gate['hourly'], hours,
                                              gate['other_usd'], gate['launch_cap'])
                if invalid:
                    raise ValueError(f'{directory.name}: {invalid}')
                cap = Decimal(str(worst))
                reserved = sum((cost for _, cost, _ in charges.values()), Decimal(0))
                rows = budget.read_ledger(root / budget.LEDGER)
                holds = [dict(total_usd=float(cost), status='reserved', line=line) for _, cost, line in charges.values()]
                problems = budget.reservation_problems(rows + holds, gate['line'], float(cap))
                if cap + reserved > available or problems:
                    continue
                pending.remove(directory)
                charges[workers.submit(_run, root, directory, client)] = directory, cap, gate['line']
            if pending and not charges:
                raise ValueError(f'no queued job fits the current balance and grant limits: {available}, queued={pending}')
            if charges:
                finished, _ = wait(charges, return_when=FIRST_COMPLETED)
                for task in finished:
                    charges.pop(task)
                    task.result()


def run_batch(configs, *, parallelism, root=None, state=None):
    """Register the configs, execute them, and return run IDs in request order."""
    root = Path.cwd() if root is None else Path(root)
    directories = [prepare(root, config, state=state) for config in configs]
    _run_prepared_batch(root, directories, parallelism)
    return [directory.name for directory in directories]


def experiment(config, *, root=None, state=None):
    """Register and execute a config. Its run ID identifies inputs, outputs and checkpoints."""
    run_id, = run_batch([config], parallelism=1, root=root, state=state)
    return run_id
