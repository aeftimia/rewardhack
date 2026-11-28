"""Complete abandoned experiment accounting after the coordinator's pipe closes."""
import os
import select
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Thread

from . import budget, gates
from .experiments import experiment_hold_directory, reconcile_billing, BILLING_POLL_SECONDS
from .run_registry import CONFIG_FILE, read, result_directory
from .state_files import file_lock, pod_receipt


@contextmanager
def watch_batch(root, directories):
    """One independent local observer, started before any job is admitted."""
    directories = [Path(d).resolve() for d in directories]
    source = Path(__file__).resolve().parents[1]
    with (directories[0] / 'recovery.log').open('ab') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'rewardhack.experiment_recovery', str(root.resolve()), *map(str, directories)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, start_new_session=True,
            cwd=source, env={**os.environ, 'PYTHONPATH':str(source)})
    try:
        if not select.select([process.stdout], [], [], budget.REGISTRY_WAIT)[0] or process.stdout.readline() != b'ready\n':
            raise RuntimeError(f'recovery observer failed to start; inspect {log.name}')
        yield
    finally:
        process.stdin.close()
        process.stdout.close()
        Thread(target=process.wait, daemon=True).start()


def recover(root, directory, client):
    """Use only positive local run identity and provider-confirmed absence."""
    registry = experiment_hold_directory(root)
    hold = registry / (directory.name + '.json')
    with file_lock(directory / 'coordinator.lock'):
        if not hold.exists():
            return
        config = read(directory / CONFIG_FILE)
        if config['run_id'] != directory.name:
            raise ValueError(f'{directory}: inconsistent run identity')
        receipt = pod_receipt(result_directory(directory))
        if receipt.exists():
            pod_id = read(receipt)['pod_id']
            request = read(directory / 'request.json')
            minutes, _ = gates.launch_duration(root, request['gate']['deadline'], request['driver'])
            expires = time.monotonic() + minutes * 60
            while client.pod(pod_id) is not None:
                remaining = expires - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'{pod_id}: removal unconfirmed; spend hold retained')
                time.sleep(min(BILLING_POLL_SECONDS, remaining))
            reconcile_billing(root, directory, client)
        elif (directory / 'launch.log').exists():
            raise ValueError(f'{directory}: ambiguous launch without pod receipt; spend hold retained')
        with file_lock(registry / 'admission.lock', wait=budget.REGISTRY_WAIT):
            hold.unlink()


def after_disconnect(root, directories):
    """Recover every owned hold independently; one unbilled pod cannot delay its peers."""
    from .runpod_ops import RunPod
    registry = experiment_hold_directory(root)
    pending = [d for d in directories if (registry / (d.name + '.json')).exists()]
    if pending:
        with ThreadPoolExecutor(min(16, len(pending))) as workers:
            jobs = [workers.submit(recover, root, d, RunPod()) for d in pending]
            failures = [error for job in jobs if (error := job.exception()) is not None]
        if failures:
            raise ExceptionGroup('experiment accounting recovery failed', failures)


if __name__ == '__main__':
    print('ready', flush=True)
    if sys.stdin.buffer.read(1):
        raise ValueError('expected coordinator disconnect')
    after_disconnect(Path(sys.argv[1]), [Path(arg) for arg in sys.argv[2:]])
