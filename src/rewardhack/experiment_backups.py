"""One detached S3 transfer worker per experiment, with durable per-artifact jobs."""
import fcntl
import json
import os
import re
import select
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Thread

from .state_files import atomic_json


def snapshot_step(name):
    match = re.fullmatch(r'adapter-step([1-9][0-9]*)', name)
    if not match:
        raise ValueError(f'invalid snapshot name {name!r}')
    return int(match[1])


@contextmanager
def backup_queue(directory):
    """Yield submit(artifact). Closing the producer releases the worker to finish and exit."""
    directory = Path(directory).resolve(strict=True)
    queue = directory / 'backups'
    queue.mkdir(exist_ok=True)
    source = directory / 'checkout/src'
    if not (source / 'rewardhack/experiment_backups.py').is_file():
        raise ValueError(f'missing backup worker in experiment checkout {source}')
    with (queue / 'worker.lock').open('a') as lock, (queue / 'worker.log').open('ab') as log:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'another backup worker owns {directory}') from error
        # The child inherits the locked file description. Closing the parent's copy leaves it
        # locked until the detached worker exits, including after a producer crash.
        process = subprocess.Popen([sys.executable, '-m', 'rewardhack.experiment_backups', str(directory)],
                                   cwd=directory, env={**os.environ, 'PYTHONPATH':str(source)},
                                   stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock.fileno(),))

    submissions = Lock()

    def submit(name):
        if name != 'outputs':
            snapshot_step(name)
        path = queue / (name + '.json')
        with submissions:
            if process.stdin.closed:
                raise RuntimeError('backup queue is closed')
            if not path.exists():
                atomic_json(path, dict(run_id=directory.name, artifact=name, status='queued'))
            if process.poll() is not None:
                raise RuntimeError(f'backup worker exited with {process.returncode}; inspect {queue}/worker.log')

    try:
        yield submit
    finally:
        with submissions:
            process.stdin.close()
        # Reap the detached child when this interpreter stays up for later runs.
        Thread(target=process.wait, daemon=True).start()


def retry_backups(directory):
    """Start a worker to finish pending, interrupted or failed uploads without starting a pod."""
    with backup_queue(directory):
        pass


def _upload(directory, path):
    from .experiment_artifacts import archive_outputs, archive_snapshot
    job = json.loads(path.read_text())
    name = path.stem
    if job['artifact'] != name or job['run_id'] != directory.name:
        raise ValueError(f'{path}: job does not belong to this run and artifact')
    if job['status'] == 'verified':
        return job
    if job['status'] not in ('queued', 'uploading', 'failed'):
        raise ValueError(f'{path}: invalid backup job status')
    record = dict(run_id=directory.name, artifact=name)
    atomic_json(path, dict(record, status='uploading'), exclusive=False)
    try:
        if name == 'outputs':
            archive_outputs(directory)
        else:
            archive_snapshot(directory, name)
    except (OSError, ValueError, subprocess.SubprocessError, SystemExit) as error:
        # The archiver raises these on failed transfers and verification. Record that failure
        # for retry; keep processing other artifacts, then exit non-zero.
        detail = f'{type(error).__name__}: {error}'
        record.update(status='failed', error=detail)
        print(f'{path}: {detail}', file=sys.stderr, flush=True)
    else:
        record['status'] = 'verified'
    atomic_json(path, record, exclusive=False)
    return record


def _work(directory):
    queue = directory / 'backups'
    attempted = set()
    success = True
    producer_closed = False
    while True:
        jobs = sorted(set(queue.glob('*.json')) - attempted, key=lambda path: (path.stem != 'outputs', path.name))
        if jobs:
            path = jobs[0]
            attempted.add(path)
            if _upload(directory, path)['status'] == 'failed':
                success = False
        elif producer_closed:
            return 0 if success else 1
        else:
            # stdin carries no data. EOF means the producer closed or died. Scan once more
            # after EOF so its last durably queued job cannot be lost between scan and close.
            producer_closed = bool(select.select([sys.stdin], [], [], 1)[0])


if __name__ == '__main__':
    sys.exit(_work(Path(sys.argv[1])))
