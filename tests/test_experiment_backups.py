import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from rewardhack import experiment_backups
from rewardhack.state_files import file_lock


def test_caller_exits_then_worker_finishes_other_jobs_and_retries_failure(tmp_path):
    directory = tmp_path / ('a' * 32)
    source = directory / 'checkout/src'
    package = source / 'rewardhack'
    package.mkdir(parents=True)
    library = Path(experiment_backups.__file__).parent
    for name in ('__init__.py', 'state_files.py', 'experiment_backups.py'):
        shutil.copy(library / name, package / name)
    (package / 'experiment_artifacts.py').write_text("""
import time

def archive_snapshot(directory, name):
    with (directory / 'calls').open('a') as log:
        print(name, file=log)
    if not (directory / 'repaired').exists():
        raise ValueError('checksum mismatch')

def archive_outputs(directory):
    with (directory / 'calls').open('a') as log:
        print('outputs', file=log)
    end = time.monotonic() + 10
    while not (directory / 'release').exists():
        if time.monotonic() >= end:
            raise TimeoutError('test producer failed to release transfer')
        time.sleep(.01)
""")
    producer = """
from pathlib import Path
from rewardhack.experiment_backups import backup_queue
with backup_queue(Path.cwd()) as submit:
    submit('outputs')
    submit('outputs')
    submit('adapter-step6')
"""
    subprocess.run([sys.executable, '-c', producer], cwd=directory,
                   env={**os.environ, 'PYTHONPATH':str(source)}, check=True, timeout=8)
    queue = directory / 'backups'
    with pytest.raises(ValueError, match='another backup worker'):
        experiment_backups.retry_backups(directory)
    (directory / 'release').touch()
    with file_lock(queue / 'worker.lock', wait=10):
        assert json.loads((queue / 'outputs.json').read_text())['status'] == 'verified'
        failure = json.loads((queue / 'adapter-step6.json').read_text())
        assert failure['status'] == 'failed' and 'checksum mismatch' in failure['error']
    (directory / 'repaired').touch()
    experiment_backups.retry_backups(directory)
    with file_lock(queue / 'worker.lock', wait=10):
        assert json.loads((queue / 'adapter-step6.json').read_text()) == {
            'status':'verified', 'run_id':directory.name, 'artifact':'adapter-step6'}
        calls = (directory / 'calls').read_text().splitlines()
        assert calls.count('outputs') == 1 and calls.count('adapter-step6') == 2
