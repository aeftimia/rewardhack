import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from rewardhack import budget, experiment_recovery as recovery, experiments
from rewardhack.state_files import atomic_json, pod_receipt
from rewardhack.run_registry import CONFIG_FILE, result_directory


@pytest.fixture
def abandoned(tmp_path):
    root = tmp_path/'project'
    subprocess.run(['git', 'init', '-q', str(root)], check=True, timeout=10)
    (root/'ops').mkdir()
    shutil.copy(Path(__file__).resolve().parents[1]/'ops/pod_pins.env', root/'ops')
    directory = root/'experiments'/('a'*32)
    arm = result_directory(directory)/'arm'
    arm.mkdir(parents=True)
    atomic_json(directory/CONFIG_FILE, dict(run_id=directory.name, output_arm='arm'))
    atomic_json(directory/'request.json', dict(driver={'JUDGE_MAX_USD':'1'}, gate={'deadline':60,'line':'gpt-oss-20b-gpu'}))
    atomic_json(arm/'summary.json', dict(model='test', judge_usage={'usd':0.1}))
    atomic_json(pod_receipt(result_directory(directory)), {'pod_id':'one'})
    (directory/'launch.log').touch()
    ledger = root/budget.LEDGER
    ledger.parent.mkdir()
    ledger.write_text(','.join(budget.FIELDS)+'\n')
    hold = experiments._reserve_experiment(root, directory, {'line':'gpt-oss-20b-gpu'}, 10, None)
    return root, directory, hold


def test_sigkill_triggers_detached_recovery_and_single_ledger_entry(abandoned, tmp_path):
    root, directory, hold = abandoned
    # Keep the real observer process, CLI, polling, and on-disk state. Replace
    # only the provider boundary in the child interpreter's package copy.
    source = tmp_path/'src'
    package = source/'rewardhack'
    shutil.copytree(Path(recovery.__file__).parent, package, ignore=shutil.ignore_patterns('__pycache__'))
    (package/'runpod_ops.py').write_text("""
from pathlib import Path
from rewardhack import experiments
class RunPod:
    def pod(self, pod_id):
        assert pod_id == 'one'
        return None
    def read(self, path, *, podId):
        assert path == 'billing/pods' and podId == 'one'
        marker = Path('seen-first-query')
        if not marker.exists():
            marker.touch()
            experiments.BILLING_POLL_SECONDS = 0
            return {'records': []}
        return {'records':[{'podId':'one'}], 'metadata':{'totals':{'totalAmount':0.3}}}
""")
    producer = """
import os, signal, sys
from pathlib import Path
from rewardhack.experiment_recovery import watch_batch
from rewardhack.state_files import file_lock
root, directory = map(Path, sys.argv[1:])
with watch_batch(root, [directory]), file_lock(directory/'coordinator.lock'):
    os.kill(os.getpid(), signal.SIGKILL)
"""
    process = subprocess.run([sys.executable, '-c', producer, str(root), str(directory)], cwd=tmp_path,
                             env={**os.environ, 'PYTHONPATH':str(source)}, timeout=20)
    assert process.returncode == -9
    expires = time.monotonic()+20
    while hold.exists() and time.monotonic()<expires:
        time.sleep(.05)
    assert not hold.exists(), (directory/'recovery.log').read_text()
    assert (source/'seen-first-query').exists()
    rows = budget.read_ledger(root/budget.LEDGER)
    row, = rows
    assert row['evidence'] == f'runpod/one/experiment/{directory.name}'
    assert float(row['total_usd']) == pytest.approx(.4)
    recovery.recover(root, directory, None)
    assert budget.read_ledger(root/budget.LEDGER) == rows


def test_live_pod_waits_and_missing_usage_keeps_hold(abandoned, monkeypatch):
    root, directory, hold = abandoned
    seen = []
    states = iter([{'id':'one'}, None])
    def lookup(pod):
        seen.append('lookup')
        return next(states)
    def bill(path, **kwargs):
        seen.append('bill')
        return {'records':[{'podId':'one'}], 'metadata':{'totals':{'totalAmount':1}}}
    monkeypatch.setattr(recovery.time, 'sleep', lambda seconds: seen.append('wait'))
    (result_directory(directory)/'arm/summary.json').write_text(json.dumps({'judge_usage':None}))
    with pytest.raises(ValueError, match='missing judge usage'):
        recovery.recover(root, directory, SimpleNamespace(pod=lookup, read=bill))
    assert seen == ['lookup','wait','lookup','bill']
    assert hold.exists() and budget.read_ledger(root/budget.LEDGER) == []


def test_crash_after_booking_replays_without_duplicate(abandoned):
    root, directory, hold = abandoned
    experiments.book_cost(root/budget.LEDGER, 'one', 'gpt-oss-20b-gpu', directory.name, 'test', .3, .1)
    client = SimpleNamespace(pod=lambda _:None, read=lambda *a, **k:{
        'records':[{'podId':'one'}], 'metadata':{'totals':{'totalAmount':.3}}})
    recovery.recover(root, directory, client)
    row, = budget.read_ledger(root/budget.LEDGER)
    assert float(row['total_usd']) == pytest.approx(.4) and not hold.exists()


def test_launch_without_pod_identity_never_clears_hold(abandoned):
    root, directory, hold = abandoned
    pod_receipt(result_directory(directory)).unlink()
    with pytest.raises(ValueError, match='ambiguous launch'):
        recovery.recover(root, directory, None)
    assert hold.exists()
