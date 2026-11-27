"""Cleanup must have a real deadline, including interpreter shutdown."""
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rewardhack import pilot_runner as runner


@pytest.fixture
def clock(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    monkeypatch.setattr(runner, 'CLEANUP_SECONDS', 0.03)
    return elapsed


def test_reap_stubborn_child_stops_at_deadline(monkeypatch, clock):
    calls = []
    def waitpid(pid, options):
        calls.append((pid, options))
        return 0, 0
    monkeypatch.setattr(runner.os, 'waitpid', waitpid)
    with pytest.raises(RuntimeError, match='termination is unconfirmed'):
        runner._reap(12345)
    assert calls and all(call == (12345, os.WNOHANG) for call in calls)
    assert clock[0] == pytest.approx(runner.CLEANUP_SECONDS)


def test_reap_preserves_exit_status_and_propagates_unknown_state(monkeypatch, clock):
    replies = iter([(0, 0), (12345, 7 << 8)])
    monkeypatch.setattr(runner.os, 'waitpid', lambda *args: next(replies))
    assert runner._reap(12345) == 7
    def already_reaped(*args):
        raise ChildProcessError('already reaped')
    monkeypatch.setattr(runner.os, 'waitpid', already_reaped)
    with pytest.raises(ChildProcessError):
        runner._reap(12345)
    def denied(*args):
        raise PermissionError('not allowed')
    monkeypatch.setattr(runner.os, 'waitpid', denied)
    with pytest.raises(PermissionError):
        runner._reap(12345)


def test_terminate_preserves_already_known_status_without_reaping(monkeypatch):
    signals = []
    monkeypatch.setattr(runner.os, 'killpg', lambda *args: signals.append(args))
    monkeypatch.setattr(runner.os, 'waitpid', lambda *args: pytest.fail('known status must not be reaped again'))
    process = SimpleNamespace(pid=12345, returncode=7)
    runner._terminate(process)
    assert process.returncode == 7 and signals == [(12345, signal.SIGKILL)]


def test_terminate_does_not_acquire_a_stuck_popen_wait_lock():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    process._waitpid_lock.acquire()
    try:
        runner._terminate(process)
        assert process.returncode == -signal.SIGKILL
    finally:
        process._waitpid_lock.release()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_execute_cleanup_failure_overrides_success_and_restores_signals(tmp_path, monkeypatch):
    previous = {number: signal.getsignal(number) for number in runner._STOP_SIGNALS}
    original = runner._terminate
    def cleanup_failure(process):
        original(process)  # Actually reap the test child before injecting the cleanup failure.
        raise RuntimeError('termination is unconfirmed')
    monkeypatch.setattr(runner, '_terminate', cleanup_failure)
    stage = dict(argv=[sys.executable, '-c', 'pass'], max_seconds=5)
    with pytest.raises(RuntimeError, match='termination is unconfirmed'):
        runner._execute(stage, tmp_path / 'stage.log')
    assert {number: signal.getsignal(number) for number in previous} == previous


def test_analysis_cleanup_failure_closes_pipe_and_rejects_result(monkeypatch, clock):
    closed = []
    receiver = SimpleNamespace(poll=lambda seconds: True, recv=lambda: ('ok', {'stop': False}),
                               close=lambda: closed.append('receiver'))
    sender = SimpleNamespace(close=lambda: closed.append('sender'))
    monkeypatch.setattr(runner.multiprocessing, 'Pipe', lambda **kwargs: (receiver, sender))
    monkeypatch.setattr(runner.os, 'fork', lambda: 12345)
    monkeypatch.setattr(runner.os, 'kill', lambda *args: None)
    monkeypatch.setattr(runner.os, 'waitpid', lambda *args: (0, 0))
    with pytest.raises(RuntimeError, match='termination is unconfirmed'):
        runner._analyze({}, [], 1)
    assert closed == ['sender', 'receiver']


@pytest.mark.parametrize('reap_error', [None, ChildProcessError, RuntimeError])
def test_analysis_missing_child_still_requires_observed_exit(monkeypatch, reap_error):
    closed, reaped = [], []
    receiver = SimpleNamespace(poll=lambda seconds: True, recv=lambda: ('ok', {'stop': False}),
                               close=lambda: closed.append('receiver'))
    sender = SimpleNamespace(close=lambda: closed.append('sender'))
    monkeypatch.setattr(runner.multiprocessing, 'Pipe', lambda **kwargs: (receiver, sender))
    monkeypatch.setattr(runner.os, 'fork', lambda: 12345)
    def missing(pid, sig):
        assert (pid, sig) == (12345, signal.SIGKILL)
        raise ProcessLookupError('child no longer exists')
    def reap(pid):
        reaped.append(pid)
        if reap_error is not None:
            raise reap_error('termination not observed')
        return 0
    monkeypatch.setattr(runner.os, 'kill', missing)
    monkeypatch.setattr(runner, '_reap', reap)
    if reap_error is None:
        assert runner._analyze({}, [], 1) == {'stop': False}
    else:
        with pytest.raises(reap_error, match='termination not observed'):
            runner._analyze({}, [], 1)
    assert reaped == [12345]
    assert closed == ['sender', 'receiver']


def test_analysis_cleanup_failure_does_not_join_at_interpreter_shutdown(tmp_path):
    marker = tmp_path / 'child.pid'
    code = f'''
import os, signal, time
from pathlib import Path
from rewardhack import pilot, pilot_runner as r
r.CLEANUP_SECONDS = .05
real_fork = os.fork
def fork():
    pid = real_fork()
    if pid:
        Path({str(marker)!r}).write_text(str(pid))
    return pid
r.os.fork = fork
r.os.kill = lambda *args: None  # Inject a child that does not finish after the kill request.
r.signal.setitimer = lambda *args: None  # Keep it alive to expose any implicit shutdown join.
pilot.read_evidence = lambda *args: []
def decide(*args):
    os.close(1)
    os.close(2)
    time.sleep(60)
pilot.decide = decide
try:
    r._analyze({{}}, [], .1)
except RuntimeError as error:
    print(error, flush=True)
'''
    env = dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parents[1]))
    try:
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0 and 'termination is unconfirmed' in result.stdout
        os.kill(int(marker.read_text()), 0)  # The parent exited even while the simulated stuck child remained.
    finally:
        if marker.exists():
            os.kill(int(marker.read_text()), signal.SIGKILL)
