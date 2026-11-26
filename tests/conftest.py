"""Shared test helpers."""
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def driver_script(tmp_path):
    """The pod driver in a copy of `ops/` whose launch check always passes, for tests of everything after that check.

    The driver runs `ops/gate.sh launch` from its own folder, so the copy carries a stub there and the real driver
    keeps no exemption for tests. The launch check itself is tested against the real gate in test_gates.py.
    """
    ops = tmp_path/'driver'/'ops'
    if not ops.exists():
        shutil.copytree(PROJECT/'ops', ops)
        (ops/'gate.sh').write_text('#!/usr/bin/env bash\nexit 0\n')
    return str(ops/'drive_runpod.sh')


import fcntl
import os
import time

import pytest

# Parallel suites took down a 16 GB laptop on 26 Nov: `-n auto` gave each run one worker per core (12), and several
# agents started full runs at once (three from one session, more from others), about 36 workers at up to 0.85 GB each.
# So `-n auto` is sized by memory here, and a suite started while another is running on this machine waits for it.
WORKER_GB = 1.0  # measured peak 0.85 GB per worker (4 workers: 3.3 GB in all, 26 Nov), rounded up
MEMORY_SHARE = 0.5  # of installed memory, so the rest of the machine keeps working while the suite runs
SUITE_LOCK = Path.home()/'.cache'/'rewardhack-pytest.lock'  # one per user, shared by every checkout and worktree
SUITE_LOCK_WAIT_SECONDS = 1200  # cap: 2x measured 145 s per suite, for up to three queued runs ahead plus slack


def pytest_xdist_auto_num_workers(config):
    """The worker count `-n auto` uses: one per core, but no more than MEMORY_SHARE of installed memory holds."""
    memory_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
    return max(1, min(os.cpu_count(), int(memory_gb * MEMORY_SHARE / WORKER_GB)))


def pytest_configure(config):
    """Run one suite at a time on this machine: the controller (not an xdist worker) holds SUITE_LOCK until it exits."""
    if hasattr(config, 'workerinput'):
        return
    SUITE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = SUITE_LOCK.open('a+')
    deadline = time.monotonic() + SUITE_LOCK_WAIT_SECONDS
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() > deadline:
                handle.seek(0)
                raise pytest.UsageError(f'another test suite has held {SUITE_LOCK} for over {SUITE_LOCK_WAIT_SECONDS} s '
                                        f'(holder: {handle.read().strip()}); stop it or wait for it, then run again') from None
            time.sleep(1)
    handle.seek(0)
    handle.truncate()
    handle.write(f'pid {os.getpid()} in {os.getcwd()}\n')
    handle.flush()
    config._suite_lock = handle  # kept open, and so held, until the run ends; the lock goes with the process


@pytest.fixture(autouse=True)
def priced_test_models(monkeypatch):
    """The stand-in model names tests call metered endpoints with get rows in model_io.PRICES, as a real model would
    (prices are stated once there). A test that depends on a price sets its own row."""
    from rewardhack import model_io
    monkeypatch.setitem(model_io.PRICES, 'stub', (0.0, 0.0))
    monkeypatch.setitem(model_io.PRICES, 'm', (0.0, 0.0))
