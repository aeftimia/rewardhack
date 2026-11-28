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
        # The driver records the commit it launches (launch-commit.txt), so the copy is a repository of its own.
        import subprocess
        identity = ['-c', 'user.name=test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null']
        for command in (['init', '-q'], ['add', '.'], ['commit', '-q', '-m', 'driver copy']):
            subprocess.run(['git', *identity, *command], cwd=ops.parent, check=True, capture_output=True)
    return str(ops/'drive_runpod.sh')


import fcntl
import os
import time
import uuid

import pytest

# Parallel suites took down a 16 GB laptop on 26 Nov: `-n auto` gave each run one worker per core (12), and several
# agents started full runs at once (three from one session, more from others), about 36 workers at up to 0.85 GB each.
# So `-n auto` is sized by memory here, and a suite started while another is running on this machine waits for it.
WORKER_GB = 1.0  # measured peak 0.85 GB per worker (4 workers: 3.3 GB in all, 26 Nov), rounded up
MEMORY_SHARE = 0.5  # of installed memory, so the rest of the machine keeps working while the suite runs
SUITE_LOCK = Path.home()/'.cache'/'rewardhack-pytest.lock'  # one per user, shared by every checkout and worktree
SUITE_LOCK_WAIT_SECONDS = 1200  # cap: 2x measured 145 s per suite, for up to three queued runs ahead plus slack


def pytest_addoption(parser):
    parser.addoption('--repeat-each', type=int, default=1, metavar='N',
                     help='run every selected test N times, as separate items the workers share: a test that fails only '
                          'sometimes, or only under load, shows how often in one suite run instead of several runs at once')
    parser.addoption('--repeat-match', action='append', default=[], metavar='TEXT',
                     help='with --repeat-each, repeat only the tests whose id contains TEXT (may be given several times); '
                          'the rest run once, so one run is the whole suite plus the repeats, all on the same workers')


def pytest_generate_tests(metafunc):
    """--repeat-each: each test gains a `repeat_index` parameter (pytest drops repeated node ids, so naming a test N times
    runs it once). Used to find and confirm fixes for timing flakes (26 Nov, the removal-timer test). pytest-repeat does
    the same (`--count`) but had no commit since April 2025, so these few lines stand in for it (CODE.md: maintained libraries)."""
    count = metafunc.config.getoption('repeat_each')
    match = metafunc.config.getoption('repeat_match')
    if count > 1 and (not match or any(text in metafunc.definition.nodeid for text in match)):
        metafunc.fixturenames.append('repeat_index')
        metafunc.parametrize('repeat_index', range(count))


def pytest_collection_modifyitems(config, items):
    """--repeat-match that repeats nothing is an error, not a run of single tests reported as repeated (review round 35)."""
    match = config.getoption('repeat_match')
    if config.getoption('repeat_each') < 1:
        raise pytest.UsageError('--repeat-each must be at least 1')
    if match and config.getoption('repeat_each') < 2:
        raise pytest.UsageError('--repeat-match needs --repeat-each of 2 or more')
    unmatched = [text for text in match if not any(text in item.nodeid for item in items)]
    if unmatched:
        raise pytest.UsageError(f"--repeat-match selects no test: {', '.join(unmatched)}")


# Echo the exact receipt requested by the driver, including when SSH printed a login banner.
LAUNCH_RECEIPT_ANSWER = r'''for arg in "$@"; do
  if [[ $arg == *'cat out/LAUNCH_ID'* && $arg == *'then echo'* ]]; then
    token=${arg#*then echo \'}
    printf '%s\n' "${token%%\'*}"
  fi
done
'''


# GNU timeout's stand-in where tests stub it: skip its options and its duration, run the rest. The runner calls
# `timeout --signal=TERM --kill-after=N secs cmd`, the driver `timeout secs cmd`; a fixed `shift 3` mangled the second.
TIMEOUT_STUB = 'while [[ $1 == -* ]]; do shift; done\nshift\nexec "$@"\n'

# A real session boundary on macOS too; all runner tests exercise session ownership.
SETSID = '''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
if args[0] == '--wait':
    args.pop(0)
os.setsid()
os.execvp(args[0], args)
'''


@pytest.fixture(autouse=True)
def session_process_tools(tmp_path_factory, monkeypatch):
    """BSD pgrep/pkill lack Linux's -s; query real OS session IDs for runner tests on macOS."""
    import sys
    if sys.platform != 'darwin':
        return
    directory = tmp_path_factory.mktemp('session-tools')
    script = '''#!/usr/bin/env python3
import os, pathlib, signal, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if '-s' not in args:
    os.execv('/usr/bin/' + name, [name, *args])
session = int(args[args.index('-s') + 1])
pids = subprocess.run(['/bin/ps', '-axo', 'pid=,stat='], check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
states = set(args[args.index('-r') + 1].split(',')) if '-r' in args else None
members = []
for text in pids:
    number, state = text.split()
    pid = int(number)
    if pid == os.getpid():
        continue
    if states is not None and state[0] not in states:
        continue
    try:
        if os.getsid(pid) == session:
            members.append(pid)
    except ProcessLookupError:
        # A process can exit between the process listing and getsid; it is no longer a session member.
        continue
if name == 'pgrep':
    for pid in members:
        print(pid)
else:
    signum = int(args[0][1:]) if args[0].startswith('-') and args[0][1:].isdigit() else signal.SIGTERM
    for pid in members:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            # The requested postcondition already holds if the process exited before its signal.
            continue
sys.exit(0 if members else 1)
'''
    for name in ('pgrep', 'pkill'):
        path = directory/name
        path.write_text(script)
        path.chmod(0o755)
    monkeypatch.setenv('PATH', f"{directory}:{os.environ['PATH']}")


@pytest.fixture
def pod_id():
    """A fake pod id no other test or repeat shares: the runner stops every timer of its pod id, by design. Ids made from
    `tmp_path.name` collided, since pytest cuts that name to 30 characters and every xdist worker numbers from 0."""
    return lambda label: f'{label}-{uuid.uuid4().hex}'


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


# What a row's provenance needs (rewardhack.provenance.meta), for tests that build run options by hand: given once here,
# so a test states only the options it is about. A test's own value wins.
PROVENANCE = dict(model='test/model', revision='test-revision', taskset='hard', tasks='all', part='train', prompt_variant='neutral', variant='neutral',
                  reasoning_effort=None, thinking='unsupported', extra_body={}, training_sampler='uniform', backward_batch_size=None,
                  profile_backward=False, grader_runs=0)


def run_options(**options):
    from types import SimpleNamespace
    return SimpleNamespace(**(PROVENANCE | options))
