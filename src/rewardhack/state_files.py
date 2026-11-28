"""Durable JSON publication and shared filesystem locks for local state."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time


def pod_receipt(result):
    return result.with_suffix('.pod.json')


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value, *, exclusive=True):
    """Publish a complete durable JSON record; existing receipts cannot be replaced."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.state-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
            temporary = None
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            os.unlink(temporary)


@contextlib.contextmanager
def file_lock(path, wait=0.0):
    """Exclusive lock, refused when held. `wait` is how long to keep trying, for locks held only briefly.

    A lock that guards a whole run (one pilot per output folder) is refused at once. A lock around one
    short read-modify-write (the reservation registry) is waited for: a second claimant being told the
    budget refused it, with allowance left, would end its stage for nothing.
    """
    deadline = time.monotonic()+wait
    with Path(path).open('a') as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError(f'another process holds {path}') from None
                time.sleep(.02)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


