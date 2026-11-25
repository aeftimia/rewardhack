"""Durable JSON publication, strict record validation and shared filesystem locks for local state."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from pydantic import BaseModel, ConfigDict, ValidationError

from .json_objects import unique_object


class Record(BaseModel):
    """A record kind on disk or at a trust boundary: no coercion, no unknown fields, no NaN or infinity.

    Subclasses declare each field with its constraint; a `model_validator` holds the rules that join fields.
    """
    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)


def describe(error):
    """A pydantic error as one line that names each offending field (a rule joining fields names none)."""
    def one(problem):
        field = '.'.join(map(str, problem['loc']))
        return f"{field}: {problem['msg']}" if field else problem['msg']
    return '; '.join(map(one, error.errors()))


def validated(model, value, where):
    """`model` checked against `value`, or ValueError naming `where` and each offending field."""
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise ValueError(f'{where}: {describe(error)}') from None


def refuse_constant(value):
    raise ValueError(f'invalid JSON number {value}')


def load_json(text, **parse):
    """JSON whose objects have unique keys and whose numbers are finite; `parse` sets `parse_float`/`parse_int`."""
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=refuse_constant, **parse)


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
        sync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


@contextlib.contextmanager
def file_lock(path, wait=0.0):
    """Exclusive lock, refused when held. `wait` is how long to keep trying, for locks held only briefly.

    A lock that guards a whole run (one pilot per output folder) is refused at once. A lock around one
    short read-modify-write (the reservation registry) is waited for: a second claimant being told the
    budget refused it, with allowance left, would end its stage for nothing.
    A forked child keeps the lock with its inherited descriptor (the pilot's analysis child relies on it).
    `filelock.FileLock` was tried and is not used: it closes inherited lock descriptors in a forked child.
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
