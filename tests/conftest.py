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


import pytest


@pytest.fixture(autouse=True)
def priced_test_models(monkeypatch):
    """The stand-in model names tests call metered endpoints with get rows in model_io.PRICES, as a real model would
    (prices are stated once there). A test that depends on a price sets its own row."""
    from rewardhack import model_io
    monkeypatch.setitem(model_io.PRICES, 'stub', (0.0, 0.0))
    monkeypatch.setitem(model_io.PRICES, 'm', (0.0, 0.0))
