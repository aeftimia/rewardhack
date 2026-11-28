"""A startup delay and measured throughput drive the transfer deadline decision."""
import sys

import pytest

from entrypoints import watch_transfer
from rewardhack.time_projection import ProgressProjection


def test_projection_waits_for_progress_then_separates_feasible_and_slow_transfers():
    fast = ProgressProjection(2100)
    slow = ProgressProjection(2100)
    for index in range(6):
        elapsed = 20 * index
        assert fast.update(max(0, elapsed-20)*2, elapsed) is None
        assert slow.update(max(0, elapsed-20), elapsed) is None
    assert fast.update(200, 120) < 1500
    assert slow.update(100, 120) > 1500


def test_projection_rejects_progress_reversal():
    projection = ProgressProjection(100)
    projection.update(10, 1)
    with pytest.raises(ValueError, match='nondecreasing'):
        projection.update(9, 2)


def test_transfer_watch_stops_a_projected_overrun(tmp_path, monkeypatch):
    source = tmp_path / 'checkpoint'
    source.write_bytes(b'x' * 1000)
    monkeypatch.setattr(watch_transfer, 'SAMPLE_SECONDS', .02)
    monkeypatch.setattr(watch_transfer, 'FINISH_MARGIN_SECONDS', 0)
    program = "import time\nfor n in range(1, 11):\n print(f'{n*100} {n*10}% 1MB/s 00:00:01', end='\\r', flush=True)\n time.sleep(.1)"
    with pytest.raises(TimeoutError, match='projected past'):
        watch_transfer.watch(source, .5, [sys.executable, '-u', '-c', program])
