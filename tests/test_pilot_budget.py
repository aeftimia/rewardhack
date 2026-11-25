"""Existing grant guards must see the pilot's shared worst-cost commitments."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rewardhack import budget, pilot_runner


def registry(tmp_path, records):
    ledger = tmp_path/'ledger.csv'
    ledger.write_text('line,total_usd,status\napi,2,completed\n')
    budget.reservation_path(ledger).write_text(json.dumps(records))
    return ledger


def reservation(key='one', line='storage', total=100):
    return dict(key=key, line=line, total_usd=total, status='pilot worst-case reservation')


def test_legacy_budget_check_refuses_a_fully_reserved_line(tmp_path):
    ledger = registry(tmp_path, [reservation()])
    _, problems = budget.check(budget.read_ledger(ledger), 'storage', 0, 0, other=.01)
    assert any('storage' in problem for problem in problems)
    script = Path(__file__).resolve().parents[1]/'entrypoints'/'budget.py'
    result = subprocess.run([sys.executable, str(script), '--ledger', str(ledger), 'check',
                             '--line', 'storage', '--hourly', '1', '--max-hours', '1'],
                            capture_output=True, text=True)
    assert result.returncode == 1 and 'REFUSED:' in result.stdout


def test_launch_that_exactly_fills_a_cap_is_not_refused_by_binary_rounding(tmp_path):
    ledger = tmp_path/'ledger.csv'
    ledger.write_text('line,total_usd,status\nstorage,35.2,completed\n')
    assert 35.2+2.7*24 > 100  # the unrounded sum
    assert budget.check(budget.read_ledger(ledger), 'storage', 2.7, 24) == (64.8, [])
    assert budget.check(budget.read_ledger(ledger), 'storage', 2.7, 24, other=.01)[1]
    assert budget.check([], 'api', 1.1, 3, launch_cap=3.3) == (3.3, [])
    for cap in (1.15*100, .57*100, .1234567):  # a cap that rounds up must still admit a launch equal to it
        assert budget.check([], 'api', 0, 0, cap, cap)[1] == []
    assert budget.check([], 'api', 0, 0, 3.31, 3.3)[1]
    # Inside a pilot stage: a launch equal to the stage's own reservation, on a line it fills exactly.
    budget.reservation_path(ledger).write_text(json.dumps([reservation(key='k', total=64.8)]))
    script = Path(__file__).resolve().parents[1]/'entrypoints'/'budget.py'
    def guard(hours, key='k'):
        return subprocess.run([sys.executable, str(script), '--ledger', str(ledger), 'claim', '--line', 'storage',
                               '--hourly', '2.7', '--max-hours', hours],
                              env=dict(os.environ, **{budget.RESERVATION_ENV: key}), capture_output=True, text=True).returncode
    # The refused claim comes first: it must not consume the allowance the exact one then needs.
    assert (guard('24.01'), guard('24'), guard('24', key='other'), guard('0.01')) == (1, 0, 1, 1)


def test_reading_reservations_never_changes_measured_csv(tmp_path):
    ledger = registry(tmp_path, [reservation(total=30)])
    original = ledger.read_bytes()
    rows = budget.read_ledger(ledger)
    assert budget.grant_spent(rows) == 32
    assert budget.grant_spent(budget.read_ledger(ledger, include_reservations=False)) == 2
    assert ledger.read_bytes() == original


def test_runner_counts_shared_reservations_once(tmp_path):
    ledger = registry(tmp_path, [])
    spec = dict(budget=dict(ledger=str(ledger), line='storage'))
    stage = dict(id='one', max_usd=40)
    first, problems = pilot_runner._reserve(spec, tmp_path/'one', stage, 'hash-one')
    assert first and not problems
    second, problems = pilot_runner._reserve(spec, tmp_path/'two', stage, 'hash-two')
    assert second and not problems
    third, problems = pilot_runner._reserve(spec, tmp_path/'three', stage, 'hash-three')
    assert third is None and problems
    assert budget.spent(budget.read_ledger(ledger), line='storage') == 80


def test_symlink_ledger_uses_the_canonical_shared_registry(tmp_path):
    ledger = registry(tmp_path, [reservation()])
    alias = tmp_path/'alias.csv'; alias.symlink_to(ledger)
    assert budget.read_ledger(alias) == budget.read_ledger(ledger)
    assert budget.reservation_path(alias) == budget.reservation_path(ledger)


@pytest.mark.parametrize('record', [reservation(line=None), reservation(line=''), reservation(total=-1),
                                    reservation(total=float('nan')), reservation(total=True)])
def test_corrupt_registry_blocks_legacy_budget_reading(tmp_path, record):
    ledger = registry(tmp_path, [record])
    with pytest.raises(ValueError):
        budget.read_ledger(ledger)


def test_missing_csv_does_not_hide_an_existing_reservation(tmp_path):
    ledger = registry(tmp_path, [reservation()])
    ledger.unlink()
    assert budget.spent(budget.read_ledger(ledger), line='storage') == 100


def test_reservations_cannot_mask_refunds_exceeding_actual_charges(tmp_path):
    ledger = registry(tmp_path, [reservation(line='api', total=100)])
    ledger.write_text('line,total_usd,status\napi,10,completed\napi,-20,refund excess\n')
    with pytest.raises(ValueError, match='refunds exceed charges'):
        budget.read_ledger(ledger)
    with pytest.raises(ValueError, match='refunds exceed charges'):
        budget.read_ledger(ledger, include_reservations=False)
