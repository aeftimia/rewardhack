"""Existing grant guards must see the pilot's shared worst-cost commitments."""
import json
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
