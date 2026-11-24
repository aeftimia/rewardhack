"""Failures must stop judging and invalid numbers must never bypass spending caps."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from rewardhack import budget, judge, model_io


@pytest.mark.parametrize('raw', ['', 'I cannot comply.', '{"not_proved": true}',
                                 json.dumps(dict.fromkeys(judge.CHECKS, 'false')),
                                 json.dumps(dict.fromkeys(judge.CHECKS, False))[:-1]])
def test_invalid_checklists_stop_after_consecutive_failures(monkeypatch, raw):
    monkeypatch.setattr(model_io, 'call', lambda *args: dict(raw=raw, finish='stop', usage={'completion_tokens': 10}))
    reader = judge.make_judge('api', model='stub', usd_per_m_output=1)
    row = dict(statement='True', report='Proved it.')
    for _ in range(judge.MAX_STREAK - 1):
        assert reader(row)['claim'] == 'UNPARSED'
    with pytest.raises(RuntimeError, match='valid complete checklist'):
        reader(row)
    # The judge answered each time: unreadable replies are counted apart from calls that failed.
    assert reader.meter.totals['unreadable'] == judge.MAX_STREAK and reader.meter.totals['errors'] == 0
    assert reader.meter.totals['output'] == judge.MAX_STREAK * 10


def test_valid_checklist_resets_failure_streak(monkeypatch):
    good = json.dumps(dict.fromkeys(judge.CHECKS, False))
    replies = iter(['', '', good, '', '', ''])
    monkeypatch.setattr(model_io, 'call', lambda *args: dict(raw=next(replies), finish='stop', usage={}))
    reader = judge.make_judge('api', model='stub')
    row = dict(statement='True', report='Proved it.')
    assert [reader(row)['claim'] for _ in range(5)] == ['UNPARSED', 'UNPARSED', 'OK', 'UNPARSED', 'UNPARSED']
    with pytest.raises(RuntimeError, match='3 times in a row'):
        reader(row)


@pytest.mark.parametrize('field', ['hourly', 'hours', 'other', 'launch_cap'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf'), -1])
def test_invalid_launch_numbers_are_rejected(field, value):
    args = dict(line='api', hourly=1, hours=1, other=0, launch_cap=10)
    args[field] = value
    with pytest.raises(ValueError, match='finite, nonnegative'):
        budget.check([], **args)


@pytest.mark.parametrize('value', ['nan', 'inf', '-1', 'bad', None])
def test_invalid_grant_ledger_costs_are_rejected(value):
    # A bad charge on another grant line must not weaken the overall cap either.
    with pytest.raises(ValueError, match='ledger row 1 total_usd'):
        budget.check([dict(line='storage', total_usd=value)], 'api', 1, 1)


def test_overflow_is_rejected_and_zero_is_allowed():
    with pytest.raises(ValueError, match='worst-case cost'):
        budget.check([], 'api', 1e308, 1e308)
    assert budget.check([], 'api', 0, 0, launch_cap=0) == (0, [])
    with pytest.raises(ValueError, match='ledger total'):
        budget.spent([dict(total_usd=1e308), dict(total_usd=1e308)])


def test_budget_cli_refuses_nan_without_traceback():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'entrypoints/budget.py'), 'check',
                             '--line', 'api', '--hourly', 'nan', '--max-hours', '2', '--launch-cap', '1'],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'REFUSED: hourly price must be a finite, nonnegative number' in result.stderr
    assert 'Traceback' not in result.stderr


GOOD = dict.fromkeys(judge.CHECKS, False)


@pytest.mark.parametrize('reply,claim', [
    ('<think>Check {each} box.</think>' + json.dumps(GOOD), 'OK'),
    (json.dumps(GOOD) + ' (see {note})', 'OK'),
    ('Reasoning {step 1}: ' + json.dumps(dict(GOOD, requested='a {braced} lemma')) + ' Done.', 'OK'),
    (json.dumps(dict(GOOD, **{next(iter(judge.CHECKS)): True})) + ' {}', 'ERROR'),
    # Two complete checklists that disagree are not a reading; nor is a Python-style literal.
    (json.dumps(GOOD) + json.dumps(dict.fromkeys(judge.CHECKS, True)), 'UNPARSED'),
    (json.dumps(GOOD).replace('false', 'False'), 'UNPARSED'),
    (None, 'UNPARSED'),
])
def test_checklist_is_found_among_other_braces(reply, claim):
    assert judge.claim_of(reply) == claim
    assert isinstance(judge.compact(reply), str)


def test_refund_rows_lower_spend_only_when_marked():
    charge = dict(line='api', total_usd='10', status='paid')
    assert budget.spent([charge, dict(line='api', total_usd='-3.5', status='refund')], line='api') == 6.5
    with pytest.raises(ValueError, match='ledger row 2 total_usd'):
        budget.spent([charge, dict(line='api', total_usd='-3.5', status='paid')], line='api')
    for bad in ('3.5', 'nan', '-inf', 'bad', None):  # a refund is never a charge and never unbounded
        with pytest.raises(ValueError, match='ledger row 2 total_usd'):
            budget.spent([charge, dict(line='api', total_usd=bad, status='refund')], line='api')
    with pytest.raises(ValueError, match='ledger total'):  # refunds cannot take a line below zero
        budget.spent([dict(line='api', total_usd='-1', status='refund')], line='api')
