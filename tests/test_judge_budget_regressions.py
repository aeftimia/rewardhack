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
    monkeypatch.setattr(model_io, 'call', lambda *args, **kw: dict(raw=raw, finish='stop', model='stub', usage={'completion_tokens': 10}))
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
    good = json.dumps(dict(requested='a', reported='b', **dict.fromkeys(judge.CHECKS, False)))
    replies = iter(['', '', good, '', '', ''])
    monkeypatch.setattr(model_io, 'call', lambda *args, **kw: dict(raw=next(replies), finish='stop', model='stub', usage={}))
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


GOOD = dict(requested='a', reported='b', **dict.fromkeys(judge.CHECKS, False))  # what each reply may and must hold: tests/test_judge_parse.py


def test_refund_rows_lower_spend_only_when_marked():
    charge = dict(line='api', total_usd='10', status='paid')
    assert budget.spent([charge, dict(line='api', total_usd='-3.5', status='refund')], line='api') == 6.5
    with pytest.raises(ValueError, match='ledger row 2 total_usd'):
        budget.spent([charge, dict(line='api', total_usd='-3.5', status='paid')], line='api')
    for bad in ('3.5', 'nan', '-inf', 'bad', None):  # a refund is never a charge and never unbounded
        with pytest.raises(ValueError, match='ledger row 2 total_usd'):
            budget.spent([charge, dict(line='api', total_usd=bad, status='refund')], line='api')
    with pytest.raises(ValueError, match="refunds exceed charges on line 'api' by \\$1.00"):  # never below zero
        budget.spent([dict(line='api', total_usd='-1', status='refund')], line='api')


@pytest.mark.parametrize('kind', ['local', 'api'])
def test_backend_failure_streak_and_recovery(monkeypatch, kind):
    good = json.dumps(GOOD)
    replies = iter(['', '', good, '', '', ''])
    monkeypatch.setattr(model_io, 'local_reader', lambda *a: lambda prompt: next(replies))
    monkeypatch.setattr(model_io, 'call', lambda *a, **kw: dict(raw=next(replies), finish='stop', model='stub', usage={}))
    reader = judge.make_judge(kind, model='stub')
    row = dict(statement='True', report='Proved it.')
    assert [reader(row)['claim'] for _ in range(5)] == ['UNPARSED', 'UNPARSED', 'OK', 'UNPARSED', 'UNPARSED']
    with pytest.raises(RuntimeError, match='3 times in a row'):
        reader(row)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf'), -1])
def test_invalid_api_costs_rejected_before_calls(monkeypatch, value):
    monkeypatch.setattr(model_io, 'call', lambda *a, **kw: pytest.fail('invalid configuration made a request'))
    for field in ('usd_per_m_input', 'usd_per_m_output', 'max_usd'):
        with pytest.raises(ValueError, match='finite, nonnegative'):
            judge.make_judge('api', model='stub', **{field: value})
    with pytest.raises(ValueError, match='validation spend cap'):
        judge.validate(None, [], max_usd=value)
    from rewardhack.openai_probe import run
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='API spend cap'):
        run(SimpleNamespace(max_usd=value))


def test_zero_judge_cap_prevents_request(monkeypatch):
    monkeypatch.setattr(model_io, 'call', lambda *a, **kw: pytest.fail('zero cap made a request'))
    reader = judge.make_judge('api', model='stub', max_usd=0)
    with pytest.raises(RuntimeError, match='spend cap'):
        reader(dict(statement='True', report='Proved it.'))


@pytest.mark.parametrize('line', ['api ', ' api', 'API', 'typo', ' '])
def test_unknown_ledger_lines_cannot_escape_caps(line):
    with pytest.raises(ValueError, match='unknown budget line'):
        budget.check([dict(line=line, total_usd='300')], 'api', 0, 0, other=300)
    assert budget.grant_spent([dict(line='', total_usd='300')]) == 0


def test_zero_probe_cap_preserves_plan_without_requests(tmp_path, monkeypatch):
    from rewardhack import openai_probe
    from types import SimpleNamespace
    monkeypatch.setattr(openai_probe, 'select_tasks', lambda *a, **kw: [dict(id='t')])
    monkeypatch.setattr(openai_probe, 'taskset_rows', lambda *a: [])
    monkeypatch.setattr(openai_probe, 'call', lambda *a, **kw: pytest.fail('zero cap made a request'))
    args = SimpleNamespace(max_usd=0, usd_per_m_input=1, usd_per_m_output=1, out=tmp_path,
                           taskset='stub', tasks=None, limit=None, samples=1, concurrency=1,
                           model='stub', variant='neutral', reasoning_effort=None, max_tokens=4, report_mode='prose', follow_on=False)
    openai_probe.run(args)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['requests'] == 0 and summary['planned'] == 1 and summary['cost_cap_hit']


def test_refunds_in_a_free_text_ledger():
    # Status is free text in the real ledger, and decimal dollars do not add exactly in binary.
    rows = [dict(line='api', total_usd='0.3', status='completed: metered'),
            dict(line='api', total_usd='-0.1', status='Refund: per console'),
            dict(line='api', total_usd='-0.2', status='refund')]
    assert budget.spent(rows, line='api') == 0 and str(budget.spent(rows, line='api')) == '0.0'
    assert budget.check(rows, 'api', 1, 1) == (1, [])
    with pytest.raises(ValueError, match='status starts with "refund"'):  # the refusal says how to book a refund
        budget.spent([dict(line='api', total_usd='-0.06', status='refunded?')])
    with pytest.raises(ValueError, match='refunds exceed charges'):  # a refund on another line than its charge
        budget.spent([dict(line='', total_usd='5', status='paid'), dict(line='api', total_usd='-5', status='refund')], line='api')


EXAMPLE = json.dumps(GOOD)
FLAGGED = json.dumps(dict(GOOD, **{next(iter(judge.CHECKS)): True}))


@pytest.mark.parametrize('reply', [
    '<think>e.g. ' + EXAMPLE + '</think> ' + EXAMPLE.replace('false', '"false"', 1),   # the answer itself is not boolean
    '<think>e.g. ' + EXAMPLE + '</think> ' + json.dumps({next(iter(judge.CHECKS)): True}),  # the answer is incomplete
    '{"requested": "x", "draft": ' + FLAGGED + ', "claims',                              # a draft inside a cut-off answer
    '{"requested": "a", "reported": "it wrote ' + EXAMPLE.replace('"', "'") + ' then stopped',
    '{"requested": "a", "reported": "it wrote ' + EXAMPLE + ' then stopped',
    EXAMPLE + FLAGGED,
    'I cannot answer in JSON. The set {} is odd.',
    '{answer: ' + EXAMPLE,                                                               # an unquoted, cut-off wrapper
    '{answer: ' + FLAGGED + '}',                                                         # a broken wrapper, closed
    # A stray closing brace inside a note of the broken answer must not end it early and leave the draft standing.
    EXAMPLE + ' Final: {"requested": "x } y", "reported": "b", "says_false": true, "not_pro',
    EXAMPLE + ' Final: {"requested": "the set } of x", "reported": "b", ' + FLAGGED[1:].replace('true', 'True'),
    'Draft: ' + EXAMPLE + ' Final: {"requested": "x", "reported": "says "done}" but admits sorry", ' + FLAGGED[1:],
    '{"requested": "a", "reported": "it wrote } ' + EXAMPLE + ' then stopped',
    EXAMPLE + ' Final: {"requested":"a","reported":"b"," says_false": true,}',
    EXAMPLE + ' {',
])
def test_a_checklist_beside_a_broken_answer_is_not_a_reading(reply):
    assert judge.claim_of(reply) == 'UNPARSED'
    assert judge.compact(reply) == reply.strip()[:200]  # what is stored is what was said, never one object picked out


def test_an_unreadable_reply_records_no_kinds(monkeypatch):
    # The first 200 characters of an unreadable reply can be a clean checklist whose contradiction came later.
    reply = FLAGGED + ' '*200 + EXAMPLE
    assert len(FLAGGED) < 200 and judge.claim_of(reply) == 'UNPARSED' and judge.claim_of(judge.compact(reply)) == 'ERROR'
    monkeypatch.setattr(model_io, 'call', lambda *args, **kw: dict(raw=reply, finish='stop', model='stub', usage={}))
    verdict = judge.make_judge('api', model='stub')(dict(statement='True', report='Proved it.'))
    assert verdict['claim'] == 'UNPARSED' and judge.kinds_for(verdict) == []
    from rewardhack import outcomes
    for name in ('evaluate_submission', 'verify_proof', 'verify_refutation'):
        monkeypatch.setattr(outcomes, name, lambda *a: dict(established='True', proof_body='', report='Proved it.', reward=0.0,
                                                            compiles=False, diagnostic='', ok=False, axioms=[], why='no'))
    task = dict(id='t', group='g', tier='easy', true=True, statement='True')
    row = outcomes.score('unused', task, json.dumps(dict(established='True', proof_body='x', report='Proved it.')), 'baseline', judge=lambda r: verdict)
    assert row['claim'] == 'UNPARSED' and row['error_kinds'] == [] and row['judge_raw'] == verdict['raw']


@pytest.mark.parametrize('flag,value', [('--max-usd', 'nan'), ('--judge-max-usd', '-1'), ('--judge-usd-in', 'inf')])
def test_validation_refuses_bad_caps_before_creating_anything(tmp_path, flag, value):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'entrypoints/validate_judge.py'), '--judge', 'api', '--judge-model', 'stub',
                             '--out', str(tmp_path/'out'), flag, value], cwd=root, capture_output=True, text=True)
    assert result.returncode == 2 and 'finite, nonnegative' in result.stderr and 'Traceback' not in result.stderr
    assert not (tmp_path/'out').exists()


def test_a_refund_of_the_metered_amount_is_not_an_excess():
    rows = [dict(line='api', total_usd='0.07', status='completed: $0.0715 metered'), dict(line='api', total_usd='-0.0715', status='refund')]
    assert str(budget.spent(rows, line='api')) == '0.0'


def test_status_shows_a_refund_booked_to_no_line(tmp_path):
    root = Path(__file__).resolve().parents[1]
    def status(rows):
        (tmp_path/'ledger.csv').write_text(','.join(budget.FIELDS)+'\n'+''.join(
            ','.join(str(dict(total_usd=total, hours=0, hourly_usd=0, other_usd=max(total, 0), status=state, line=line).get(field, '')) for field in budget.FIELDS)+'\n' for total, state, line in rows))
        return subprocess.run([sys.executable, str(root/'entrypoints/budget.py'), '--ledger', str(tmp_path/'ledger.csv'), 'status'],
                              cwd=root, capture_output=True, text=True).stdout
    assert 'other funds): $-3.00' in status([(5, 'completed', 'api'), (-3, 'refund: console', '')])
    assert 'other funds): $0.00' in status([(0.07, 'completed', 'api'), (-0.074, 'refund', 'api'), (1, 'completed', 'storage')])


@pytest.mark.parametrize('kind', ['local', 'api'])
@pytest.mark.parametrize('first,last', [(True, False), (False, True), (False, False)])
def test_duplicate_check_keys_are_unreadable_and_stop_judging(monkeypatch, kind, first, last):
    raw = json.dumps(dict(GOOD, says_false=first))[:-1] + ', "says_false": ' + json.dumps(last) + '}'
    monkeypatch.setattr(model_io, 'local_reader', lambda *a: lambda prompt: raw)
    monkeypatch.setattr(model_io, 'call', lambda *a, **kw: dict(raw=raw, finish='stop', model='stub', usage={}))
    assert judge.claim_of(raw) == 'UNPARSED'
    assert judge.compact(raw) == raw[:200]
    reader = judge.make_judge(kind, model='stub')
    row = dict(statement='True', report='Proved it.')
    for _ in range(judge.MAX_STREAK - 1):
        verdict = reader(row)
        assert verdict['claim'] == 'UNPARSED'
        assert judge.kinds_for(verdict) == []
    with pytest.raises(RuntimeError, match='3 times in a row'):
        reader(row)


def test_a_malformed_ledger_row_is_refused(tmp_path):
    """A row one comma short loses its budget line and would count against nothing; a total its own parts contradict is a typo."""
    header = ','.join(budget.FIELDS)+'\n'
    good = '2026-11-25,runpod,m,p,r,1,2.70,0,2.70,completed,e,gpt-oss-20b-gpu\n'
    ledger = tmp_path/'ledger.csv'
    ledger.write_text(header+good)
    assert budget.spent(budget.read_ledger(ledger, include_reservations=False), line='gpt-oss-20b-gpu') == 2.70
    for bad, message in ((good.rsplit(',', 1)[0]+'\n', 'fields'), (good.replace(',2.70,completed', ',0.27,completed'), 'total_usd is'),
                         (good.rstrip('\n')+',extra\n', 'fields')):
        ledger.write_text(header+bad)
        with pytest.raises(ValueError, match=message):
            budget.read_ledger(ledger, include_reservations=False)
