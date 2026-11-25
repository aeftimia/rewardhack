"""Every rule of the strict record models, one case each, and exact dollars in the budget.

Each refusal must name the offending field (or the rule, for a rule that joins fields).
"""
import json
from decimal import Decimal
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rewardhack import budget, pilot, pilot_model, pilot_runner as runner
from rewardhack.pilot_simulation import example_spec, synthetic_rows
from rewardhack.state_files import file_lock

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spec(tmp_path):
    return pilot.load_spec(example_spec(tmp_path/'demo'))


def stage0(s):
    return s['stages'][0]


SPEC_RULES = [  # (what changes, a word the refusal must contain)
    (lambda s: s.update(extra=1), 'extra'),
    (lambda s: s.pop('resource'), 'resource'),
    (lambda s: s.update(schema='compute-pilot-v0'), 'schema'),
    (lambda s: s.update(id='   '), 'id'),
    (lambda s: s.update(question=3), 'question'),
    (lambda s: s.update(question='bad \ud800'), 'surrogate'),
    (lambda s: s['context'].update(report_mode='poetry'), 'report_mode'),
    (lambda s: s['context'].pop('baseline'), 'baseline'),
    (lambda s: s.update(seeds=['a', 'b']), 'seeds'),
    (lambda s: s.update(seeds=['a', 'a', 'b']), 'seeds'),
    (lambda s: s.update(seeds=['a', 'b', '']), 'seeds'),
    (lambda s: s.update(seeds=('a', 'b', 'c')), 'seeds'),  # a list, not a tuple, as JSON gives
    (lambda s: s.update(checkpoints=[1, 2]), 'checkpoints'),
    (lambda s: s.update(checkpoints=list(range(1, 34))), 'checkpoints'),
    (lambda s: s.update(checkpoints=[1, 4, 2, 8]), 'increase'),
    (lambda s: s.update(checkpoints=[True, 2, 4, 8]), 'checkpoints'),
    (lambda s: s.update(checkpoints=[0, 2, 4, 8]), 'checkpoints'),
    (lambda s: s.update(checkpoints=[1, 2, 4, 1e16]), 'checkpoints'),
    (lambda s: s.update(endpoints=[]), 'endpoints'),
    (lambda s: s.update(endpoints=[dict(name=f'e{i}', target=None, plateau_delta=.02) for i in range(9)]), 'endpoints'),
    (lambda s: s['endpoints'][0].update(name='checkpoint'), 'reserved'),
    (lambda s: s['endpoints'][1].update(name=s['endpoints'][0]['name']), 'unique'),
    (lambda s: s['endpoints'][0].pop('target'), 'target'),
    (lambda s: s['endpoints'][0].update(target=1.5), 'target'),
    (lambda s: s['endpoints'][0].update(target=False), 'target'),
    (lambda s: s['endpoints'][0].update(plateau_delta=0), 'plateau_delta'),
    (lambda s: s['model'].update(prior_scales=[.25]), 'prior_scales'),
    (lambda s: s['model'].update(seed_sds=[.15, .2, .3, .4, .5]), 'seed_sds'),
    (lambda s: s['model'].update(delayed_knees=[.5, .99]), 'delayed_knees'),
    (lambda s: s['model'].update(noise_floors=[.05, .05]), 'distinct'),
    (lambda s: s['stop'].update(min_checkpoints=True), 'min_checkpoints'),
    (lambda s: s['stop'].update(min_checkpoints=3.0), 'min_checkpoints'),
    (lambda s: s['stop'].update(min_checkpoints=2), 'min_checkpoints'),
    (lambda s: s['stop'].update(min_checkpoints=5), 'min_checkpoints'),
    (lambda s: s['stop'].update(max_invalid_rate=.1), 'max_invalid_rate'),
    (lambda s: s['budget'].update(max_usd=-1), 'max_usd'),
    (lambda s: s['budget'].update(max_usd=1e7), 'max_usd'),
    (lambda s: s['budget'].update(max_seconds=0), 'max_seconds'),
    (lambda s: s['budget'].update(line='API'), 'line'),
    (lambda s: s['budget'].update(ledger='ledger.csv'), 'ledger'),
    (lambda s: s['resource'].update(kind='cloud'), 'kind'),
    (lambda s: s['resource'].update(note=''), 'note'),
    (lambda s: s['stages'].pop(), 'one bounded stage'),
    (lambda s: stage0(s).update(id='a/b'), 'safe'),
    (lambda s: s['stages'][1].update(id=stage0(s)['id']), 'unique'),
    (lambda s: stage0(s).update(checkpoint=3), 'position'),
    (lambda s: stage0(s).update(argv=[]), 'argv'),
    (lambda s: stage0(s).update(argv=[sys.executable, 'a\x00b']), 'NUL'),
    (lambda s: stage0(s).update(argv=[sys.executable, 7]), 'argv'),
    (lambda s: stage0(s).update(inputs={'relative.py': '0'*64}), 'inputs'),
    (lambda s: stage0(s).update(inputs={'/abs.py': 'A'*64}), 'inputs'),
    (lambda s: stage0(s).update(inputs={'/abs.py': '0'*63}), 'inputs'),
    (lambda s: s['stages'][1].update(evidence=stage0(s)['evidence']), 'evidence'),
    (lambda s: stage0(s).update(max_seconds=61.), 'within the pilot budget'),
    (lambda s: stage0(s).update(max_usd=.01), 'within the pilot budget'),
    (lambda s: stage0(s).update(max_usd=True), 'max_usd'),
    (lambda s: stage0(s).update(note='x'), 'note'),
]


@pytest.mark.parametrize('change,word', SPEC_RULES)
def test_every_spec_rule_refuses_and_names_the_field(spec, change, word):
    change(spec)
    with pytest.raises(ValueError, match='invalid pilot specification') as refused:
        pilot.validate_spec(spec)
    assert word in str(refused.value)


def test_a_valid_spec_is_returned_unchanged(spec):
    before = json.dumps(spec, sort_keys=True)
    assert pilot.validate_spec(spec) is spec and json.dumps(spec, sort_keys=True) == before


def test_spec_file_with_a_duplicate_key_or_nan_is_refused(spec, tmp_path):
    path = tmp_path/'twice.json'
    text = json.dumps(spec)
    path.write_text(text[:-1] + ', "id": "again"}')
    with pytest.raises(ValueError, match='duplicate'):
        pilot.load_spec(path)
    path.write_text(text.replace('"max_usd": 0.0', '"max_usd": NaN', 1))
    with pytest.raises(ValueError, match='NaN'):
        pilot.load_spec(path)


ROW_RULES = [
    (lambda r: r.update(extra=1), 'extra'),
    (lambda r: r.pop('quality'), 'quality'),
    (lambda r: r.update(seed=1), 'seed'),
    (lambda r: r.update(checkpoint=True), 'checkpoint'),
    (lambda r: r.update(checkpoint=-1), 'checkpoint'),
    (lambda r: r['effects'][next(iter(r['effects']))].update(mean=1.5), 'mean'),
    (lambda r: r['effects'][next(iter(r['effects']))].update(se=-.1), 'se'),
    (lambda r: r['effects'][next(iter(r['effects']))].update(extra=0), 'extra'),
    (lambda r: r['effects'].pop(next(iter(r['effects']))), 'endpoints'),
    (lambda r: r['quality'].update(valid=1), 'valid'),
    (lambda r: r['quality'].update(invalid_rate=2), 'invalid_rate'),
    (lambda r: r['quality'].update(reason=None), 'reason'),
    (lambda r: r.update(sources=[]), 'sources'),
    (lambda r: r.update(sources=['0'*64, '0'*64]), 'sources'),
    (lambda r: r.update(sources=['xyz']), 'sources'),
    (lambda r: r.update(spec_hash='0'*64), 'spec hash'),
    (lambda r: r.update(seed='unplanned'), 'seed'),
    (lambda r: r.update(intervention_compute=2), 'compute'),
]


@pytest.mark.parametrize('change,word', ROW_RULES)
def test_every_evidence_rule_refuses_and_names_the_field(spec, change, word):
    rows = synthetic_rows(spec, 1, 'null')
    change(rows[0])
    with pytest.raises(ValueError) as refused:
        pilot.validate_rows(spec, rows)
    assert word in str(refused.value)


def test_pilot_model_numbers_use_the_shared_check():
    assert pilot_model._number is budget.finite
    for value in (True, '1', float('nan'), Decimal('Infinity'), Decimal('1e400'), None):
        with pytest.raises(ValueError, match='a finite number'):
            budget.finite(value, 'x')
    assert budget.finite(-2, 'x') == -2.0 and budget.finite(Decimal('0.5'), 'x') == .5
    with pytest.raises(ValueError, match='positive'):
        budget.finite(0, 'x', positive=True)
    with pytest.raises(ValueError, match='nonnegative'):
        budget.nonnegative_finite(-1e-9, 'x')
    with pytest.raises(ValueError, match='x must be a finite, nonnegative number'):
        budget.nonnegative_finite('1.5', 'x')  # text is parsed by the option that reads it, never here


# --- Runner records -------------------------------------------------------------------------------------------

@pytest.fixture
def finished_pilot(tmp_path, monkeypatch):
    from test_pilot_runner import fixture_spec
    def decide(spec, rows):
        points = sorted({row['checkpoint'] for row in rows})
        done = len(points) == len(spec['checkpoints'])
        return dict(stop=done, reason='compute_cap' if done else 'continue', posterior={},
                    next_checkpoint=None if done else spec['checkpoints'][len(points)])
    monkeypatch.setattr(pilot, 'decide', decide)
    path, _ = fixture_spec(tmp_path)
    out = tmp_path/'out'
    runner.run(path, out, execute=True)
    return path, out


RECEIPT_CHANGES = [  # (file, change, refusal)
    ('freeze.json', lambda r: r.update(started_at=True), 'freeze record'),
    ('freeze.json', lambda r: r.update(extra=1), 'freeze record'),
    ('freeze.json', lambda r: r.update(started_at=4e9), 'freeze record'),
    ('freeze.json', lambda r: r.update(sources=[]), 'freeze record'),
    ('stages/0000.started.json', lambda r: r.update(reservation_key=7), 'reservation_key'),
    ('stages/0000.started.json', lambda r: r.update(extra=1), 'extra'),
    ('stages/0000.result.json', lambda r: r.update(status='done'), 'status'),
    ('stages/0000.result.json', lambda r: r.update(exit_code=0.0), 'exit_code'),
    ('stages/0000.result.json', lambda r: r.update(extra=1), 'extra'),
    ('stages/0000.result.json', lambda r: r.pop('decision'), 'completed stage'),
    ('stages/0000.result.json', lambda r: r['decision'].update(stop='yes'), 'decision.stop'),
    ('stages/0000.result.json', lambda r: r.update(failure_reason='time_cap'), 'completed stage'),
    ('decision.json', lambda r: r.update(stop=1), 'stop'),
    ('decision.json', lambda r: r.update(reserved_usd='3'), 'reserved_usd'),
    ('decision.json', lambda r: r.update(extra=1), 'extra'),
    ('decision.json', lambda r: r.update(phase='other'), 'phase'),
]


@pytest.mark.parametrize('name,change,word', RECEIPT_CHANGES)
def test_tampered_receipts_refuse_resume_and_name_the_field(finished_pilot, monkeypatch, name, change, word):
    path, out = finished_pilot
    record = json.loads((out/name).read_text())
    change(record)
    (out/name).write_text(json.dumps(record))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('a tampered pilot relaunched'))
    with pytest.raises(ValueError) as refused:
        runner.run(path, out, execute=True)
    assert word in str(refused.value)


def test_a_receipt_that_is_not_an_object_is_refused_not_crashed(finished_pilot):
    path, out = finished_pilot
    (out/'stages'/'0000.started.json').write_text('[]\n')
    with pytest.raises(ValueError, match='invalid stage reservation receipt'):
        runner.run(path, out, execute=True)


# --- The reservation registry and the ledger ------------------------------------------------------------------

def registry(tmp_path, text):
    ledger = tmp_path/'ledger.csv'
    ledger.write_text('line,total_usd,status\n')
    budget.reservation_path(ledger).write_text(text)
    return ledger


GOOD = '{"key": "k", "line": "api", "total_usd": 10, "status": "pilot worst-case reservation"}'


@pytest.mark.parametrize('text,word', [
    ('{}', 'not a list'),
    ('[' + GOOD[:-1] + ', "key": "j"}]', 'duplicate'),
    ('[' + GOOD.replace('10', 'Infinity') + ']', 'Infinity'),
    ('[' + GOOD[:-1] + ', "extra": 1}]', 'extra'),
    ('[' + GOOD.replace('pilot worst-case', 'pilot') + ']', 'status'),
    ('[' + GOOD.replace('"k"', '7') + ']', 'key'),
    ('[' + GOOD.replace('10', '"10"') + ']', 'total_usd'),
    ('[' + GOOD[:-1] + ', "approved_usd": null}]', 'approved_usd'),
    ('[' + GOOD[:-1] + ', "approved_usd": 10.000001}]', 'approved_usd'),
    ('[' + GOOD + ', ' + GOOD + ']', 'key appears twice'),
])
def test_registry_rules(tmp_path, text, word):
    with pytest.raises(ValueError) as refused:
        budget.read_reservations(registry(tmp_path, text))
    assert word in str(refused.value)


def test_registry_numbers_are_exact_and_round_trip(tmp_path):
    ledger = registry(tmp_path, '[' + GOOD.replace('10', '0.3') + ']')
    for _ in range(3):
        assert not budget.claim(ledger, 'k', 'api', 0, 0, '0.1')['problems']
    assert budget.load_reservations(ledger)[0].approved_usd == Decimal('0.3')
    assert json.loads(budget.reservation_path(ledger).read_text())[0]['approved_usd'] == .3
    assert budget.claim(ledger, 'k', 'api', 0, 0, '0.000001')['problems']


def test_ledger_rows_are_strict(tmp_path):
    ledger = tmp_path/'ledger.csv'
    ledger.write_text('line,total_usd,status,colour\napi,1,completed,red\n')
    with pytest.raises(ValueError, match='ledger row 1: colour'):
        budget.read_ledger(ledger)
    ledger.write_text('line,total_usd\napi,1\n')
    with pytest.raises(ValueError, match='ledger row 1: status'):
        budget.read_ledger(ledger)
    ledger.write_text(','.join(budget.FIELDS) + '\n2026-11-25,r,m,p,r,x,2.70,0,2.70,completed,e,api\n')
    with pytest.raises(ValueError, match='hours must be'):
        budget.read_ledger(ledger)


def test_ledger_sums_are_exact_decimals(tmp_path):
    ledger = tmp_path/'ledger.csv'
    ledger.write_text('line,total_usd,status\napi,0.1,completed\napi,0.2,completed\n')
    assert budget._spent(budget.read_ledger(ledger), line='api') == Decimal('0.3') != Decimal(.1+.2)
    ledger.write_text('line,total_usd,status\n' + 'api,0.1,completed\n'*3 + 'api,99.7,completed\n')
    rows = budget.read_ledger(ledger)
    assert budget._spent(rows, line='api') == Decimal('100.000000')
    # The line now holds exactly $100 of its $300: exactly $200 more fits, a microdollar more does not.
    assert budget.check(rows, 'api', '0', '0', '200')[1] == []
    assert budget.check(rows, 'api', '0', '0', '200.000001')[1]


def test_prices_are_parsed_from_text_without_a_float():
    assert budget.number_text('0.1') == Decimal('0.1') and budget.dollars('2.70', 'x') == Decimal('2.70')
    assert budget.dollars(1.15*100, 'x') == Decimal('114.99999999999999')  # a float is read as its shortest decimal
    for value in (True, None, [], 'nan', '1e400', '-0.01', 'twelve'):
        with pytest.raises(ValueError, match='x must be a finite, nonnegative number'):
            budget.dollars(value, 'x')
    assert budget.dollars('-0.01', 'x', refund=True) == Decimal('-0.01')
    with pytest.raises(ValueError, match='no greater than zero'):
        budget.dollars('0.01', 'x', refund=True)


def test_rounding_directions_live_in_one_place():
    assert budget.micro(Decimal('0.0000005')) == 0 and budget.micro(Decimal('0.0000015')) == Decimal('0.000002')  # half even
    assert budget.micro(Decimal('0.0000001'), up=True) == Decimal('0.000001')  # a debit is never free
    assert budget.micro(Decimal('114.99999999999999'), up=True) == 115  # binary noise on a whole price is absorbed


def test_what_the_spend_gate_passes_still_works(tmp_path):
    """`gates.py` passes floats from argparse and writes the worst case into a JSON launch record."""
    worst, problems = budget.check(budget.read_ledger(ROOT/'spending'/'ledger.csv'), 'gpt-oss-20b-gpu', 2.70, 85/60, 1.0, None)
    assert type(worst) is float and json.loads(json.dumps(worst)) == worst and problems == []


def test_cli_reads_prices_as_text(tmp_path):
    ledger = tmp_path/'ledger.csv'
    ledger.write_text('line,total_usd,status\napi,299.9,completed\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    env.pop(budget.RESERVATION_ENV, None)
    def check(other):
        return subprocess.run([sys.executable, str(ROOT/'entrypoints'/'budget.py'), '--ledger', str(ledger), 'check', '--line', 'api',
                               '--hourly', '0.1', '--max-hours', '0.5', '--other-usd', other], env=env, capture_output=True, text=True)
    assert check('0.05').returncode == 0 and check('0.050001').returncode == 1
    refused = check('twelve')
    assert refused.returncode == 2 and "'twelve' is not a number" in refused.stderr


def test_registry_lock_semantics(tmp_path):
    """Refused at once with no wait, waited for up to `wait`, exclusive across processes, released on exit."""
    import time
    lock = tmp_path/'r.lock'
    holder = subprocess.Popen([sys.executable, '-c', 'import sys,time\nfrom rewardhack.state_files import file_lock\n'
                               f'with file_lock({str(lock)!r}):\n print("held", flush=True); time.sleep(1.5)'],
                              env=dict(os.environ, PYTHONPATH=str(ROOT/'src')), stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == 'held'
    started = time.monotonic()
    with pytest.raises(ValueError, match='another process holds'):
        with file_lock(lock):
            pass
    assert time.monotonic()-started < .2
    started = time.monotonic()
    with pytest.raises(ValueError, match='another process holds'):
        with file_lock(lock, wait=.3):
            pass
    assert .3 <= time.monotonic()-started < 1
    with file_lock(lock, wait=5):
        assert time.monotonic()-started >= .9  # acquired only once the holder's 1.5 s hold ended
    assert holder.wait(timeout=5) == 0
    with file_lock(lock):
        pass  # released on exit
