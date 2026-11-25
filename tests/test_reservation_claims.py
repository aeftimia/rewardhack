"""Claiming is a durable, cumulative authorization; read-only checks never spend it."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rewardhack import budget
from rewardhack.state_files import file_lock

SCRIPT = Path(__file__).resolve().parents[1]/'entrypoints'/'budget.py'


@pytest.fixture
def ledger(tmp_path):
    path=tmp_path/'ledger.csv'
    path.write_text('line,total_usd,status\n')
    budget.reservation_path(path).write_text(json.dumps([dict(key='stage-k',line='api',total_usd=100,
                                                            status='pilot worst-case reservation')]))
    return path


def run_guard(ledger, amount, command='claim', key='stage-k', line='api'):
    env=dict(os.environ)
    env.pop(budget.RESERVATION_ENV,None)
    if key is not None:
        env[budget.RESERVATION_ENV]=key
    return subprocess.run([sys.executable,str(SCRIPT),'--ledger',str(ledger),command,'--line',line,
                           '--hourly','0','--max-hours','0','--other-usd',str(amount)],
                          env=env,capture_output=True,text=True)


def test_two_60_claims_cannot_share_100_even_after_recording_charge(ledger):
    assert run_guard(ledger,60).returncode==0
    ledger.write_text('line,total_usd,status\napi,60,completed first launch\n')
    result=run_guard(ledger,60)
    assert result.returncode==1 and 'cumulative launch claims' in result.stdout
    assert budget.read_reservations(ledger)[0]['approved_usd']==60
    assert run_guard(ledger,40).returncode==0
    assert budget.read_reservations(ledger)[0]['approved_usd']==100
    assert run_guard(ledger,.01).returncode==1


def test_no_charge_or_refund_can_reset_consumed_allowance(ledger):
    assert not budget.claim(ledger,'stage-k','api',0,0,60)['problems']
    assert budget.claim(ledger,'stage-k','api',0,0,60)['problems']
    ledger.write_text('line,total_usd,status\napi,60,completed\napi,-60,refund cancelled\n')
    assert budget.claim(ledger,'stage-k','api',0,0,60)['problems']
    assert budget.read_reservations(ledger)[0]['approved_usd']==60


def test_refused_claim_does_not_consume_or_rewrite(ledger):
    registry=budget.reservation_path(ledger)
    before=registry.read_bytes()
    result=budget.claim(ledger,'stage-k','api',0,0,101)
    assert result['problems'] and registry.read_bytes()==before
    result=budget.claim(ledger,'stage-k','api',0,0,60,launch_cap=50)
    assert result['problems'] and registry.read_bytes()==before


@pytest.mark.parametrize('key,line',[(None,'api'),('unknown','api'),('stage-k','storage')])
def test_missing_unknown_or_cross_line_claim_is_refused(ledger,key,line):
    before=budget.reservation_path(ledger).read_bytes()
    result=run_guard(ledger,1,key=key,line=line)
    assert result.returncode==1 and 'REFUSED' in result.stderr
    assert budget.reservation_path(ledger).read_bytes()==before


def test_readonly_check_never_omits_own_reservation_or_consumes_it(ledger):
    before=budget.reservation_path(ledger).read_bytes()
    # API cap300, stage100: a prospective250 must not pass by hiding own100.
    assert run_guard(ledger,250,command='check').returncode==1
    assert run_guard(ledger,60,command='check',key='stage-k').returncode==1
    assert run_guard(ledger,60,command='check',key='stage-k',line='storage').returncode==1
    assert run_guard(ledger,60,command='check',key=None).returncode==0
    assert budget.reservation_path(ledger).read_bytes()==before


def test_claim_cannot_ignore_existing_unbooked_commitments(ledger):
    assert not budget.claim(ledger,'stage-k','api',0,0,60)['problems']
    # An externally entered charge overcommits the grant while our first approval
    # has not been booked. Omitting own reservation would wrongly permit40 more.
    ledger.write_text('line,total_usd,status\napi,250,unrelated completed charge\n')
    result=budget.claim(ledger,'stage-k','api',0,0,40)
    assert result['problems'] and budget.read_reservations(ledger)[0]['approved_usd']==60


@pytest.mark.parametrize('amount',[None,True,-1,float('nan'),float('inf'),100.01])
def test_corrupt_approval_cannot_create_allowance(ledger,amount):
    records=budget.read_reservations(ledger)
    records[0]['approved_usd']=amount
    budget.reservation_path(ledger).write_text(json.dumps(records))
    with pytest.raises(ValueError):
        budget.read_ledger(ledger)
    with pytest.raises(ValueError):
        budget.claim(ledger,'stage-k','api',0,0,1)


def test_concurrent_claims_cannot_both_consume_same_allowance(ledger,tmp_path):
    barrier=tmp_path/'start'
    code='''import sys,time
from pathlib import Path
from rewardhack.budget import claim
while not Path(sys.argv[2]).exists(): time.sleep(.001)
try:
 result=claim(sys.argv[1],'stage-k','api',0,0,60)
 sys.exit(1 if result['problems'] else 0)
except ValueError:
 sys.exit(1)
'''
    first=subprocess.Popen([sys.executable,'-c',code,str(ledger),str(barrier)])
    second=subprocess.Popen([sys.executable,'-c',code,str(ledger),str(barrier)])
    barrier.touch()
    assert sorted([first.wait(timeout=10),second.wait(timeout=10)])==[0,1]
    assert budget.read_reservations(ledger)[0]['approved_usd']==60


def test_claim_uses_same_registry_lock_as_stage_reservation(ledger):
    registry=budget.reservation_path(ledger)
    with file_lock(registry.with_suffix(registry.suffix+'.lock')):
        result=run_guard(ledger,60)
    assert result.returncode==1 and 'another process holds' in result.stderr
    assert 'approved_usd' not in budget.read_reservations(ledger)[0]


def test_claim_publication_interruption_retains_consumed_allowance(ledger,monkeypatch):
    original=budget.atomic_json
    def write_then_interrupt(*args,**kwargs):
        original(*args,**kwargs)
        raise KeyboardInterrupt('caller did not get successful return')
    monkeypatch.setattr(budget,'atomic_json',write_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        budget.claim(ledger,'stage-k','api',0,0,60)
    monkeypatch.setattr(budget,'atomic_json',original)
    assert budget.read_reservations(ledger)[0]['approved_usd']==60
    assert budget.claim(ledger,'stage-k','api',0,0,60)['problems']


def test_claims_use_ledger_precision_at_exact_boundary(ledger):
    records=budget.read_reservations(ledger)
    records[0]['total_usd']=.3
    budget.reservation_path(ledger).write_text(json.dumps(records))
    for _ in range(3):
        assert not budget.claim(ledger,'stage-k','api',0,0,.1)['problems']
    assert budget.read_reservations(ledger)[0]['approved_usd']==.3
    assert budget.claim(ledger,'stage-k','api',0,0,.000001)['problems']


@pytest.mark.parametrize('amount',[.00000049,.00000001,1e-13,1e-100])
def test_positive_fractional_microclaims_never_disappear(ledger,amount):
    records=budget.read_reservations(ledger)
    records[0]['total_usd']=.000003
    budget.reservation_path(ledger).write_text(json.dumps(records))
    for _ in range(3):
        result=budget.claim(ledger,'stage-k','api',0,0,amount)
        assert not result['problems'] and result['worst_usd']==.000001
    assert budget.read_reservations(ledger)[0]['approved_usd']==.000003
    assert budget.claim(ledger,'stage-k','api',0,0,amount)['problems']


def test_claim_cost_uses_decimal_inputs_and_tolerates_binary_cap_noise(ledger):
    for hourly,hours,other,expected in [(1.15,100,0,115),(0,0,1.15*100,115),
                                       (0,0,.1+.2,.3),(2.7,24,0,64.8),
                                       (0,0,.00000149,.000002),(1e-200,1e-200,0,.000001)]:
        assert budget.claim_cost(hourly,hours,other)==expected
    records=budget.read_reservations(ledger);records[0]['total_usd']=1.15*100
    budget.reservation_path(ledger).write_text(json.dumps(records))
    assert not budget.claim(ledger,'stage-k','api',1.15,100)['problems']
    assert budget.read_reservations(ledger)[0]['approved_usd']==115


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-1])
def test_invalid_decimal_claim_inputs_fail_without_consumption(ledger,value):
    before=budget.reservation_path(ledger).read_bytes()
    with pytest.raises(ValueError):
        budget.claim(ledger,'stage-k','api',0,0,value)
    assert budget.reservation_path(ledger).read_bytes()==before


def test_huge_finite_claim_is_refused_without_decimal_exception(ledger):
    assert budget.claim(ledger,'stage-k','api',0,0,1e100)['problems']
    assert budget.claim(ledger,'stage-k','api',1e308,1,0)['problems']
    with pytest.raises(ValueError):
        budget.claim(ledger,'stage-k','api',1e308,1e308)
    assert 'approved_usd' not in budget.read_reservations(ledger)[0]


def test_boolean_launch_cap_does_not_authorize_claim(ledger):
    with pytest.raises(ValueError,match='launch cap'):
        budget.claim(ledger,'stage-k','api',0,0,.5,launch_cap=True)
    assert 'approved_usd' not in budget.read_reservations(ledger)[0]
