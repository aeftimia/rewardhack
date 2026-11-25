"""Offline subprocesses exercise bounded execution, receipts, and crash recovery."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from rewardhack import pilot, pilot_runner as runner

PRODUCER = '''import hashlib,json,sys
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text())
sha=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
point=float(sys.argv[3])
rows=[dict(spec_hash=sha,seed=seed,checkpoint=point,baseline_compute=point,intervention_compute=point,
 effects={'effect':{'mean':0.1,'se':0.05}},quality={'valid':True,'invalid_rate':0,'reason':''},
 sources=[hashlib.sha256(f'{seed}:{point}'.encode()).hexdigest()]) for seed in spec['seeds']]
Path(sys.argv[2]).write_text(''.join(json.dumps(row)+'\\n' for row in rows))
'''


def fixture_spec(tmp_path, *, ledger=None, cost=1, cap=10):
    tmp_path.mkdir(exist_ok=True)
    ledger = ledger or tmp_path/'ledger.csv'
    if not ledger.exists():
        ledger.write_text('line,total_usd,status\n')
    path = tmp_path/'spec.json'
    producer = tmp_path/'producer.py'
    producer.write_text(PRODUCER)
    inputs = {str(producer): hashlib.sha256(producer.read_bytes()).hexdigest()}
    spec = dict(schema='compute-pilot-v1', id='test', question='synthetic test',
                context=dict(population='synthetic', compute_axis='units', report_mode='typed',
                             baseline='base', intervention='test', evaluation_protocol='fixed'),
                seeds=['a', 'b', 'c'], checkpoints=[1, 2, 4],
                endpoints=[dict(name='effect', target=None, plateau_delta=.01)],
                model=dict(prior_scales=[.25, .5], noise_floors=[.05, .1], seed_sds=[.1, .2], delayed_knees=[.4, .8]),
                stop=dict(confidence=.95, min_checkpoints=3, min_compute_fraction=.5, max_invalid_rate=.05),
                budget=dict(max_usd=cap, max_seconds=60, line='storage', ledger=str(ledger)),
                resource=dict(kind='local_process', note='offline synthetic test'), stages=[])
    for index, point in enumerate(spec['checkpoints']):
        evidence = tmp_path/f'evidence-{index}.jsonl'
        spec['stages'].append(dict(id=f's{index}', checkpoint=point, max_usd=cost, max_seconds=5,
                                  evidence=str(evidence), inputs=inputs, argv=[sys.executable, str(producer), str(path), str(evidence), str(point)]))
    path.write_text(json.dumps(spec))
    return path, spec


@pytest.fixture
def cheap_decision(monkeypatch):
    def decide(spec, rows):
        points = sorted({row['checkpoint'] for row in rows})
        done = len(points) == len(spec['checkpoints'])
        return dict(stop=done, reason='compute_cap' if done else 'continue', posterior={},
                    next_checkpoint=None if done else spec['checkpoints'][len(points)])
    monkeypatch.setattr(pilot, 'decide', decide)


def test_default_plan_does_not_create_output_or_start_process(tmp_path, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('plan executed a command'))
    result = runner.run(path, tmp_path/'out')
    assert result['status'] == 'planned' and result['planned_worst_usd'] == 3
    assert not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('*.pilot-reservations*'))
    assert not any(Path(stage['evidence']).exists() for stage in spec['stages'])


def test_complete_matched_stages_and_latched_resume(tmp_path, cheap_decision, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    out = tmp_path/'out'
    result = runner.run(path, out, execute=True)
    assert result['reason'] == 'compute_cap' and result['reserved_usd'] == 3
    assert len(list((out/'stages').glob('*.result.json'))) == 3
    reservations = json.loads((tmp_path/'ledger.csv.pilot-reservations.json').read_text())
    assert len(reservations) == 3 and sum(r['total_usd'] for r in reservations) == 3
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('terminal pilot relaunched'))
    assert runner.run(path, out, execute=True) == result


def test_dollar_cap_suppresses_queued_stages_and_final_evaluation(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path, cap=2)
    result = runner.run(path, tmp_path/'out', execute=True)
    assert result['reason'] == 'budget_cap' and result['reserved_usd'] == 2
    assert not Path(spec['stages'][2]['evidence']).exists()


def test_model_stop_suppresses_queued_stages(tmp_path, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    monkeypatch.setattr(pilot, 'decide', lambda *a: dict(stop=True, reason='practical_plateau_within_budget', posterior={}))
    result = runner.run(path, tmp_path/'out', execute=True)
    assert result['reason'] == 'practical_plateau_within_budget' and result['reserved_usd'] == 1
    assert not Path(spec['stages'][1]['evidence']).exists()


def test_reservations_are_shared_across_pilots(tmp_path, monkeypatch):
    ledger = tmp_path/'ledger.csv'
    path, _ = fixture_spec(tmp_path/'one', ledger=ledger, cost=60, cap=180)
    monkeypatch.setattr(pilot, 'decide', lambda *a: dict(stop=True, reason='test_stop', posterior={}))
    runner.run(path, tmp_path/'out-one', execute=True)
    path2, spec2 = fixture_spec(tmp_path/'two', ledger=ledger, cost=60, cap=180)
    result = runner.run(path2, tmp_path/'out-two', execute=True)
    assert result['reason'] == 'grant_budget_cap' and result['reserved_usd'] == 0
    assert not Path(spec2['stages'][0]['evidence']).exists()


def test_failed_command_retains_reservation_and_never_retries(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    spec['stages'][0]['argv'] = [sys.executable, '-c', 'raise SystemExit(7)']
    path.write_text(json.dumps(spec))
    result = runner.run(path, tmp_path/'out', execute=True)
    assert result['reason'] == 'stage_failure' and result['reserved_usd'] == 1
    assert result['stage_result']['exit_code'] == 7
    assert runner.run(path, tmp_path/'out', execute=True) == result


def test_missing_completion_after_crash_does_not_rerun(tmp_path, cheap_decision, monkeypatch):
    path, _ = fixture_spec(tmp_path)
    original = runner._write
    def fail_result(path, value, **kw):
        if str(path).endswith('.result.json'):
            raise OSError('simulated crash before receipt')
        return original(path, value, **kw)
    monkeypatch.setattr(runner, '_write', fail_result)
    with pytest.raises(OSError, match='simulated crash'):
        runner.run(path, tmp_path/'out', execute=True)
    monkeypatch.setattr(runner, '_write', original)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('uncertain work repeated'))
    with pytest.raises(ValueError, match='no completion receipt'):
        runner.run(path, tmp_path/'out', execute=True)
    assert len(json.loads((tmp_path/'ledger.csv.pilot-reservations.json').read_text())) == 1


def test_completed_evidence_cannot_change_on_resume(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    runner.run(path, tmp_path/'out', execute=True)
    Path(spec['stages'][0]['evidence']).write_text('{}\n')
    with pytest.raises(ValueError, match='evidence changed'):
        runner.run(path, tmp_path/'out', execute=True)


def test_spec_and_source_changes_refuse_resume(tmp_path, cheap_decision, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    runner.run(path, tmp_path/'out', execute=True)
    spec['question'] = 'changed question'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='specification, source code'):
        runner.run(path, tmp_path/'out', execute=True)
    spec['question'] = 'synthetic test'
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(runner, '_sources', lambda: {})
    with pytest.raises(ValueError, match='specification, source code'):
        runner.run(path, tmp_path/'out', execute=True)


def test_stale_evidence_and_missing_ledger_refuse_before_spawn(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    Path(spec['stages'][0]['evidence']).write_text('{}\n')
    with pytest.raises(ValueError, match='stale evidence'):
        runner.run(path, tmp_path/'out', execute=True)
    Path(spec['stages'][0]['evidence']).unlink()
    Path(spec['budget']['ledger']).unlink()
    with pytest.raises(ValueError, match='ledger must exist'):
        runner.run(path, tmp_path/'out', execute=True)


def test_gap_in_receipts_refuses_before_launch(tmp_path, cheap_decision, monkeypatch):
    path, _ = fixture_spec(tmp_path)
    out = tmp_path/'out'
    runner.run(path, out, execute=True)
    (out/'stages'/'0000.started.json').unlink()
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('gapped receipts relaunched'))
    with pytest.raises(ValueError, match='contiguous prefix'):
        runner.run(path, out, execute=True)


def test_timeout_kills_child_process_group(tmp_path):
    marker = tmp_path/'child-was-not-killed'
    code = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",'
            + repr(f'import time; from pathlib import Path; time.sleep(.6); Path({str(marker)!r}).touch()')
            + ']); time.sleep(10)')
    result = runner._execute(dict(argv=[sys.executable, '-c', code], max_seconds=.1), tmp_path/'log')
    assert result['timed_out']
    time.sleep(.7)
    assert not marker.exists()


def test_missing_evidence_is_measurement_failure(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    spec['stages'][0]['argv'] = [sys.executable, '-c', 'pass']
    path.write_text(json.dumps(spec))
    result = runner.run(path, tmp_path/'out', execute=True)
    assert result['reason'] == 'measurement_failure' and result['reserved_usd'] == 1


def test_output_lock_prevents_concurrent_launch(tmp_path, cheap_decision):
    path, _ = fixture_spec(tmp_path)
    out = tmp_path/'out'; out.mkdir()
    with runner._lock(out/'pilot.lock'):
        with pytest.raises(ValueError, match='another process holds'):
            runner.run(path, out, execute=True)


def test_overall_time_cap_prevents_stage_start(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    spec['budget']['max_seconds'] = 5
    path.write_text(json.dumps(spec))
    result = runner.run(path, tmp_path/'out', execute=True)
    assert result['reason'] == 'time_cap' and result['reserved_usd'] == 0


def test_static_inputs_are_checked_even_for_plan_and_resume(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    runner.run(path, tmp_path/'out', execute=True)
    script = Path(next(iter(spec['stages'][0]['inputs'])))
    script.write_text(script.read_text()+'\n# changed producer\n')
    with pytest.raises(ValueError, match='static stage input changed'):
        runner.run(path, tmp_path/'planned')
    with pytest.raises(ValueError, match='static stage input changed'):
        runner.run(path, tmp_path/'out', execute=True)


def test_terminal_decision_is_recomputed_from_evidence(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    out = tmp_path/'out'
    runner.run(path, out, execute=True)
    terminal = json.loads((out/'decision.json').read_text())
    terminal['reason'] = 'evidence_for_at_tested_compute'
    (out/'decision.json').write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match='disagrees with immutable evidence'):
        runner.run(path, out, execute=True)


def test_missing_shared_reservation_refuses_resume(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    out = tmp_path/'out'
    runner.run(path, out, execute=True)
    (tmp_path/'ledger.csv.pilot-reservations.json').unlink()
    with pytest.raises(ValueError, match='missing or changed in shared registry'):
        runner.run(path, out, execute=True)


def test_reservation_write_crash_before_stage_receipt_never_relaunches(tmp_path, cheap_decision, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    original = runner._write
    def fail_started(path, value, **kw):
        if str(path).endswith('.started.json'):
            raise OSError('crash after reservation')
        return original(path, value, **kw)
    monkeypatch.setattr(runner, '_write', fail_started)
    with pytest.raises(OSError, match='crash after reservation'):
        runner.run(path, tmp_path/'out', execute=True)
    monkeypatch.setattr(runner, '_write', original)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: pytest.fail('uncertain reservation relaunched'))
    with pytest.raises(ValueError, match='already reserved'):
        runner.run(path, tmp_path/'out', execute=True)


def test_remaining_wall_time_bounds_spawn_after_reservation_overhead(tmp_path, cheap_decision, monkeypatch):
    path, spec = fixture_spec(tmp_path)
    wall, mono = [1000.0], [100.0]
    monkeypatch.setattr(runner.time, 'time', lambda: wall[0])
    monkeypatch.setattr(runner.time, 'monotonic', lambda: mono[0])
    reserve = runner._reserve
    def slow_reservation(*args):
        result = reserve(*args)
        mono[0] += 59
        wall[0] -= 10  # a backward wall-clock correction cannot reset the live cap
        return result
    monkeypatch.setattr(runner, '_reserve', slow_reservation)
    observed = []
    def execute(stage, log, timeout=None, **options):
        observed.append(timeout)
        return dict(exit_code=1, timed_out=False, elapsed_seconds=0)
    monkeypatch.setattr(runner, '_execute', execute)
    runner.run(path, tmp_path/'out', execute=True)
    assert observed == [1.0]


def test_invalid_shared_registry_cannot_weaken_budget(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path)
    registry = tmp_path/'ledger.csv.pilot-reservations.json'
    registry.write_text(json.dumps([dict(key='x', line='storage', total_usd=-100, status='pilot worst-case reservation')]))
    with pytest.raises(ValueError, match='finite, nonnegative'):
        runner.run(path, tmp_path/'out', execute=True)
    assert not Path(spec['stages'][0]['evidence']).exists()


@pytest.mark.parametrize('changed', ['spec', 'input', 'source'])
def test_child_cannot_change_frozen_inputs_before_scientific_decision(tmp_path, cheap_decision, monkeypatch, changed):
    path, spec = fixture_spec(tmp_path)
    real_execute = runner._execute
    source_hashes = runner._sources()
    calls = []
    def mutate_after_child(stage, log, timeout=None, **options):
        result = real_execute(stage, log, timeout, **options)
        calls.append(stage['id'])
        if changed == 'spec':
            altered = dict(spec, question='changed while running')
            path.write_text(json.dumps(altered))
        elif changed == 'input':
            script = Path(next(iter(stage['inputs'])))
            script.write_text(script.read_text()+'\n# mutated while running\n')
        else:
            monkeypatch.setattr(runner, '_sources', lambda: dict(source_hashes, changed='yes'))
        return result
    monkeypatch.setattr(runner, '_execute', mutate_after_child)
    monkeypatch.setattr(pilot, 'decide', lambda *a: pytest.fail('mutated inputs reached a scientific decision'))
    with pytest.raises(ValueError, match='changed'):
        runner.run(path, tmp_path/'out', execute=True)
    decision = json.loads((tmp_path/'out'/'decision.json').read_text())
    assert decision['reason'] == 'stage_failure' and decision['reserved_usd'] == 1
    assert calls == ['s0']


@pytest.mark.parametrize('line', [None, '', 'unknown'])
def test_reservation_registry_cannot_hide_cost_on_untracked_line(tmp_path, cheap_decision, line):
    path, spec = fixture_spec(tmp_path)
    (tmp_path/'ledger.csv.pilot-reservations.json').write_text(json.dumps([
        dict(key='old', line=line, total_usd=99, status='pilot worst-case reservation')]))
    with pytest.raises(ValueError, match='invalid pilot reservation'):
        runner.run(path, tmp_path/'out', execute=True)


@pytest.mark.parametrize('number', [signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT, signal.SIGINT])
def test_catchable_signal_to_runner_kills_the_detached_stage(tmp_path, number):
    marker = tmp_path/'stage-outlived-the-runner'
    stage = dict(argv=[sys.executable, '-c', f'import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()'],
                 max_seconds=10)
    code = ('import sys; from pathlib import Path; from rewardhack import pilot_runner as r; '
            f'print("ready", flush=True); r._execute({stage!r}, Path({str(tmp_path/"log")!r}))')
    env = dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parents[1]))
    process = subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout.readline().strip() == b'ready'
    time.sleep(.4)
    process.send_signal(number)
    process.wait(timeout=5)
    time.sleep(1)
    assert not marker.exists()


def test_ignored_hangup_stays_ignored_so_nohup_still_protects_a_run(tmp_path):
    stage = dict(argv=[sys.executable, '-c', 'import os,signal,time; os.kill(os.getppid(), signal.SIGHUP); time.sleep(.3)'], max_seconds=10)
    code = ('import signal; from pathlib import Path; from rewardhack import pilot_runner as r; signal.signal(signal.SIGHUP, signal.SIG_IGN); '
            f'print(r._execute({stage!r}, Path({str(tmp_path/"log")!r}))["exit_code"], signal.getsignal(signal.SIGHUP) is signal.SIG_IGN)')
    env = dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parents[1]))
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=20)
    assert result.stdout.split() == ['0', 'True'], result.stderr


def test_stages_that_exactly_fill_the_pilot_cap_all_run(tmp_path, cheap_decision):
    path, spec = fixture_spec(tmp_path, cost=.1, cap=.3)  # 0.1+0.1+0.1 > 0.3 in binary floating point
    assert runner.plan(path)['planned_worst_usd'] == .3
    result = runner.run(path, tmp_path/'out', execute=True)
    assert result['reason'] == 'compute_cap' and result['reserved_usd'] == .3
    assert runner.run(path, tmp_path/'out', execute=True) == result


def test_stage_claims_its_allowance_once_without_double_counting(tmp_path, cheap_decision):
    """A stage's own reservation covers its launch; anything above it, or on another line, is still refused."""
    path, spec = fixture_spec(tmp_path, cost=60, cap=60)
    guard = Path(__file__).resolve().parents[1]/'entrypoints'/'budget.py'
    report = tmp_path/'guard.json'
    code = ('import json,os,subprocess,sys; from pathlib import Path\n'
            f'run=lambda line,usd: subprocess.run([sys.executable,{str(guard)!r},"--ledger",{spec["budget"]["ledger"]!r},"claim",'
            '"--line",line,"--hourly","0","--max-hours","0","--other-usd",usd]).returncode\n'
            f'Path({str(report)!r}).write_text(json.dumps([run("storage","60"),run("storage","60.01"),run("api","300")]))')
    spec['stages'][0]['argv'] = [sys.executable, '-c', code]
    path.write_text(json.dumps(spec))
    env = dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parents[1]))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, 'environ', env)
        runner.run(path, tmp_path/'out', execute=True)
    assert json.loads(report.read_text()) == [0, 1, 1]
    # Outside the stage the reservation counts in full again.
    outside = subprocess.run([sys.executable, str(guard), '--ledger', spec['budget']['ledger'], 'check', '--line', 'storage',
                              '--hourly', '0', '--max-hours', '0', '--other-usd', '60'], env=env, capture_output=True, text=True)
    assert outside.returncode == 1 and 'REFUSED' in outside.stdout


def test_second_stop_signal_does_not_interrupt_the_cleanup(tmp_path):
    marker = tmp_path/'stage-outlived-the-runner'
    stage = dict(argv=[sys.executable, '-c', f'import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()'],
                 max_seconds=10)
    code = ('from pathlib import Path; from rewardhack import pilot_runner as r\n'
            'print("ready", flush=True)\n'
            f'try: r._execute({stage!r}, Path({str(tmp_path/"log")!r}))\n'
            'except KeyboardInterrupt as stop: print(stop, flush=True)')
    env = dict(os.environ, PYTHONPATH=str(Path(runner.__file__).parents[1]))
    process = subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == 'ready'
    time.sleep(.4)
    process.send_signal(signal.SIGTERM)
    time.sleep(.001)
    process.send_signal(signal.SIGHUP)
    output, errors = process.communicate(timeout=15)  # a hang here is the defect
    # Pending handlers run in signal-number order, so either signal may be the one reported.
    assert output.strip() in ('pilot received SIGTERM', 'pilot received SIGHUP') and not errors
    time.sleep(1)
    assert not marker.exists()


@pytest.mark.parametrize('moment', ['starting', 'cleaning'])
def test_stop_signal_while_the_stage_is_started_or_cleaned_up_still_kills_it(tmp_path, monkeypatch, moment):
    """A handler that raised here would skip the kill: mid-start `process` is unset, mid-cleanup the kill is the next statement."""
    marker = tmp_path/'stage-outlived-the-runner'
    hold = '2' if moment == 'starting' else '0'
    stage = dict(argv=[sys.executable, '-c', f'import time; from pathlib import Path; time.sleep({hold}); Path({str(marker)!r}).touch()'],
                 max_seconds=10)
    if moment == 'starting':
        real = subprocess._fork_exec
        def fork_then_signal(*args, **kwargs):
            result = real(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGTERM)  # handled before Popen returns
            return result
        monkeypatch.setattr(subprocess, '_fork_exec', fork_then_signal)
    else:
        real = runner._terminate
        def signal_then_terminate(process):
            os.kill(os.getpid(), signal.SIGTERM)
            real(process)
        monkeypatch.setattr(runner, '_terminate', signal_then_terminate)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt, match='pilot received SIGTERM'):
        runner._execute(stage, tmp_path/'log')
    assert signal.getsignal(signal.SIGTERM) is before
    time.sleep(2.3 if moment == 'starting' else 0)
    assert marker.exists() == (moment == 'cleaning')  # mid-start: killed before it wrote; mid-cleanup: it had finished


@pytest.mark.parametrize('cost,cap,count', [(.00000049, .000001, 1), (10/3, 10, 2)])
def test_fractional_stage_commitments_bound_claims_and_replay(tmp_path, cheap_decision, cost, cap, count):
    from rewardhack import budget
    path, spec = fixture_spec(tmp_path, cost=cost, cap=cap)
    producer = tmp_path/'producer.py'
    producer.write_text(producer.read_text()+'''
import os
from rewardhack.budget import claim
stage=next(stage for stage in spec['stages'] if stage['checkpoint']==point)
assert not claim(spec['budget']['ledger'],os.environ['REWARDHACK_PILOT_RESERVATION'],
                 spec['budget']['line'],0,0,stage['max_usd'])['problems']
''')
    for stage in spec['stages']:
        stage['inputs'] = {str(producer): hashlib.sha256(producer.read_bytes()).hexdigest()}
    path.write_text(json.dumps(spec))
    unit_cost = budget.claim_cost(0, 0, cost)
    assert runner.plan(path)['planned_worst_usd'] == round(3*unit_cost, 6)
    out = tmp_path/'out'
    result = runner.run(path, out, execute=True)
    assert result['reason'] == 'budget_cap'
    assert result['reserved_usd'] == round(count*unit_cost, 6)
    records = budget.read_reservations(spec['budget']['ledger'])
    assert len(records) == count
    assert all(row['total_usd'] == row['approved_usd'] == unit_cost for row in records)
    assert not Path(spec['stages'][count]['evidence']).exists()
    assert runner.run(path, out, execute=True) == result


@pytest.mark.parametrize('cap', [.1234567, 1.15*100])
def test_a_stage_equal_to_a_cap_that_rounds_up_runs_and_resumes(tmp_path, cheap_decision, cap):
    path, spec = fixture_spec(tmp_path, cost=cap, cap=cap)
    spec['budget']['line'] = 'api'  # enough grant headroom for the $115 case
    path.write_text(json.dumps(spec))
    out = tmp_path/'out'
    result = runner.run(path, out, execute=True)
    assert result['reason'] == 'budget_cap'  # one full-cap stage, then no second launch
    assert result['reserved_usd'] == round(cap, 6)
    assert Path(spec['stages'][0]['evidence']).exists()
    assert not Path(spec['stages'][1]['evidence']).exists()
    assert runner.run(path, out, execute=True) == result


def test_evidence_analysis_is_bounded_by_remaining_pilot_time(tmp_path,monkeypatch):
    path,spec=fixture_spec(tmp_path)
    spec['budget']['max_seconds']=1.5
    for stage in spec['stages']:
        stage['max_seconds']=1  # generous for a slow interpreter start; the analysis is what must be cut off
    path.write_text(json.dumps(spec))
    def too_slow(*unused):
        time.sleep(6)
    monkeypatch.setattr(pilot,'decide',too_slow)
    started=time.monotonic()
    result=runner.run(path,tmp_path/'out',execute=True)
    assert time.monotonic()-started < 4
    assert result['reason']=='time_cap' and result['reserved_usd']==1
    receipt=json.loads((tmp_path/'out'/'stages'/'0000.result.json').read_text())
    assert receipt['failure_reason']=='time_cap' and receipt['status']=='failed'
    assert Path(spec['stages'][0]['evidence']).exists()
    assert not Path(spec['stages'][1]['evidence']).exists()
    assert runner.run(path,tmp_path/'out',execute=True)==result  # no analysis rerun or paid retry


def test_terminal_replay_has_fresh_bounded_audit_allowance(tmp_path,cheap_decision,monkeypatch):
    path,spec=fixture_spec(tmp_path)
    out=tmp_path/'out';result=runner.run(path,out,execute=True)
    original=runner.time.time
    monkeypatch.setattr(runner.time,'time',lambda: original()+3600)
    assert runner.run(path,out,execute=True)==result  # original run deadline is past
    original_analyze=runner._analyze
    def bounded(spec,paths,seconds):
        assert seconds==min(spec['budget']['max_seconds'],30)
        raise runner.TimeCap('slow terminal audit')
    monkeypatch.setattr(runner,'_analyze',bounded)
    before=(out/'decision.json').read_bytes()
    with pytest.raises(ValueError,match='terminal decision audit'):
        runner.run(path,out,execute=True)
    assert (out/'decision.json').read_bytes()==before
    monkeypatch.setattr(runner,'_analyze',original_analyze)


def test_analysis_leaves_the_callers_alarm_timer_and_handler_alone(tmp_path,cheap_decision):
    path,_=fixture_spec(tmp_path)
    def custom(*unused):
        pytest.fail('the caller\'s alarm must not fire during a short pilot')
    original=signal.signal(signal.SIGALRM,custom)
    signal.setitimer(signal.ITIMER_REAL,60)
    try:
        assert runner.run(path,tmp_path/'out',execute=True)['reason']=='compute_cap'
        assert signal.getsignal(signal.SIGALRM) is custom and signal.getitimer(signal.ITIMER_REAL)[0]>50
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,original)


def test_analysis_result_errors_and_deadline_cross_the_process_boundary(tmp_path,monkeypatch):
    path,spec=fixture_spec(tmp_path)
    monkeypatch.setattr(pilot,'read_evidence',lambda spec,paths:paths)
    monkeypatch.setattr(pilot,'decide',lambda spec,rows:dict(stop=False,value=.1+.2,rows=[str(r) for r in rows],none=None))
    assert runner._analyze(spec,[Path('a')],5)==dict(stop=False,value=.1+.2,rows=['a'],none=None)
    def invalid(*unused):
        raise OSError('evidence unreadable')
    monkeypatch.setattr(pilot,'decide',invalid)
    with pytest.raises(ValueError,match='evidence unreadable'):
        runner._analyze(spec,[],5)
    monkeypatch.setattr(pilot,'decide',lambda *a:os._exit(3))
    with pytest.raises(RuntimeError,match='without a result'):
        runner._analyze(spec,[],5)
    def swallows(*unused):
        while True:  # code that catches every ordinary error cannot outlast the deadline
            try:
                time.sleep(.01)
            except Exception:
                pass
    monkeypatch.setattr(pilot,'decide',swallows)
    started=time.monotonic()
    with pytest.raises(runner.TimeCap):
        runner._analyze(spec,[],.2)
    assert time.monotonic()-started<3
    with pytest.raises(runner.TimeCap):
        runner._analyze(spec,[],0)
    # The child's own alarm ending it (the parent was suspended across the deadline) is the time cap, not a crash.
    monkeypatch.setattr(pilot,'decide',lambda *a:os.kill(os.getpid(),signal.SIGALRM))
    with pytest.raises(runner.TimeCap):
        runner._analyze(spec,[],5)


def test_pilot_off_the_main_thread_is_refused_before_any_reservation(tmp_path):
    import threading
    path,spec=fixture_spec(tmp_path)
    seen=[]
    def attempt():
        try:
            runner.run(path,tmp_path/'out',execute=True)
        except ValueError as error:
            seen.append(str(error))
    thread=threading.Thread(target=attempt); thread.start(); thread.join()
    assert seen==['a pilot can only be executed from the main thread']
    assert not (tmp_path/'out').exists() and not (tmp_path/'ledger.csv.pilot-reservations.json').exists()


def test_late_resume_keeps_a_stop_recorded_in_the_last_receipt(tmp_path,cheap_decision,monkeypatch):
    path,spec=fixture_spec(tmp_path)
    out=tmp_path/'out'
    result=runner.run(path,out,execute=True)
    assert result['reason']=='compute_cap'
    (out/'decision.json').unlink()  # the runner died after the stage receipt, before the terminal decision
    original=runner.time.time
    monkeypatch.setattr(runner.time,'time',lambda:original()+3600)
    resumed=runner.run(path,out,execute=True)
    assert resumed['reason']=='compute_cap' and resumed['decision']==result['decision']


def test_expired_active_replay_stops_before_analysis(tmp_path,cheap_decision,monkeypatch):
    path,spec=fixture_spec(tmp_path,cap=1)
    out=tmp_path/'out'
    result=runner.run(path,out,execute=True)
    assert result['reason']=='budget_cap'
    (out/'decision.json').unlink()  # simulate a completed stage before terminal publication
    original=runner.time.time
    monkeypatch.setattr(runner.time,'time',lambda:original()+3600)
    monkeypatch.setattr(pilot,'decide',lambda *a:pytest.fail('expired run must not analyze'))
    result=runner.run(path,out,execute=True)
    assert result['reason']=='time_cap' and result['phase']=='evidence_analysis'


def test_stage_reservation_waits_for_a_registry_held_by_a_claim(tmp_path):
    import threading
    from rewardhack import budget
    from rewardhack.state_files import file_lock
    path,spec=fixture_spec(tmp_path)
    registry=budget.reservation_path(spec['budget']['ledger'])
    held,release=threading.Event(),threading.Event()
    def claimant():
        with file_lock(registry.with_suffix(registry.suffix+'.lock')):
            held.set(); release.wait(5)
    thread=threading.Thread(target=claimant); thread.start(); held.wait(5)
    threading.Timer(.5,release.set).start()
    started=time.monotonic()
    key,problems=runner._reserve(spec,tmp_path/'out',spec['stages'][0],pilot.digest(spec))
    thread.join()
    assert key and not problems and time.monotonic()-started>=.4


def test_analysis_keeps_the_callers_blocked_alarm_mask(monkeypatch):
    monkeypatch.setattr(pilot,'read_evidence',lambda *a:[])
    monkeypatch.setattr(pilot,'decide',lambda *a:dict(stop=False))
    original=signal.pthread_sigmask(signal.SIG_BLOCK,[signal.SIGALRM])
    try:
        assert runner._analyze({},[],5)==dict(stop=False)
        assert signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK,[])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,original)


@pytest.mark.parametrize('blocked_alarm',[False,True])
def test_analysis_child_ends_itself_if_the_runner_is_killed_while_waiting(tmp_path,blocked_alarm):
    from rewardhack.state_files import file_lock
    marker=tmp_path/'child'
    lock=tmp_path/'pilot.lock'
    code=('import os,sys,time,signal; from pathlib import Path; from rewardhack import pilot, pilot_runner as r\n'
          'from rewardhack.state_files import file_lock\n'
          f'if {blocked_alarm!r}: signal.pthread_sigmask(signal.SIG_BLOCK,[signal.SIGALRM])\n'
          'pilot.read_evidence=lambda spec,paths:paths\n'
          f'def decide(spec,rows):\n Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)\n'
          'pilot.decide=decide\n'
          f'with file_lock({str(lock)!r}): r._analyze({{}},[],.5)')
    env=dict(os.environ,PYTHONPATH=str(Path(runner.__file__).parents[1]))
    parent=subprocess.Popen([sys.executable,'-c',code],env=env)
    deadline=time.monotonic()+10
    while not (marker.exists() and marker.read_text()) and time.monotonic()<deadline:
        time.sleep(.02)
    child=int(marker.read_text())
    parent.kill(); parent.wait()  # before the 0.5 s deadline could be enforced by the parent
    def alive():
        try:
            os.kill(child,0); return True
        except ProcessLookupError:
            return False
    try:
        with pytest.raises(ValueError,match='another process holds'):
            with file_lock(lock):
                pass  # the orphan still holds its inherited lock until the watchdog fires
        deadline=time.monotonic()+12
        while alive() and time.monotonic()<deadline:
            time.sleep(.1)
        assert not alive()
        with file_lock(lock):
            pass  # a new runner can acquire the lock after the orphan exits
    finally:
        if alive():
            os.kill(child,signal.SIGKILL)
