"""Bounded serial execution of a frozen compute pilot; no resource provisioning.

Each stage is a complete matched checkpoint block. Its declared worst dollar cost
is reserved before starting and is never automatically refunded. Existing child
budget caps still apply; the wrapper bounds subprocess time, not provider billing.
"""
import contextlib
import hashlib
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from . import budget, pilot
from .state_files import atomic_json as _write, file_lock as _lock

_read = pilot.read_json
# Signals that end the runner while a stage is live. SIGKILL cannot be caught: after it, the stage's
# process group survives and has to be found and killed by hand before anything is resumed.
_STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGINT)


def _usd(value):
    """Dollar sums at the ledger's precision, so stages that fill a cap exactly are not refused by binary rounding."""
    return round(value, 6)


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sources():
    root = Path(__file__).parent
    return {path.name: _hash_file(path) for path in sorted(root.glob('*.py'))}


def _reserve(spec, out, stage, spec_hash):
    """Share conservative reservations across all pilot outputs using this ledger."""
    ledger = Path(spec['budget']['ledger']).resolve()
    if not ledger.is_file():
        raise ValueError('the grant ledger must exist before execution')
    registry = budget.reservation_path(ledger)
    key = pilot.digest({'out': str(out), 'spec_hash': spec_hash, 'stage_id': stage['id']})
    with _lock(registry.with_suffix(registry.suffix + '.lock'), wait=budget.REGISTRY_WAIT):
        reservations = budget.read_reservations(ledger)
        keys = {record['key'] for record in reservations}
        if key in keys:
            raise ValueError('stage already reserved without a completed receipt; manual reconciliation required')
        rows = budget.read_ledger(ledger, include_reservations=False) + reservations
        _, problems = budget.check(rows, spec['budget']['line'], 0, 0, stage['max_usd'], stage['max_usd'])
        if problems:
            return None, problems
        reservations.append(dict(key=key, line=spec['budget']['line'], total_usd=stage['max_usd'],
                                 status='pilot worst-case reservation'))
        _write(registry, reservations, exclusive=False)
    return key, []


def _terminate(process):
    # A child may exit while its workers remain in the original process group.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    # An interrupt that landed inside Popen.wait can leave its internal lock held, and an untimed wait
    # would then block for ever on a child that is already dead: bound it and reap directly.
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(process.pid, 0)


def _execute(stage, log, timeout=None, reservation_key=None):
    """Run argv directly, terminating every process in its session on any catchable exit.

    The stage runs in its own session, so a closed terminal would not reach it: hangup and quit are
    handled like termination, unless the caller ignores them (nohup), in which case runner and stage
    both go on. The child receives its reservation key so the budget guard can
    atomically claim part of the stage allowance before each launch.
    """
    started = time.monotonic()
    with Path(log).open('xb') as sink:
        # A stop raises only while the child is known and not yet being cleaned up. While it is being
        # started (forked, `process` still unset) or killed, the signal is recorded and raised afterwards:
        # raising there would leave the stage running with nobody holding its process group.
        stopping, armed, raised = [], [], []
        def stop():
            raised.append(True)
            return KeyboardInterrupt(f'pilot received {signal.Signals(stopping[0]).name}')
        def interrupted(number, frame):
            if stopping:  # one stop is enough: a second signal must not interrupt the cleanup
                return
            stopping.append(number)
            if armed:
                raise stop()
        # A signal the caller chose to ignore (nohup) stays ignored: the run was meant to outlive its terminal.
        previous = {number: signal.signal(number, interrupted) for number in _STOP_SIGNALS
                    if signal.getsignal(number) not in (signal.SIG_IGN, None)}  # None: installed outside Python, not restorable
        process = None
        try:
            environment = dict(os.environ)
            environment.pop(budget.RESERVATION_ENV, None)
            if reservation_key is not None:
                environment[budget.RESERVATION_ENV] = reservation_key
            process = subprocess.Popen(stage['argv'], stdout=sink, stderr=subprocess.STDOUT, env=environment,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            armed.append(True)
            if stopping and not raised:
                raise stop()
            code = process.wait(timeout=stage['max_seconds'] if timeout is None else timeout)
            return dict(exit_code=code, timed_out=False, elapsed_seconds=time.monotonic()-started)
        except subprocess.TimeoutExpired:
            return dict(exit_code=None, timed_out=True, elapsed_seconds=time.monotonic()-started)
        finally:
            try:
                armed.clear()
            finally:
                if process is not None:
                    _terminate(process)
                for number, handler in previous.items():
                    signal.signal(number, handler)
            if stopping and not raised:  # arrived during the cleanup: the pilot was still told to stop
                raise stop()


class TimeCap(TimeoutError):
    """Pure evidence analysis exhausted its bounded wall-time allowance."""


def _analyze(spec, paths, seconds):
    """Read the evidence and decide, in a forked child that is killed at the deadline.

    The parent only waits, so the deadline cannot fire inside a receipt write, be swallowed by code
    that catches `OSError`, or arrive late and kill the runner, as an alarm signal in this process
    could. The child shares nothing it can change; its result returns over a pipe.
    A fork, not a fresh interpreter: the child must see exactly the frozen code and spec the parent holds.
    The runner has no other threads, which is what makes forking it safe.
    """
    if seconds <= 0:
        raise TimeCap('pilot analysis exhausted the remaining wall-time budget')
    context = multiprocessing.get_context('fork')
    receiver, sender = context.Pipe(duplex=False)
    def work():
        try:
            message = ('ok', pilot.decide(spec, pilot.read_evidence(spec, paths)))
        except (ValueError, OSError) as error:
            message = ('invalid', str(error))
        except BaseException as error:
            message = ('crash', f'{type(error).__name__}: {error}')
        sender.send(message)
    child = context.Process(target=work, daemon=True)
    child.start()
    sender.close()
    try:
        if not receiver.poll(seconds):
            raise TimeCap('pilot evidence analysis reached its wall-time cap')
        try:
            kind, value = receiver.recv()
        except EOFError:
            raise RuntimeError('pilot analysis process ended without a result') from None
    finally:
        if child.is_alive():
            child.kill()
        child.join()
        receiver.close()
    if kind == 'ok':
        return value
    if kind == 'invalid':
        raise ValueError(value)
    raise RuntimeError(value)


def _verify_inputs(spec, spec_path):
    if pilot.load_spec(spec_path) != spec:
        raise ValueError('pilot specification changed during execution')
    for stage in spec['stages']:
        declared = {str(Path(path).resolve()) for path in stage['inputs']}
        for arg in stage['argv']:
            if '\n' not in arg and Path(arg).suffix in ('.py', '.sh'):
                if Path(arg).resolve() != Path(spec_path).resolve() and str(Path(arg).resolve()) not in declared:
                    raise ValueError('script arguments must be declared in stage inputs')
        for path, expected in stage['inputs'].items():
            if _hash_file(path) != expected:
                raise ValueError('static stage input changed')


def plan(spec_path):
    """Read-only plan: does not reserve, create directories, or execute anything."""
    spec = pilot.load_spec(spec_path)
    _verify_inputs(spec, spec_path)
    return dict(status='planned', spec_hash=pilot.digest(spec), id=spec['id'], budget=spec['budget'],
                resource=spec['resource'], stages=spec['stages'],
                planned_worst_usd=_usd(sum(s['max_usd'] for s in spec['stages'])),
                planned_max_seconds=sum(s['max_seconds'] for s in spec['stages']),
                billing_scope='Subprocesses only. External resources require their own teardown and billing cap.')


def run(spec_path, out, *, execute=False):
    """Execute only with explicit opt-in; crash recovery never re-runs uncertain work."""
    if not execute:
        return plan(spec_path)
    if threading.current_thread() is not threading.main_thread():
        # A stage's stop signals can only be handled there; refuse before any reservation or launch.
        raise ValueError('a pilot can only be executed from the main thread')
    spec = pilot.load_spec(spec_path)
    _verify_inputs(spec, spec_path)
    spec_hash = pilot.digest(spec)
    out = Path(out).resolve()
    sources = _sources()
    invocation_wall, invocation_mono = time.time(), time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    with _lock(out/'pilot.lock'):
        freeze_path = out/'freeze.json'
        if freeze_path.exists():
            frozen = _read(freeze_path)
            if (not isinstance(frozen, dict) or frozen.get('spec') != spec
                    or frozen.get('spec_hash') != spec_hash or frozen.get('sources') != sources
                    or type(frozen.get('started_at')) not in (int, float)
                    or not 0 < frozen['started_at'] <= time.time()):
                raise ValueError('pilot specification, source code, or freeze record changed; refusing resume')
        else:
            if any(p.name != 'pilot.lock' for p in out.iterdir()):
                raise ValueError('pilot output must be empty or contain a valid freeze record')
            frozen = dict(spec=spec, spec_hash=spec_hash, sources=sources, started_at=time.time())
            _write(freeze_path, frozen)
        (out/'stages').mkdir(exist_ok=True)
        terminal_path = out/'decision.json'
        def elapsed_time():
            return max(time.time()-frozen['started_at'],
                       invocation_wall-frozen['started_at'] + time.monotonic()-invocation_mono)
        # A corrupt/gapped receipt prefix must be found before launching anything.
        started_indices, result_indices, log_indices = set(), set(), set()
        for path in (out/'stages').iterdir():
            parts = path.name.split('.')
            if (len(parts) != 3 or parts[1:] not in (['started', 'json'], ['result', 'json'])):
                if len(parts) == 2 and parts[1] == 'log' and parts[0].isdigit():
                    if parts[0] != f'{int(parts[0]):04d}':
                        raise ValueError('invalid stage log index')
                    log_indices.add(int(parts[0]))
                    continue
                raise ValueError('unknown file in pilot stage receipts')
            if not parts[0].isdigit() or parts[0] != f'{int(parts[0]):04d}':
                raise ValueError('invalid stage receipt index')
            (started_indices if parts[1] == 'started' else result_indices).add(int(parts[0]))
        if (started_indices != set(range(len(started_indices)))
                or len(started_indices) > len(spec['stages'])
                or not log_indices <= started_indices
                or not result_indices <= started_indices
                or result_indices != set(range(len(result_indices)))):
            raise ValueError('stage receipts must form a contiguous prefix')

        def finish(reason, **details):
            result = dict(stop=True, reason=reason, spec_hash=spec_hash,
                          reserved_usd=charged, elapsed_seconds=elapsed_time(),
                          resource=spec['resource'], **details)
            _write(terminal_path, result)
            return result

        charged, paths, evidence_hashes = 0.0, [], {}
        reservations = {r['key']: r for r in budget.read_reservations(spec['budget']['ledger'])}
        failed_result = None
        # Replay receipts in order; never infer completion from raw output alone.
        completed = 0
        for index, stage in enumerate(spec['stages']):
            prefix = out/'stages'/f'{index:04d}'
            started_path, result_path = prefix.with_suffix('.started.json'), prefix.with_suffix('.result.json')
            if not started_path.exists():
                if result_path.exists():
                    raise ValueError('stage result exists without a reservation receipt')
                break
            receipt = _read(started_path)
            if (receipt.get('stage') != stage or receipt.get('spec_hash') != spec_hash
                    or not isinstance(receipt.get('reservation_key'), str)):
                raise ValueError('invalid stage reservation receipt')
            reservation = reservations.get(receipt['reservation_key'])
            expected_key = pilot.digest({'out': str(out), 'spec_hash': spec_hash, 'stage_id': stage['id']})
            if (receipt['reservation_key'] != expected_key or reservation is None
                    or reservation['total_usd'] != stage['max_usd'] or reservation['line'] != spec['budget']['line']):
                raise ValueError('stage reservation is missing or changed in shared registry')
            charged = _usd(charged + stage['max_usd'])
            if charged > _usd(spec['budget']['max_usd']):
                raise ValueError('stage receipts exceed the frozen pilot budget')
            if not result_path.exists():
                raise ValueError('stage was started but has no completion receipt; refusing automatic retry '
                                 '(if the runner was killed, the stage\'s processes may still be running)')
            result = _read(result_path)
            if result.get('spec_hash') != spec_hash or result.get('stage_id') != stage['id']:
                raise ValueError('invalid stage completion receipt')
            if result.get('status') != 'completed':
                failed_result = result
                if index != len(started_indices)-1:
                    raise ValueError('stages were started after a failed stage')
                if terminal_path.exists():
                    break
                return finish(result.get('failure_reason', 'stage_failure'), stage_id=stage['id'], stage_result=result)
            evidence = Path(stage['evidence'])
            if not evidence.is_file() or _hash_file(evidence) != result.get('evidence_sha256'):
                raise ValueError('completed stage evidence changed or disappeared')
            paths.append(evidence)
            evidence_hashes[evidence] = result['evidence_sha256']
            completed += 1
        if terminal_path.exists():
            result = _read(terminal_path)
            if (result.get('stop') is not True or result.get('spec_hash') != spec_hash
                    or result.get('reserved_usd') != charged or result.get('resource') != spec['resource']):
                raise ValueError('invalid terminal decision')
            if failed_result is not None:
                if result.get('reason') != failed_result.get('failure_reason', 'stage_failure'):
                    raise ValueError('terminal decision disagrees with failed stage')
            elif 'decision' in result:
                try:
                    replayed = _analyze(spec, paths, min(spec['budget']['max_seconds'], 30.0))
                except TimeCap as error:
                    raise ValueError('terminal decision audit exceeded its bounded allowance; saved decision unchanged') from error
                if result['decision'] != replayed or result['reason'] != replayed['reason'] or not replayed['stop']:
                    raise ValueError('terminal decision disagrees with immutable evidence')
            elif result.get('reason') not in ('budget_cap', 'time_cap', 'grant_budget_cap'):
                raise ValueError('terminal decision has no supporting evidence')
            return result
        if paths:
            # A stop already recorded in the last stage's receipt is re-derived, not newly decided: like the
            # audit of a terminal decision it gets its own allowance, so a late resume cannot replace it with a time cap.
            recorded_stop = _read(out/'stages'/f'{completed-1:04d}.result.json').get('decision', {}).get('stop') is True
            allowance = spec['budget']['max_seconds']-elapsed_time()
            try:
                decision = _analyze(spec, paths, max(allowance, min(spec['budget']['max_seconds'], 30.0)) if recorded_stop else allowance)
            except TimeCap:
                return finish('time_cap', phase='evidence_analysis')
            if decision['stop']:
                return finish(decision['reason'], decision=decision)
        for index, stage in enumerate(spec['stages'][completed:], completed):
            if _sources() != sources:
                raise ValueError('pilot source code changed during execution')
            _verify_inputs(spec, spec_path)
            elapsed = elapsed_time()
            if _usd(charged + stage['max_usd']) > _usd(spec['budget']['max_usd']):
                return finish('budget_cap', next_stage=stage['id'])
            if elapsed + stage['max_seconds'] > spec['budget']['max_seconds']:
                return finish('time_cap', next_stage=stage['id'])
            if Path(stage['evidence']).exists():
                raise ValueError('unstarted stage evidence already exists; refusing stale evidence')
            key, problems = _reserve(spec, out, stage, spec_hash)
            if problems:
                return finish('grant_budget_cap', problems=problems, next_stage=stage['id'])
            prefix = out/'stages'/f'{index:04d}'
            _write(prefix.with_suffix('.started.json'), dict(stage=stage, spec_hash=spec_hash, reservation_key=key))
            charged = _usd(charged + stage['max_usd'])
            try:
                _verify_inputs(spec, spec_path)
                if _sources() != sources:
                    raise ValueError('pilot source code changed before dispatch')
                remaining = spec['budget']['max_seconds']-elapsed_time()
                if remaining <= 0:
                    result = dict(exit_code=None, timed_out=True, elapsed_seconds=0.0)
                else:
                    result = _execute(stage, prefix.with_suffix('.log'), timeout=min(stage['max_seconds'], remaining),
                                      reservation_key=key)
                if result['timed_out'] or result['exit_code'] != 0:
                    result.update(status='failed')
                else:
                    if _sources() != sources:
                        raise ValueError('pilot source code changed during stage execution')
                    _verify_inputs(spec, spec_path)
                    try:
                        for evidence, expected_hash in evidence_hashes.items():
                            if _hash_file(evidence) != expected_hash:
                                raise ValueError('completed stage evidence changed during execution')
                        candidate_paths = paths + [Path(stage['evidence'])]
                        decision = _analyze(spec, candidate_paths, spec['budget']['max_seconds']-elapsed_time())
                        result.update(status='completed', evidence_sha256=_hash_file(stage['evidence']), decision=decision)
                    except TimeCap as error:
                        result.update(status='failed', failure_reason='time_cap', error=str(error))
                    except (ValueError, OSError) as error:
                        result.update(status='failed', failure_reason='measurement_failure', error=str(error)[:1000])
            except BaseException as error:
                result = dict(status='failed', error=f'{type(error).__name__}: {str(error)[:1000]}')
                _write(prefix.with_suffix('.result.json'), dict(result, spec_hash=spec_hash, stage_id=stage['id']))
                finish('stage_failure', stage_id=stage['id'], stage_result=result)
                raise
            _write(prefix.with_suffix('.result.json'), dict(result, spec_hash=spec_hash, stage_id=stage['id']))
            if result['status'] != 'completed':
                return finish(result.get('failure_reason', 'stage_failure'), stage_id=stage['id'], stage_result=result)
            paths.append(Path(stage['evidence']))
            evidence_hashes[paths[-1]] = result['evidence_sha256']
            if decision['stop']:
                return finish(decision['reason'], decision=decision)
        return finish('pilot_complete', decision=decision)
