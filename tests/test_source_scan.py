"""Every obligation needs a valid review of its current evidence."""
import json
from pathlib import Path
import subprocess
import threading

import pytest

from rewardhack import source_scan
from rewardhack.codex_hooks import load_checker


@pytest.fixture
def scan_repo(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path/source_scan.REVIEW_FILE).write_text('[]')
    (tmp_path/'source.py').write_text('value = 1\n')
    checker, receipts = source_scan.review_tools()
    monkeypatch.setattr(checker, 'OPENAI_CREDENTIALS_FILE', tmp_path / 'openai.json')
    monkeypatch.setattr(source_scan, 'review_tools', lambda: (checker, receipts))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.delenv(checker.GUARD, raising=False)
    monkeypatch.delenv('NO_SILENT_FALLBACKS_JUDGE', raising=False)
    monkeypatch.setattr(checker, 'judge', lambda cases: pytest.fail('cached judge must not be used'))
    return tmp_path, checker


def test_clean_scan_still_requires_fresh_judgment(scan_repo, monkeypatch):
    root, checker = scan_repo
    calls = []
    monkeypatch.setattr(checker, 'ask_judge', lambda cases: calls.append(cases) or [])
    assert source_scan.full_scan(root) == 0
    assert calls == [[]]
    assert source_scan.full_scan(root) == 0
    assert calls == [[]]


@pytest.mark.parametrize('setting', ['key', 'guard', 'substitute'])
def test_no_unjudged_commit(scan_repo, monkeypatch, setting):
    root, checker = scan_repo
    if setting == 'key':
        monkeypatch.delenv('OPENAI_API_KEY')
    elif setting == 'guard':
        monkeypatch.setenv(checker.GUARD, '1')
    else:
        monkeypatch.setenv('NO_SILENT_FALLBACKS_JUDGE', 'true')
    monkeypatch.setattr(checker, 'ask_judge', lambda cases: pytest.fail('invalid configuration'))
    with pytest.raises((ValueError, checker.JudgeError)):
        source_scan.full_scan(root)


@pytest.mark.parametrize('reply', ['not JSON', '[]', '[{"verdict":"maybe","reasoning":"x"}]'])
def test_bad_judge_response_cannot_allow_a_finding(scan_repo, monkeypatch, reply):
    root, checker = scan_repo
    (root/'source.py').write_text("value = settings.get('count', 0)  # fallback-ok: test explanation\n")
    monkeypatch.setattr(checker, 'judge_reply', lambda prompt: reply)
    with pytest.raises(checker.JudgeError):
        source_scan.full_scan(root)


def test_judge_transport_failure_propagates(scan_repo, monkeypatch):
    root, checker = scan_repo
    def unavailable(cases):
        raise checker.JudgeError('test transport failure')
    monkeypatch.setattr(checker, 'ask_judge', unavailable)
    with pytest.raises(checker.JudgeError, match='test transport failure'):
        source_scan.full_scan(root)


@pytest.mark.parametrize('allowed', [True, False])
def test_review_notes_are_judged_with_current_context(scan_repo, monkeypatch, allowed):
    root, checker = scan_repo
    source = "value = settings.get('count', 0)\n"
    (root/'source.py').write_text(source)
    found, error = checker.scan(source, 'py', checker.project_settings(str(root/'source.py')))
    assert error is None and len(found) == 1
    (root/source_scan.REVIEW_FILE).write_text(json.dumps([
        dict(path='source.py', rule=found[0]['rule'], text=found[0]['text'], reason='Test explanation')]))
    calls = []
    def review(cases):
        calls.extend(cases)
        return [(allowed, 'test verdict') for _ in cases]
    monkeypatch.setattr(checker, 'ask_judge', review)
    assert source_scan.full_scan(root) == (0 if allowed else 1)
    assert source_scan.full_scan(root) == (0 if allowed else 1)
    assert len(calls) == 1
    assert calls[0][1]['reason'] == 'Test explanation'
    assert source.strip() in calls[0][1]['context']
    (root/'source.py').write_text('value = 1\n')
    with pytest.raises(ValueError, match='stale'):
        source_scan.full_scan(root)


def test_unexplained_finding_and_parse_failure_block_commit(scan_repo, monkeypatch):
    root, checker = scan_repo
    monkeypatch.setattr(checker, 'ask_judge', lambda cases: [])
    (root/'source.py').write_text("value = settings.get('count', 0)\n")
    assert source_scan.full_scan(root) == 1
    (root/'source.py').write_text('def broken(:\n')
    with pytest.raises(ValueError):
        source_scan.full_scan(root)


def test_hook_runs_even_when_no_source_is_staged():
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root/'.pre-commit-config.yaml').read_text())
    hook, = [h for repo in config['repos'] for h in repo['hooks'] if h['id'] == 'no-silent-failures']
    assert hook['always_run'] and not hook['pass_filenames']
    assert hook['entry'] == 'bash ops/gate.sh scan-sources'
    assert hook['stages'] == ['pre-commit']


def test_context_keeps_whole_function_and_local_helper():
    source = 'def helper():\n    return 1\n\ndef target():\n    value = helper()\n    return value\n'
    context, = load_checker().review_contexts(source, 'py', 5)
    assert 'def target' in context and 'return value' in context and 'def helper' in context
    assert '>>    5' in context


def test_shell_context_keeps_complete_file():
    source = '\n'.join(f'line {i}' for i in range(400))
    contexts = ['\n'.join(load_checker().review_contexts(source, 'sh', line)) for line in (80, 81, 160, 161, 399)]
    for line, context in zip((80, 81, 160, 161, 399), contexts, strict=True):
        assert f'>> {line:4}  line {line-1}' in context
    assert 'line 100' in contexts[0] and 'line 100' in contexts[1]


def test_extensionless_shell_source_is_scanned(scan_repo, monkeypatch):
    root, checker = scan_repo
    (root/'launch').write_text('#!/bin/bash\nset -euo pipefail\nfalse || true\n')
    monkeypatch.setattr(checker, 'ask_judge', lambda cases: [])
    assert source_scan.full_scan(root) == 1


def test_one_obligation_denial_blocks_despite_local_approval(scan_repo, monkeypatch):
    root, checker = scan_repo
    (root/'source.py').write_text("value = settings.get('count', 0)  # fallback-ok: test explanation\n")
    monkeypatch.setattr(checker, 'review_cases', lambda path, finding: [
        (path, dict(finding, context='first view', obligation='local')),
        (path, dict(finding, context='second view', obligation='propagation'))])
    def review(cases):
        return [(finding['context'] == 'first view', 'test verdict') for _, finding in cases]
    monkeypatch.setattr(checker, 'ask_judge', review)
    assert source_scan.full_scan(root) == 1


def test_configured_consumer_is_reviewed_and_stale_evidence_refused(scan_repo, monkeypatch):
    root, checker = scan_repo
    (root/'source.py').write_text("value = settings.get('count', 0)  # fallback-ok: explicit failed state\n")
    (root/'consumer.py').write_text('def consume(value):\n    raise ValueError(value)\n')
    found, _ = checker.scan((root/'source.py').read_text(), 'py')
    item, = found
    entry = dict(path='source.py', rule=item['rule'], text=item['text'], obligations=[
        dict(claim='The consumer raises.', sources=[dict(path='consumer.py', symbol='consume')])])
    (root/checker.CONFIG_NAME).write_text(json.dumps(dict(review_evidence=[entry])))
    calls = []
    monkeypatch.setattr(checker, 'ask_judge', lambda cases: calls.extend(cases) or [(True, 'fixture') for _ in cases])
    assert source_scan.full_scan(root) == 0
    assert len(calls) == 2 and any('raise ValueError(value)' in item['context'] for _, item in calls)
    (root/'consumer.py').write_text('def consume(value):\n    return None\n')
    monkeypatch.setattr(checker, 'ask_judge', lambda cases: calls.extend(cases) or [(False, 'lost propagation')])
    assert source_scan.full_scan(root) == 1
    assert len(calls) == 3 and 'return None' in calls[-1][1]['context']
    (root/'source.py').write_text('value = 1\n')
    with pytest.raises(ValueError, match='stale review evidence'):
        source_scan.full_scan(root)


def test_judge_requests_really_run_concurrently(scan_repo, monkeypatch):
    root, checker = scan_repo
    workers = source_scan.JUDGE_WORKERS
    (root/'source.py').write_text(''.join(
        f"value_{i} = settings.get('count', 0)  # fallback-ok: concurrency fixture\n"
        for i in range(workers * 2)))
    barrier, lock = threading.Barrier(workers), threading.Lock()
    active, peak, requests = 0, 0, 0
    def review(cases):
        nonlocal active, peak, requests
        assert len(cases) == 1
        with lock:
            active += 1
            peak = max(peak, active)
            requests += 1
        barrier.wait(timeout=10)  # A serial implementation fails instead of merely running slower.
        with lock:
            active -= 1
        return [(True, 'fixture')]
    monkeypatch.setattr(checker, 'ask_judge', review)
    assert source_scan.full_scan(root) == 0
    assert peak == workers and requests == workers * 2


def test_identical_handlers_use_their_own_symbol_notes(scan_repo, monkeypatch):
    root, checker = scan_repo
    (root/'source.py').write_text(''.join(
        f'def {name}():\n    try:\n        call()\n    except ValueError:\n        return None\n'
        for name in ('first', 'second')))
    (root/source_scan.REVIEW_FILE).write_text(json.dumps([
        dict(path='source.py', rule='py-except-swallow', text='except ValueError:', symbol=name, reason=name)
        for name in ('first', 'second')]))
    calls = []
    monkeypatch.setattr(checker, 'ask_judge', lambda cases: calls.extend(cases) or [(True, 'fixture') for _ in cases])
    assert source_scan.full_scan(root) == 0
    assert {(f['symbol'], f['reason']) for _, f in calls} == {('first', 'first'), ('second', 'second')}
