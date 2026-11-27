"""Scan repository sources and require fresh LLM review on every commit.

Review notes are explanations, not exclusions: the judge reads each finding's
current context. External notes keep the grader's verbatim prompt unchanged.
"""
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .codex_hooks import load_checker
from .json_objects import unique_object

REVIEW_FILE = '.source-review.json'
# One finding per request: multi-case reviews mixed the rationale of adjacent cases.
BATCH_SIZE = 1
JUDGE_WORKERS = 4


def full_scan(root):
    root = Path(root).resolve()
    checker = load_checker()
    if not all(hasattr(checker, name) for name in ('VERDICT_SCHEMA', 'review_cases', 'openai_api_key')):
        raise RuntimeError('update the shared checker: the commit scan requires explicit review obligations')
    if os.environ.get(checker.GUARD):
        raise ValueError('full source scan cannot run with the judge recursion guard set')
    choice = os.environ.get('NO_SILENT_FALLBACKS_JUDGE')
    if choice not in (None, 'openai'):
        raise ValueError('the commit scan requires the OpenAI LLM judge, not a substitute command')
    os.environ['NO_SILENT_FALLBACKS_JUDGE'] = 'openai'
    checker.openai_api_key()  # Validate the selected credential source before scanning or starting workers.
    notes = json.loads((root / REVIEW_FILE).read_text(), object_pairs_hook=unique_object)
    if not isinstance(notes, list):
        raise TypeError(f'{REVIEW_FILE} must contain a list of review notes')
    explanations = {}
    for note in notes:
        if (not isinstance(note, dict) or not {'path', 'rule', 'text', 'reason'} <= set(note)
                or set(note) - {'path', 'rule', 'text', 'reason', 'symbol'}):
            raise ValueError(f'invalid review note in {REVIEW_FILE}')
        if any(not isinstance(value, str) or not value.strip() for value in note.values()):
            raise ValueError(f'empty or non-text review note in {REVIEW_FILE}')
        key = (note['path'], note['rule'], note['text'], note['symbol'] if 'symbol' in note else None)
        if key in explanations:
            raise ValueError(f'duplicate source review note: {key}')
        explanations[key] = note['reason']
    paths = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                           cwd=root, check=True, capture_output=True, timeout=30).stdout.decode().split('\0')
    findings, files, used, evidence_used = [], 0, set(), set()
    settings = checker.project_settings(str(root / '__source_scan__.py'))
    for relative in sorted(set(paths) - {''}):
        path = root / relative
        if checker.skipped(relative):
            continue
        if path.suffix not in ('.py', '.sh', '.bash', '.zsh'):
            with path.open('rb') as handle:
                first = handle.readline(4096)
            if not first.startswith(b'#!') or checker.language(str(path), first.decode()) is None:
                continue
        source = path.read_text()
        language = checker.language(str(path), source)
        if language is None:
            raise ValueError(f'unrecognized source language: {relative}')
        found, error = checker.scan(source, language, checker.project_settings(str(path)))
        if error:
            raise ValueError(f'{relative}: {error}')
        files += 1
        for finding in found:
            selector = (relative, finding['rule'], finding['text'])
            exact = (*selector, finding['symbol'])
            key = exact if exact in explanations else (*selector, None)
            if key in explanations and finding['reason'] is None:
                used.add(key)
                finding['reason'] = explanations[key]
            if 'obligations' in finding:
                evidence_used.add(selector)
            findings.extend(checker.review_cases(relative, finding))
    stale = explanations.keys() - used
    if stale:
        raise ValueError(f'stale source review notes; remove or update them: {sorted(stale, key=repr)}')
    if 'review_evidence' in settings:
        stale_evidence = {(entry['path'], entry['rule'], entry['text']) for entry in settings['review_evidence']} - evidence_used
        if stale_evidence:
            raise ValueError(f'stale review evidence; remove or update it: {sorted(stale_evidence)}')
    marked = [(path, finding) for path, finding in findings if finding['reason'] is not None]
    refused = [(path, dict(finding, verdict=None)) for path, finding in findings if finding['reason'] is None]
    # No cached verdicts: every commit needs a live, valid response, even with zero cases.
    batches = [marked[i:i + BATCH_SIZE] for i in range(0, len(marked), BATCH_SIZE)]
    if not batches:
        batches.append([])
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        for batch, verdicts in zip(batches, pool.map(checker.ask_judge, batches), strict=True):
            for (path, finding), (allowed, reason) in zip(batch, verdicts, strict=True):
                print(f'{path}:{finding["line"]} {finding["rule"]} [{finding["obligation"]}]: '
                      f'{"allow" if allowed else "deny"}: {reason}', flush=True)
                if not allowed:
                    refused.append((path, dict(finding, verdict=reason)))
    for path, finding in refused:
        print('\n'.join(checker.describe(finding, prefix=f'{path}: ')), flush=True)
    print(f'{files} source files scanned; {len(refused)} refused; {len(marked)} obligations reviewed by the live judge', flush=True)
    return 1 if refused else 0
