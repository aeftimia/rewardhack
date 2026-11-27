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
    if not hasattr(checker, 'VERDICT_SCHEMA') or not hasattr(checker, 'review_contexts'):
        raise RuntimeError('update the shared checker: the commit scan requires its strict OpenAI verdict schema')
    if os.environ.get(checker.GUARD):
        raise ValueError('full source scan cannot run with the judge recursion guard set')
    choice = os.environ.get('NO_SILENT_FALLBACKS_JUDGE')
    if choice not in (None, 'openai'):
        raise ValueError('the commit scan requires the OpenAI LLM judge, not a substitute command')
    os.environ['NO_SILENT_FALLBACKS_JUDGE'] = 'openai'
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('full source scan needs OPENAI_API_KEY; commits without an LLM judgment are refused')
    notes = json.loads((root / REVIEW_FILE).read_text(), object_pairs_hook=unique_object)
    if not isinstance(notes, list):
        raise TypeError(f'{REVIEW_FILE} must contain a list of review notes')
    explanations = {}
    for note in notes:
        if not isinstance(note, dict) or set(note) != {'path', 'rule', 'text', 'reason'}:
            raise ValueError(f'invalid review note in {REVIEW_FILE}')
        if any(not isinstance(value, str) or not value.strip() for value in note.values()):
            raise ValueError(f'empty or non-text review note in {REVIEW_FILE}')
        key = (note['path'], note['rule'], note['text'])
        if key in explanations:
            raise ValueError(f'duplicate source review note: {key}')
        explanations[key] = note['reason']
    paths = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                           cwd=root, check=True, capture_output=True, timeout=30).stdout.decode().split('\0')
    findings, files, used = [], 0, set()
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
            key = (relative, finding['rule'], finding['text'])
            if key in explanations and finding['reason'] is None:
                used.add(key)
                finding['reason'] = explanations[key]
            for context in checker.review_contexts(source, language, finding['line']):
                findings.append((relative, dict(finding, context=context)))
    stale = explanations.keys() - used
    if stale:
        raise ValueError(f'stale source review notes; remove or update them: {sorted(stale)}')
    marked = [(path, finding) for path, finding in findings if finding['reason'] is not None]
    refused = [(path, dict(finding, verdict=None)) for path, finding in findings if finding['reason'] is None]
    # No cached verdicts: every commit needs a live, valid response, even with zero cases.
    batches = [marked[i:i + BATCH_SIZE] for i in range(0, len(marked), BATCH_SIZE)]
    if not batches:
        batches.append([])
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        for batch, verdicts in zip(batches, pool.map(checker.ask_judge, batches), strict=True):
            for (path, finding), (allowed, reason) in zip(batch, verdicts, strict=True):
                print(f'{path}:{finding["line"]} {finding["rule"]}: {"allow" if allowed else "deny"}: {reason}', flush=True)
                if not allowed:
                    refused.append((path, dict(finding, verdict=reason)))
    for path, finding in refused:
        print('\n'.join(checker.describe(finding, prefix=f'{path}: ')), flush=True)
    print(f'{files} source files scanned; {len(refused)} refused; {len(marked)} findings reviewed by the live judge', flush=True)
    return 1 if refused else 0
