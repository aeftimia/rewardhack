#!/usr/bin/env python3
"""Adapt Codex patches to the existing no-silent-fallbacks checker, without writes."""
import importlib.util
import json
import os
from pathlib import Path
import sys


def proposed_files(patch, cwd):
    """Yield (destination, before, after). Reject unsupported/ambiguous patches.

    Exact context matching is intentional: never approve contents different from
    what the patch tool will write. A rejected fuzzy patch can be retried with
    exact context. No patch is applied to the working tree by this hook.
    """
    lines = patch.splitlines()
    if not lines or lines[0] != '*** Begin Patch' or lines[-1] != '*** End Patch':
        raise ValueError('expected a complete Codex patch')
    i, seen = 1, set()
    while i < len(lines) - 1:
        header = lines[i]
        i += 1
        kind, name = header.removeprefix('*** ').split(': ', 1)
        if kind not in ('Add File', 'Update File', 'Delete File'):
            raise ValueError(f'unsupported patch header: {header}')
        path = (cwd / name).resolve()
        if path in seen:
            raise ValueError(f'repeated patch target: {path}')
        seen.add(path)
        before = path.read_text() if path.exists() else ''
        if kind == 'Delete File':
            if not path.is_file():
                raise ValueError(f'delete target does not exist: {path}')
            continue
        if kind == 'Add File':
            if path.exists():
                raise ValueError(f'add target already exists: {path}')
            added = []
            while i < len(lines) - 1 and lines[i].startswith('+'):
                added.append(lines[i][1:])
                i += 1
            yield path, before, '\n'.join(added) + ('\n' if added else '')
            continue
        if not path.is_file():
            raise ValueError(f'update target does not exist: {path}')
        destination = path
        if lines[i].startswith('*** Move to: '):
            destination = (cwd / lines[i][13:]).resolve()
            i += 1
            if destination != path:
                if destination.exists() or destination in seen:
                    raise ValueError(f'move destination already exists or is repeated: {destination}')
                seen.add(destination)
        original, output, cursor = before.splitlines(), [], 0
        while i < len(lines) - 1 and not lines[i].startswith('*** '):
            if not (lines[i] == '@@' or lines[i].startswith('@@ ')):
                raise ValueError('expected a patch hunk')
            anchor = lines[i][3:] if lines[i].startswith('@@ ') else None
            i += 1
            if anchor is not None:
                positions = [n for n in range(cursor, len(original)) if original[n] == anchor]
                if len(positions) != 1:
                    raise ValueError('patch anchor is absent or ambiguous; use exact unique context')
                end = positions[0] + 1
                output.extend(original[cursor:end])
                cursor = end
            old, new = [], []
            while i < len(lines) - 1 and not lines[i].startswith(('@@', '*** ')):
                line = lines[i]
                if not line or line[0] not in ' +-':
                    raise ValueError('unsupported hunk line; provide explicit context markers')
                if line[0] in ' -':
                    old.append(line[1:])
                if line[0] in ' +':
                    new.append(line[1:])
                i += 1
            eof = lines[i] == '*** End of File'
            if eof:
                i += 1
            positions = [n for n in range(cursor, len(original) - len(old) + 1)
                         if original[n:n + len(old)] == old
                         and (not eof or n + len(old) == len(original))]
            if not old:
                positions = [len(original)]
            if len(positions) != 1:
                raise ValueError('patch context is absent or ambiguous; use exact unique context')
            start = positions[0]
            output.extend(original[cursor:start])
            output.extend(new)
            cursor = start + len(old)
        output.extend(original[cursor:])
        yield destination, before, '\n'.join(output) + ('\n' if output else '')


def load_checker():
    path = Path.home() / '.claude/hooks/no-silent-fallbacks/no_silent_fallbacks.py'  # Missing checker raises; never skip it.
    spec = importlib.util.spec_from_file_location('shared_no_silent_fallbacks', path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


def check(call, checker):
    if os.environ.get(checker.GUARD):
        return 0
    # This changes only this adapter's subprocess environment, never Claude's settings.
    if 'NO_SILENT_FALLBACKS_JUDGE' not in os.environ:
        os.environ['NO_SILENT_FALLBACKS_JUDGE'] = 'openai'
    if call['tool_name'] == 'Bash':
        return checker.shell_call(call)
    if call['tool_name'] != 'apply_patch':
        return 0
    cwd = Path(call['cwd'])
    for path, before, after in proposed_files(call['tool_input']['command'], cwd):
        if checker.skipped(str(path)):
            continue
        lang = checker.language(str(path), after)
        if lang is None:
            continue
        fresh, note = checker.introduced(before, after, lang, checker.project_settings(str(path)))
        refused, _ = checker.refusals([(str(path), finding) for finding in fresh])
        if refused:
            print(f'no-silent-fallbacks: refused Codex patch to {path}', file=sys.stderr)
            for _, finding in refused:
                print('\n'.join(checker.describe(finding)), file=sys.stderr)
            if note:
                print(note, file=sys.stderr)
            print('\n'.join(checker.ADVICE), file=sys.stderr)
            return 2
    return 0


def main():
    # entrypoints/gate.py reports any exception and exits 2, refusing the call.
    return check(json.load(sys.stdin), load_checker())
