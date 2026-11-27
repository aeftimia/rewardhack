#!/usr/bin/env python3
"""Index the recorded settings of past runs and resolve settings for new runs."""

import argparse
import json
from pathlib import Path
from uuid import uuid4


def recorded_runs(results):
    records = {}
    for path in sorted(results.rglob('run-settings.json')):
        relative = path.relative_to(results)
        # Copied trainer checkpoints may also contain settings. The run-level
        # record is one directory below the job's result folder.
        if len(relative.parts) != 3:
            continue
        job, arm, _ = relative.parts
        if '-snapshot' in job or '-records-' in job:
            continue
        run_id = f'{job}/{arm}'
        commit_file = results / job / 'launch-commit.txt'
        settings = json.loads(path.read_text())
        # Store the source digests as SHA-256/path pairs, like a checksum
        # manifest. Historical run settings use filenames as JSON keys.
        source_digests = [[digest, name] for name, digest in settings.pop('code_sha256', {}).items()]
        records[run_id] = {
            'settings': settings,
            'source_digests': source_digests,
            'launch_commit': commit_file.read_text().strip() if commit_file.is_file() else None,
            'source': relative.as_posix(),
        }
    return records


def resolve(defaults, parameters, *, parent=None, inherit=None):
    if parent is None:
        if inherit is not None:
            raise ValueError('inheritance requires a parent run')
        return {**defaults['fresh'], **parameters}
    policy = defaults['continuation']
    if inherit is None:
        inherit = policy['inherit']
    inherited = parent['settings'] if inherit else defaults['fresh']
    # Runtime provenance describes the parent, and must be measured again by
    # the child. A continuation supplies its own total step count.
    settings = {k: v for k, v in inherited.items() if k not in {'code_sha256', 'libraries', 'steps'}}
    return {**settings, **policy['overrides'], **parameters}


def new_run(defaults, parameters, *, parent=None, checkpoint_step=None, inherit=None):
    """Allocate a run ID and record its requested settings and lineage."""
    if (parent is None) != (checkpoint_step is None):
        raise ValueError('a continuation needs both a predecessor and a checkpoint step')
    settings = resolve(defaults, parameters, parent=parent, inherit=inherit)
    predecessor = None
    if parent is not None:
        if type(checkpoint_step) is not int or checkpoint_step < 1:
            raise ValueError('checkpoint step must be a positive integer')
        predecessor = {'run_id': parent['run_id'], 'checkpoint_step': checkpoint_step}
        if inherit is None:
            inherit = defaults['continuation']['inherit']
    return {
        'run_id': uuid4().hex,
        'parent': predecessor,
        'inherit': inherit,
        'parameters': dict(parameters),
        'settings': settings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--lookup')
    args = parser.parse_args()
    records = recorded_runs(args.results)
    if args.lookup is not None:
        if args.lookup not in records:
            parser.error(f'unknown run ID {args.lookup!r}')
        print(json.dumps(records[args.lookup], indent=2))
    else:
        print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
