#!/usr/bin/env python3
"""Index the recorded settings of past runs and resolve settings for new runs."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4
from .gates import file_sha256
from .state_files import atomic_json

CONFIG_FILE = 'run-config.json'


def read(path):
    return json.loads(Path(path).read_text())


def checked_source(directory):
    source = read(directory / 'source.json')
    if (source['config_sha256'] != file_sha256(directory / CONFIG_FILE)
            or source['request_sha256'] != file_sha256(directory / 'request.json')):
        raise ValueError('prepared run configuration was changed')
    return source


def run_id_directory(state, run_id):
    if not re.fullmatch('[a-f0-9]{32}', run_id):
        raise ValueError(f'invalid experiment ID {run_id!r}')
    return Path(state) / run_id

# These changes preserve the shapes and ordering of the saved optimizer and sampler.
CONTINUATION_OVERRIDES = frozenset({'max_completion_length', 'prompt_variant', 'report_mode', 'arm', 'penalty'})


def training_keys():
    from .hf_grpo import RESUME_SETTINGS
    return set(RESUME_SETTINGS) | {'steps', 'profile_backward'}


def training_arguments(parser, argv, config):
    """Resolve the prepared scientific settings through the trainer's own argument parser."""
    permitted = training_keys()
    settings = config['settings']
    unknown = set(settings) - permitted
    if unknown:
        raise ValueError(f'unsupported training settings: {sorted(unknown)}')
    base = parser.parse_args(argv)
    flags = []
    for key, value in settings.items():
        if json.loads(json.dumps(vars(base)[key], default=str)) == value:
            continue
        if key in {'adapter', 'rollout_backend', 'model', 'revision', 'judge', 'judge_url', 'judge_model', 'judge_revision'}:
            raise ValueError(f'{key} conflicts with the provisioned runtime')
        if value is None:
            raise ValueError(f'{key} cannot remove an explicit runtime value')
        option = '--' + key.replace('_', '-')
        if type(value) is bool:
            flags.append(option if value else '--no-' + key.replace('_', '-'))
        else:
            flags.extend([option, json.dumps(value) if isinstance(value, dict) else str(value)])
    args = parser.parse_args([*argv, *flags])
    if config['parent'] is not None:
        step = config['parent']['checkpoint_step']
        source = Path('.tools/experiment-parent') / config['checkpoint_arm']
        args.resume_from = source / 'trainer' / f'checkpoint-{step}'
        state = json.loads((args.resume_from / 'trainer_state.json').read_text())
        if state['global_step'] != step:
            raise ValueError('predecessor checkpoint step differs from the requested step')
        permitted_changes = set(config['parameters']) | set(config['default_overrides'])
        inheritance = config['inherit']
        if inheritance is False:
            permitted_changes.update(CONTINUATION_OVERRIDES)
        elif isinstance(inheritance, dict):
            permitted_changes.update(key for key, enabled in inheritance.items() if not enabled)
        args.continuation_overrides = permitted_changes & CONTINUATION_OVERRIDES
    args.out.mkdir(parents=True, exist_ok=True)
    effective = {key: vars(args)[key] for key in sorted(permitted)}
    atomic_json(args.out / CONFIG_FILE, {**config, 'settings':json.loads(json.dumps(effective, default=str))})
    return args, flags


def result_directory(directory):
    return directory / 'checkout/results' / directory.name


def launch_commit(job):
    path = job / 'launch-commit.txt'
    return path.read_text().strip() if path.is_file() else None


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
        settings = read(path)
        # Store the source digests as SHA-256/path pairs, like a checksum
        # manifest. Historical run settings use filenames as JSON keys.
        source_digests = [[digest, name] for name, digest in settings.pop('code_sha256', {}).items()]
        records[run_id] = {
            'run_id': run_id,
            'settings': settings,
            'settings_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_digests': source_digests,
            'launch_commit': launch_commit(results / job),
            'source': relative.as_posix(),
        }
    return records


def historical_index(roots):
    """Reconcile evidence from multiple checkouts. Conflicting copies need inspection."""
    index = {}
    for root in roots:
        candidates = recorded_runs(root)
        for summary in sorted(root.glob('*/*/summary.json')):
            job, arm, _ = summary.relative_to(root).parts
            key = f'{job}/{arm}'
            if '-snapshot' in job or '-records-' in job or key in candidates:
                continue
            data = read(summary)
            if 'model' not in data:
                continue
            fields = ('model', 'revision', 'template', 'arm', 'penalty', 'prompt_variant',
                      'report_mode', 'max_tokens', 'seed', 'samples', 'argv', 'resumed_from')
            commit = launch_commit(summary.parent.parent)
            candidates[key] = dict(run_id=key, settings={k: data[k] for k in fields if k in data},
                                   source=summary.relative_to(root).as_posix(),
                                   launch_commit=commit)
        known_jobs = {key.split('/')[0] for key in candidates}
        for launch in sorted(root.glob('*/launch-id.txt')):
            job = launch.parent.name
            if job in known_jobs or '-snapshot' in job or '-records-' in job:
                continue
            commit = launch_commit(launch.parent)
            candidates[job] = dict(run_id=job, settings=None, launch_id=launch.read_text().strip(),
                                   source=launch.relative_to(root).as_posix(),
                                   launch_commit=commit)
        for key, record in candidates.items():
            if key in index:
                old = index[key]
                if old['source'] != record['source']:
                    if record['source'].endswith('summary.json'):
                        continue
                elif old['settings'] != record['settings']:
                    raise ValueError(f'conflicting recorded settings for {key}')
                if old['launch_commit'] and record['launch_commit'] and old['launch_commit'] != record['launch_commit']:
                    raise ValueError(f'conflicting launch commits for {key}')
                if record['launch_commit'] is None:
                    record['launch_commit'] = old['launch_commit']
            index[key] = record
    jobs_with_settings = {key.split('/')[0] for key in index if '/' in key}
    return {key: index[key] for key in sorted(index) if key not in jobs_with_settings}


def inheritance_policy(policy, parent, inherit):
    if type(policy['inherit']) is not bool:
        raise ValueError('default inherit must be a boolean')
    if type(inherit) is bool:
        return inherit
    if inherit is not None and not isinstance(inherit, dict):
        raise ValueError('inherit must be a boolean or a per-parameter object')
    fields = policy['inherit_parameters'].copy()
    if inherit is not None:
        if set(inherit) - parent['settings'].keys():
            raise ValueError('requested inheritance for a setting absent from the parent')
        fields.update(inherit)
    if any(type(value) is not bool for value in fields.values()):
        raise ValueError('per-parameter inheritance must be boolean')
    if not fields:
        return policy['inherit']
    return {key: fields[key] if key in fields else policy['inherit'] for key in parent['settings']}


def _resolve(defaults, parameters, parent, inherit):
    if parent is None:
        if inherit is not None:
            raise ValueError('inheritance requires a parent run')
        return {**defaults['fresh'], **parameters}
    policy = defaults['continuation']
    inherited = dict(defaults['fresh'])
    if isinstance(inherit, dict):
        inherited.update({key: value for key, value in parent['settings'].items() if inherit[key]})
    elif inherit:
        inherited.update(parent['settings'])
    # Runtime provenance describes the parent, and must be measured again by
    # the child. A continuation supplies its own total step count.
    settings = {k: v for k, v in inherited.items() if k not in {'code_sha256', 'libraries', 'steps'}}
    return {**settings, **policy['overrides'], **parameters}


def new_run(defaults, parameters, *, parent=None, checkpoint_step=None, inherit=None):
    """Allocate a run ID and record its requested settings and lineage."""
    if (parent is None) != (checkpoint_step is None):
        raise ValueError('a continuation needs both a predecessor and a checkpoint step')
    predecessor = None
    if parent is not None:
        if type(checkpoint_step) is not int or checkpoint_step < 1:
            raise ValueError('checkpoint step must be a positive integer')
        predecessor = {'run_id': parent['run_id'], 'checkpoint_step': checkpoint_step}
        inherit = inheritance_policy(defaults['continuation'], parent, inherit)
    settings = _resolve(defaults, parameters, parent, inherit)
    return {
        'run_id': uuid4().hex,
        'parent': predecessor,
        'inherit': inherit,
        'parameters': dict(parameters),
        'settings': settings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+')
    parser.add_argument('--lookup')
    args = parser.parse_args()
    records = historical_index(args.results)
    if args.lookup is not None:
        if args.lookup not in records:
            parser.error(f'unknown run ID {args.lookup!r}')
        print(json.dumps(records[args.lookup], indent=2))
    else:
        print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
