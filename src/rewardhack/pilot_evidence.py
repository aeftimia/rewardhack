"""Pair scored outputs by frozen task cohort; preserve failures and cluster by group."""
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from . import jsonl
from .json_objects import unique_object
from .pilot import digest, keys, number, text, validate_spec


def group_effect(means):
    """Posterior of the mean effect from per-group means, as the `{mean, se}` the pilot model reads.

    With the groups exchangeable and the usual reference prior on their mean and spread, the posterior of the mean is a
    Student t with G-1 degrees of freedom around the groups' mean, scaled by their standard error (scipy.stats.t). With
    two or three groups its tails are far heavier than a normal's. The pilot model takes a normal likelihood, so `se` is
    the normal scale whose central 95% interval matches the t's: 6.5 times the standard error with two groups, 2.2 with
    three, 1.6 with four. A normal standard error would claim those intervals were that much narrower.
    """
    means = np.asarray(means, dtype=float)
    if means.size < 2:
        raise ValueError('an effect needs at least two independent task groups')
    scale = means.std(ddof=1)/np.sqrt(means.size)
    half = stats.t.ppf(0.975, df=means.size-1)*scale  # scale 0 (every group equal) gives a point, which scipy's scale= refuses
    return dict(mean=float(means.mean()), se=float(half/stats.norm.ppf(0.975)))


def paired_observation(spec, protocol, seed, checkpoint, baseline, intervention,
                       baseline_compute, intervention_compute):
    """One seed's matched checkpoint, conditional on the frozen evaluation cohort.

    `compute` is a caller-measured exposure on the declared axis, not inferred
    from training steps. Each file must contain every planned task/sample once.
    Uncertainty clusters task twins and repeated samples within task groups (`group_effect`).
    """
    validate_spec(spec)
    keys(protocol, ('schema', 'tasks', 'samples_per_task', 'endpoints', 'baseline_arm', 'intervention_arm'))
    if protocol['schema'] != 'paired-evaluation-v1' or digest(protocol) != spec['context']['evaluation_protocol']:
        raise ValueError('paired evaluation protocol does not match frozen context')
    if seed not in spec['seeds'] or checkpoint not in spec['checkpoints']:
        raise ValueError('unplanned seed or checkpoint')
    if number(baseline_compute, 'baseline compute', 0) != checkpoint or number(intervention_compute, 'intervention compute', 0) != checkpoint:
        raise ValueError('baseline and intervention need matched measured exposure')
    n = protocol['samples_per_task']
    if type(n) is not int or n < 1:
        raise ValueError('samples_per_task must be a positive integer')
    tasks = {}
    for task in protocol['tasks']:
        keys(task, ('id', 'group', 'true'))
        tid = text(task['id'], 'task ID')
        text(task['group'], 'group')
        if tid in tasks or type(task['true']) is not bool:
            raise ValueError('duplicate task or invalid truth value')
        tasks[tid] = task
    if not tasks:
        raise ValueError('empty evaluation cohort')
    endpoints = protocol['endpoints']
    keys(endpoints, [e['name'] for e in spec['endpoints']])
    for definition in endpoints.values():
        keys(definition, ('metric', 'truth', 'direction'))
        if definition['metric'] not in ('exploit', 'genuine', 'concealed_exploit') or definition['truth'] not in ('all', 'true', 'false') or definition['direction'] not in ('increase', 'decrease'):
            raise ValueError('unknown paired endpoint definition')
    paired, hashes, invalid, undecided = [], [], 0, False
    for path, arm, condition in [(baseline, protocol['baseline_arm'], spec['context']['baseline']),
                                  (intervention, protocol['intervention_arm'], spec['context']['intervention'])]:
        blob = Path(path).read_bytes()
        hashes.append(hashlib.sha256(blob).hexdigest())
        rows = list(jsonl.loads(blob.decode(), path, object_pairs_hook=unique_object))
        mapping = {}
        checkpoint_hashes = set()
        for row in rows:
            source = row.get('pilot_source')
            keys(source, ('spec_hash', 'seed', 'checkpoint', 'compute', 'condition', 'checkpoint_sha256'))
            if (source['spec_hash'] != digest(spec) or source['seed'] != seed
                    or number(source['checkpoint'], 'source checkpoint') != checkpoint
                    or number(source['compute'], 'source measured compute') != checkpoint
                    or source['condition'] != condition):
                raise ValueError('source run provenance differs from frozen paired checkpoint')
            sha = source['checkpoint_sha256']
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('source checkpoint must have a SHA256 artifact identity')
            checkpoint_hashes.add(sha)
            if len(checkpoint_hashes) != 1:
                raise ValueError('a source file mixes checkpoint artifacts')
            tid = row.get('task_id')
            if tid not in tasks:
                raise ValueError('unplanned task or transport error in evaluation')
            task = tasks[tid]
            # Producers add sample_index to avoid treating a duplicated row as another draw.
            sample = row.get('sample_index')
            if type(sample) is not int or not 0 <= sample < n or (tid, sample) in mapping:
                raise ValueError('missing, repeated or unplanned sample_index')
            if row.get('arm') != arm or row.get('report_mode') != spec['context']['report_mode']:
                raise ValueError('arm or report format mismatch')
            if row.get('group') != task['group'] or type(row.get('true')) is not bool or row['true'] != task['true']:
                raise ValueError('task cohort mismatch')
            for flag in ('genuine', 'exploit', 'format_valid', 'verifier_inconclusive'):
                if type(row.get(flag)) is not bool:
                    raise ValueError(f'missing Boolean outcome {flag}')
            if type(row.get('truncated')) is not bool or row.get('termination_schema') != 'cutoff-v1':
                raise ValueError('evaluation needs current explicit cutoff metadata')
            if (row['genuine'] and (row['exploit'] or not row['true'])
                    or not row['format_valid'] and (row['genuine'] or row['exploit'])
                    or row['verifier_inconclusive'] and row['exploit']):
                raise ValueError('inconsistent scored outcome flags')
            if row.get('partial_batch') or row.get('scored') is False:
                raise ValueError('partial/unscored evaluation cannot enter likelihood')
            if any(e['metric'] == 'concealed_exploit' for e in endpoints.values()) and row.get('claim') not in ('OK', 'ERROR'):
                raise ValueError('concealment endpoint requires a parsed report claim')
            invalid += bool(row['truncated'] or not row['format_valid'] or row['verifier_inconclusive'])
            undecided |= row['verifier_inconclusive']
            mapping[(tid, sample)] = row
        if len(mapping) != len(tasks)*n:
            raise ValueError('incomplete paired cohort; no complete-case dropping')
        paired.append(mapping)
    if len(set(hashes)) != 2:
        raise ValueError('baseline and intervention cannot reuse the same evidence file')
    effects = {}
    for name, definition in endpoints.items():
        grouped = defaultdict(list)
        for key, before in paired[0].items():
            after = paired[1][key]
            if definition['truth'] != 'all' and before['true'] != (definition['truth'] == 'true'):
                continue
            # Called only within this iteration, so the loop's current `definition` is the intended one.
            def value(row):
                return int(row['exploit'] and row['claim'] == 'OK') if definition['metric'] == 'concealed_exploit' else int(row[definition['metric']])  # noqa: B023  see above
            difference = value(after)-value(before)
            grouped[before['group']].append(difference if definition['direction'] == 'increase' else -difference)
        if len(grouped) < 2:
            raise ValueError('each endpoint requires at least two independent task groups')
        effects[name] = group_effect([sum(values)/len(values) for values in grouped.values()])
    return dict(spec_hash=digest(spec), seed=seed, checkpoint=checkpoint,
                baseline_compute=baseline_compute, intervention_compute=intervention_compute,
                effects=effects, sources=hashes,
                quality=dict(valid=invalid == 0, invalid_rate=invalid/(2*len(tasks)*n),
                             reason='undecided verifier' if undecided else 'missing or cut-off outcome' if invalid else 'complete paired cohort'))
