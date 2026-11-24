"""Frozen bounded-pilot specifications, paired evidence, and compute-aware decisions.

Effects are oriented so larger is better. A pilot with unset targets describes the
curve; it cannot claim success or forecast futility for an undefined hypothesis.
"""
import hashlib
import json
import math
from pathlib import Path

from .budget import LINES
from .json_objects import unique_object
from .reporting import validate_mode

SCHEMA = 'compute-pilot-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {value}')))


def number(value, name, low=None, high=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    value.encode('utf-8')
    return value


def keys(value, required, optional=()):
    if not isinstance(value, dict) or set(required) - value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f'expected keys {sorted(required)}; optional {sorted(optional)}')


def distinct(values, name):
    if not isinstance(values, list) or not values or len({digest(x) for x in values}) != len(values):
        raise ValueError(f'{name} must be a nonempty list without duplicates')
    return values


def validate_spec(spec):
    keys(spec, ('schema', 'id', 'question', 'context', 'seeds', 'checkpoints', 'endpoints', 'model', 'stop', 'stages', 'budget', 'resource'))
    if spec['schema'] != SCHEMA:
        raise ValueError('unknown pilot schema')
    for name in ('id', 'question'):
        text(spec[name], name)
    context = spec['context']
    keys(context, ('population', 'compute_axis', 'report_mode', 'baseline', 'intervention', 'evaluation_protocol'))
    for name, value in context.items():
        text(value, name)
    validate_mode(context['report_mode'])
    seeds = distinct(spec['seeds'], 'seeds')
    if len(seeds) < 3:
        raise ValueError('at least three independent training seeds are required')
    for seed in seeds:
        text(seed, 'seed')
    checkpoints = distinct(spec['checkpoints'], 'checkpoints')
    if not 3 <= len(checkpoints) <= 32:
        raise ValueError('plan 3 to 32 checkpoints')
    for point in checkpoints:
        number(point, 'checkpoint', 1e-12, 1e15)
    if any(right <= left for left, right in zip(checkpoints, checkpoints[1:])):
        raise ValueError('checkpoints must increase strictly')
    endpoints = distinct(spec['endpoints'], 'endpoints')
    if len(endpoints) > 8:
        raise ValueError('at most eight endpoints')
    names = []
    for endpoint in endpoints:
        keys(endpoint, ('name', 'target', 'plateau_delta'))
        names.append(text(endpoint['name'], 'endpoint name'))
        if endpoint['target'] is not None:
            number(endpoint['target'], 'target', -1, 1)
        number(endpoint['plateau_delta'], 'plateau_delta', 1e-6, 1)
    if 'checkpoint' in names:
        raise ValueError('checkpoint is reserved for forecast metadata')
    if len(set(names)) != len(names):
        raise ValueError('endpoint names must be unique')
    model = spec['model']
    keys(model, ('prior_scales', 'noise_floors', 'seed_sds', 'delayed_knees'))
    for name, bounds in [('prior_scales', (.05, 2)), ('noise_floors', (.01, 1)), ('seed_sds', (.1, 1)), ('delayed_knees', (.1, .95))]:
        values = distinct(model[name], name)
        if not 2 <= len(values) <= 4:
            raise ValueError(f'{name} needs two to four sensitivity values')
        for value in values:
            number(value, name, *bounds)
        if len(set(values)) != len(values):
            raise ValueError(f'{name} must contain numerically distinct sensitivities')
    stop = spec['stop']
    keys(stop, ('confidence', 'min_checkpoints', 'min_compute_fraction', 'max_invalid_rate'))
    number(stop['confidence'], 'confidence', .95, .9999)
    if type(stop['min_checkpoints']) is not int or not 3 <= stop['min_checkpoints'] <= len(checkpoints):
        raise ValueError('min_checkpoints must be an integer between 3 and the planned count')
    number(stop['min_compute_fraction'], 'min_compute_fraction', .5, 1)
    number(stop['max_invalid_rate'], 'max_invalid_rate', 0, .05)
    budget = spec['budget']
    keys(budget, ('max_usd', 'max_seconds', 'line', 'ledger'))
    number(budget['max_usd'], 'budget max_usd', 0, 1e6)
    number(budget['max_seconds'], 'budget max_seconds', .001, 1e7)
    if budget['line'] not in LINES or not Path(text(budget['ledger'], 'ledger')).is_absolute():
        raise ValueError('budget needs a known grant line and absolute ledger path')
    keys(spec['resource'], ('kind', 'note'))
    if spec['resource']['kind'] not in ('local_process', 'externally_managed'):
        raise ValueError('unknown resource kind')
    text(spec['resource']['note'], 'resource note')
    if not isinstance(spec['stages'], list) or len(spec['stages']) != len(checkpoints):
        raise ValueError('one bounded stage is required per checkpoint')
    ids, paths = set(), set()
    for stage, point in zip(spec['stages'], checkpoints, strict=True):
        keys(stage, ('id', 'checkpoint', 'argv', 'evidence', 'max_seconds', 'max_usd', 'inputs'))
        sid = text(stage['id'], 'stage id')
        if not all(c.isalnum() or c in '-_' for c in sid) or sid in ids:
            raise ValueError('stage IDs must be unique safe filenames')
        ids.add(sid)
        if type(stage['checkpoint']) not in (int, float) or stage['checkpoint'] != point:
            raise ValueError('stage checkpoint must match its position')
        argv = stage['argv']
        if not isinstance(argv, list) or not argv:
            raise ValueError('stage argv must be a nonempty list')
        for arg in argv:
            text(arg, 'argv item')
            if '\x00' in arg:
                raise ValueError('NUL in argv')
        if not Path(argv[0]).is_absolute():
            raise ValueError('stage executable must be an absolute path')
        if not isinstance(stage['inputs'], dict) or not stage['inputs']:
            raise ValueError('declare static stage inputs with SHA256 hashes')
        for filename, sha in stage['inputs'].items():
            if not Path(text(filename, 'input path')).is_absolute() or not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('stage inputs need absolute paths and SHA256 hashes')
        path = Path(text(stage['evidence'], 'evidence path'))
        if not path.is_absolute() or str(path.resolve()) in paths:
            raise ValueError('evidence paths must be absolute and unique')
        paths.add(str(path.resolve()))
        number(stage['max_seconds'], 'stage max_seconds', .001, budget['max_seconds'])
        number(stage['max_usd'], 'stage max_usd', 0, budget['max_usd'])
    return spec


def load_spec(path):
    return validate_spec(read_json(path))


def validate_rows(spec, records):
    """One validation path for file readers and direct decision API callers."""
    validate_spec(spec)
    if not isinstance(records, list) or len(records) % len(spec['seeds']):
        raise ValueError('checkpoint must contain every planned seed exactly once')
    count = len(records)//len(spec['seeds'])
    if count > len(spec['stages']):
        raise ValueError('too many evidence checkpoints')
    sources = set()
    for index, stage in enumerate(spec['stages'][:count]):
        rows = records[index*len(spec['seeds']):(index+1)*len(spec['seeds'])]
        seen = set()
        for row in rows:
            keys(row, ('spec_hash', 'seed', 'checkpoint', 'baseline_compute', 'intervention_compute', 'effects', 'quality', 'sources'))
            if row['spec_hash'] != digest(spec) or row['seed'] not in spec['seeds'] or row['seed'] in seen:
                raise ValueError('wrong spec hash, unknown seed, or duplicate evidence')
            seen.add(row['seed'])
            for name in ('checkpoint', 'baseline_compute', 'intervention_compute'):
                if number(row[name], name, 0) != stage['checkpoint']:
                    raise ValueError('paired arms must match the declared compute checkpoint')
            keys(row['effects'], [e['name'] for e in spec['endpoints']])
            for value in row['effects'].values():
                keys(value, ('mean', 'se'))
                number(value['mean'], 'effect mean', -1, 1)
                number(value['se'], 'within-seed evaluation se', 0, 2)
            keys(row['quality'], ('valid', 'invalid_rate', 'reason'))
            if type(row['quality']['valid']) is not bool or not isinstance(row['quality']['reason'], str):
                raise ValueError('invalid quality metadata')
            number(row['quality']['invalid_rate'], 'invalid rate', 0, 1)
            hashes = distinct(row['sources'], 'source hashes')
            for sha in hashes:
                if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                    raise ValueError('source must be a SHA256 hex digest')
                if sha in sources:
                    raise ValueError('source evidence reused across seeds or checkpoints')
                sources.add(sha)
    return records


def read_evidence(spec, paths):
    """Read complete ordered checkpoints, bound to frozen stage paths."""
    validate_spec(spec)
    if len(paths) > len(spec['stages']):
        raise ValueError('too many evidence files')
    records = []
    for path, stage in zip(paths, spec['stages']):
        if Path(path).resolve() != Path(stage['evidence']).resolve():
            raise ValueError('evidence path does not match stage')
        rows = [json.loads(line, object_pairs_hook=unique_object) for line in Path(path).read_text().splitlines() if line.strip()]
        if len(rows) != len(spec['seeds']):
            raise ValueError('checkpoint must contain every planned seed exactly once')
        records.extend(rows)
    return validate_rows(spec, records)


def decide(spec, rows):
    """No negative conclusion from an early weak checkpoint; forecast only within cap."""
    validate_rows(spec, rows)
    points = sorted({r['checkpoint'] for r in rows})
    if not rows:
        return dict(stop=False, reason='collect_first_checkpoint', posterior=None, next_checkpoint=spec['checkpoints'][0])
    if points != spec['checkpoints'][:len(points)]:
        raise ValueError('evidence must be a prefix of planned checkpoints')
    for point in points:
        seeds = [r['seed'] for r in rows if r['checkpoint'] == point]
        if sorted(seeds) != sorted(spec['seeds']):
            raise ValueError('each checkpoint needs each seed exactly once')
    if any(not r['quality']['valid'] or r['quality']['invalid_rate'] > spec['stop']['max_invalid_rate'] for r in rows):
        return dict(stop=True, reason='measurement_failure', posterior=None, next_checkpoint=None)
    from .pilot_model import analyze
    posterior = analyze(spec, rows)
    variants = posterior['variants']
    confidence = spec['stop']['confidence']
    mature = len(points) >= spec['stop']['min_checkpoints'] and points[-1] / spec['checkpoints'][-1] >= spec['stop']['min_compute_fraction']
    reason = None
    # A pilot's posterior is conditional on its curve model/cohort, not confirmation.
    if any(v.get('model_misfit', False) for v in variants):
        reason = 'model_misfit'
    elif mature and all(v['success_lower'] is not None and v['success_lower'] > confidence for v in variants):
        reason = 'evidence_for_at_tested_compute'
    elif mature and len(points) < len(spec['checkpoints']) and all(v['future_success_upper'] is not None and v['future_success_upper'] < 1-confidence for v in variants):
        reason = 'predicted_futility_within_budget'
    elif mature and all(v['plateau_lower'] is not None and v['plateau_lower'] > confidence for v in variants):
        reason = 'practical_plateau_within_budget'
    elif len(points) == len(spec['checkpoints']):
        reason = 'compute_cap'
    return dict(stop=reason is not None, reason=reason or 'continue', posterior=posterior,
                next_checkpoint=None if reason else spec['checkpoints'][len(points)],
                mature=mature, hypothesis_defined=all(e['target'] is not None for e in spec['endpoints']),
                recommendation=None if reason else 'Complete the next matched checkpoint; keep seed and evaluation protocol fixed.')
