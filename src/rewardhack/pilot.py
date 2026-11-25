"""Frozen bounded-pilot specifications, paired evidence, and compute-aware decisions.

Effects are oriented so larger is better. A pilot with unset targets describes the
curve; it cannot claim success or forecast futility for an undefined hypothesis.

A specification and an evidence row are checked against strict models (`Spec`, `EvidenceRow`): every
field has its type and range, unknown fields are refused, and bool is never a number. The checked
value stays the caller's dict, so its digest is the digest of what was written.
"""
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, StrictBool, StrictInt, model_validator

from .budget import LINES, finite
from .reporting import validate_mode
from .state_files import Record, load_json, validated

SCHEMA = 'compute-pilot-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return load_json(Path(path).read_text())


# --- Field types ----------------------------------------------------------------------------------------------

def _nonempty(value):
    if not value.strip():
        raise ValueError('must be a nonempty string')
    value.encode('utf-8')  # a lone surrogate cannot be written out
    return value


def _absolute(value):
    if not Path(value).is_absolute():
        raise ValueError('must be an absolute path')
    return value


def _no_nul(value):
    if '\x00' in value:
        raise ValueError('NUL in argv')
    return value


def _safe_id(value):
    if not all(c.isalnum() or c in '-_' for c in value):
        raise ValueError('stage IDs must be safe filenames')
    return value


def _distinct(values):
    if len({digest(x) for x in values}) != len(values):
        raise ValueError('must not repeat a value')
    return values


def _numerically_distinct(values):
    if len(set(values)) != len(values):
        raise ValueError('must contain numerically distinct sensitivities')
    return values


def Between(low, high=None):
    """A finite number (int or float, never bool) in [low, high]."""
    return Annotated[float, Field(ge=low, le=high)]


Text = Annotated[str, AfterValidator(_nonempty)]
AbsolutePath = Annotated[Text, AfterValidator(_absolute)]
Sha256 = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


def Sensitivities(low, high):
    return Annotated[list[Between(low, high)], Field(min_length=2, max_length=4), AfterValidator(_numerically_distinct)]


# --- The specification ----------------------------------------------------------------------------------------

class Context(Record):
    population: Text
    compute_axis: Text
    report_mode: Annotated[Text, AfterValidator(validate_mode)]
    baseline: Text
    intervention: Text
    evaluation_protocol: Text


class Endpoint(Record):
    name: Text
    target: Between(-1, 1) | None  # None: no hypothesis, the pilot describes the curve
    plateau_delta: Between(1e-6, 1)


class Sensitivity(Record):
    prior_scales: Sensitivities(.05, 2)
    noise_floors: Sensitivities(.01, 1)
    seed_sds: Sensitivities(.1, 1)
    delayed_knees: Sensitivities(.1, .95)


class Stop(Record):
    confidence: Between(.95, .9999)
    min_checkpoints: Annotated[StrictInt, Field(ge=3)]  # and at most the planned count (Spec)
    min_compute_fraction: Between(.5, 1)
    max_invalid_rate: Between(0, .05)


class Budget(Record):
    max_usd: Between(0, 1e6)
    max_seconds: Between(.001, 1e7)
    line: Literal[tuple(LINES)]
    ledger: AbsolutePath


class Resource(Record):
    kind: Literal['local_process', 'externally_managed']
    note: Text


class Stage(Record):
    id: Annotated[Text, AfterValidator(_safe_id)]
    checkpoint: Between(1e-12, 1e15)  # equal to the planned checkpoint at its position (Spec)
    argv: Annotated[list[Annotated[Text, AfterValidator(_no_nul)]], Field(min_length=1)]
    evidence: AbsolutePath
    max_seconds: Between(.001)  # within the pilot's budget (Spec)
    max_usd: Between(0)
    inputs: Annotated[dict[AbsolutePath, Sha256], Field(min_length=1)]  # static inputs with SHA256 hashes

    @model_validator(mode='after')
    def absolute_executable(self):
        if not Path(self.argv[0]).is_absolute():
            raise ValueError('stage executable must be an absolute path')
        return self


class Spec(Record):
    schema_: Annotated[Literal[SCHEMA], Field(alias='schema')]
    id: Text
    question: Text
    context: Context
    seeds: Annotated[list[Text], Field(min_length=3), AfterValidator(_distinct)]  # at least three independent training seeds
    checkpoints: Annotated[list[Between(1e-12, 1e15)], Field(min_length=3, max_length=32), AfterValidator(_numerically_distinct)]
    endpoints: Annotated[list[Endpoint], Field(min_length=1, max_length=8)]
    model: Sensitivity
    stop: Stop
    budget: Budget
    resource: Resource
    stages: list[Stage]

    @model_validator(mode='after')
    def plan_is_consistent(self):
        if any(right <= left for left, right in zip(self.checkpoints, self.checkpoints[1:])):
            raise ValueError('checkpoints must increase strictly')
        names = [endpoint.name for endpoint in self.endpoints]
        if 'checkpoint' in names:
            raise ValueError('checkpoint is reserved for forecast metadata')
        if len(set(names)) != len(names):
            raise ValueError('endpoint names must be unique')
        if self.stop.min_checkpoints > len(self.checkpoints):
            raise ValueError('min_checkpoints must be an integer between 3 and the planned count')
        if len(self.stages) != len(self.checkpoints):
            raise ValueError('one bounded stage is required per checkpoint')
        ids, paths = set(), set()
        for stage, point in zip(self.stages, self.checkpoints, strict=True):
            evidence = str(Path(stage.evidence).resolve())
            if stage.id in ids or evidence in paths:
                raise ValueError('stage IDs and evidence paths must be unique')
            ids.add(stage.id)
            paths.add(evidence)
            if stage.checkpoint != point:
                raise ValueError('stage checkpoint must match its position')
            if stage.max_seconds > self.budget.max_seconds or stage.max_usd > self.budget.max_usd:
                raise ValueError(f'stage {stage.id} max_seconds and max_usd must be within the pilot budget')
        return self


def validate_spec(spec):
    validated(Spec, spec, 'invalid pilot specification')
    return spec


def load_spec(path):
    return validate_spec(read_json(path))


# --- Evidence -------------------------------------------------------------------------------------------------

class Effect(Record):
    mean: Between(-1, 1)
    se: Between(0, 2)  # within-seed evaluation standard error


class Quality(Record):
    valid: StrictBool
    invalid_rate: Between(0, 1)
    reason: str


class EvidenceRow(Record):
    """One seed's paired effects at one checkpoint. Which hash, seed, checkpoint and endpoints are allowed depends on the spec (`validate_rows`)."""
    spec_hash: str
    seed: str
    checkpoint: Between(0)
    baseline_compute: Between(0)
    intervention_compute: Between(0)
    effects: dict[str, Effect]
    quality: Quality
    sources: Annotated[list[Sha256], Field(min_length=1), AfterValidator(_distinct)]


def validate_rows(spec, records):
    """One validation path for file readers and direct decision API callers."""
    validate_spec(spec)
    if not isinstance(records, list) or len(records) % len(spec['seeds']):
        raise ValueError('checkpoint must contain every planned seed exactly once')
    count = len(records)//len(spec['seeds'])
    if count > len(spec['stages']):
        raise ValueError('too many evidence checkpoints')
    sources, spec_hash, names = set(), digest(spec), sorted(e['name'] for e in spec['endpoints'])
    for index, stage in enumerate(spec['stages'][:count]):
        seen = set()
        for row in records[index*len(spec['seeds']):(index+1)*len(spec['seeds'])]:
            row = validated(EvidenceRow, row, 'invalid evidence row')
            if row.spec_hash != spec_hash or row.seed not in spec['seeds'] or row.seed in seen:
                raise ValueError('wrong spec hash, unknown seed, or duplicate evidence')
            seen.add(row.seed)
            if not row.checkpoint == row.baseline_compute == row.intervention_compute == stage['checkpoint']:
                raise ValueError('paired arms must match the declared compute checkpoint')
            if sorted(row.effects) != names:
                raise ValueError(f'effects must name exactly the endpoints {names}')
            if sources & set(row.sources):
                raise ValueError('source evidence reused across seeds or checkpoints')
            sources |= set(row.sources)
    return records


# --- Bare checks still used by `pilot_evidence` (its protocol and source records have no model yet) ----------

def number(value, name, low=None, high=None):
    finite(value, name)
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return _nonempty(value)


def keys(value, required, optional=()):
    if not isinstance(value, dict) or set(required) - value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f'expected keys {sorted(required)}; optional {sorted(optional)}')


def read_evidence(spec, paths):
    """Read complete ordered checkpoints, bound to frozen stage paths."""
    validate_spec(spec)
    if len(paths) > len(spec['stages']):
        raise ValueError('too many evidence files')
    records = []
    for path, stage in zip(paths, spec['stages']):
        if Path(path).resolve() != Path(stage['evidence']).resolve():
            raise ValueError('evidence path does not match stage')
        rows = [load_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
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
    # Before maturity nothing is decided from the posterior, so a misfit there is reported and the pilot
    # goes on: one seed improving late is the delayed learning the maturity rule exists to wait for.
    misfit = any(v['model_misfit'] for v in variants)  # `pilot_model.analyze` writes it on every variant; without it a misfit could not stop the pilot
    # A declared skeptical sensitivity must be moved across each decision boundary
    # by evidence. Counts alone do not prevent a strong prior deciding the result.
    boundaries = (
        ('evidence_for_at_tested_compute', 'success_lower', 'prior_success_lower', lambda p: p > confidence),
        ('predicted_futility_within_budget', 'future_success_upper', 'prior_future_success_upper', lambda p: p < 1-confidence),
        ('practical_plateau_within_budget', 'plateau_lower', 'prior_plateau_lower', lambda p: p > confidence),
    )
    crossed, prior_driven = [], []
    for boundary, posterior_key, prior_key, test in boundaries:
        if posterior_key != 'success_lower' and len(points) == len(spec['checkpoints']):
            continue  # no future horizon exists on which to forecast futility or plateau
        if all(v[posterior_key] is not None and test(v[posterior_key]) for v in variants):
            if all(v[prior_key] is not None and test(v[prior_key]) for v in variants):
                prior_driven.append(boundary)
            elif any(v[prior_key] is not None and not test(v[prior_key]) for v in variants):
                crossed.append(boundary)
    if mature and misfit:
        reason = 'model_misfit'
    elif mature and crossed:
        reason = crossed[0]
    elif len(points) == len(spec['checkpoints']):
        reason = 'compute_cap'
    return dict(stop=reason is not None, reason=reason or 'continue', posterior=posterior,
                next_checkpoint=None if reason else spec['checkpoints'][len(points)],
                mature=mature, model_misfit=misfit, prior_driven_boundaries=prior_driven, hypothesis_defined=all(e['target'] is not None for e in spec['endpoints']),
                recommendation=None if reason else 'Complete the next matched checkpoint; keep seed and evaluation protocol fixed.')
