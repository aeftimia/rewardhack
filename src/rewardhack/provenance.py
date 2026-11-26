"""What every scored or raw row says about where it came from, and the loader that turns results into counts per cell.

The prior for the next run's stopping rule is a hierarchical fit across models and λ (owner, 26 Nov; notes section 77):
logit(rate) = model effect + λ effect + per-cell noise, one fit per outcome. Its input is counts per cell. So that results
can be read without joins or guessing, every row a run writes carries `meta` (see `meta`), stated once here.

Rows written before schema 1 have no `meta`. `cells` takes theirs from the `summary.json` beside them, using only what
that summary states (its own fields, or a flag present in its `argv`); a field it does not state is never assumed, and the
file is reported as refused instead.
"""
import json
from collections import defaultdict
from pathlib import Path

from . import jsonl, reporting
from .outcomes import ARMS
from .prompts import VARIANTS

SCHEMA = 1
# Where a row was sampled, and what that says: whether the model was trained under an arm (so the row has an arm and λ),
# and whether the run's steps.jsonl counts its prompt groups. `untrained` is a model never trained here (vLLM or API probes).
STAGES = {
    'training': dict(trained=True, steps=True),
    'before': dict(trained=True, steps=False),
    'after': dict(trained=True, steps=False),
    'probe': dict(trained=True, steps=False),
    'untrained': dict(trained=False, steps=False),
}
FIELDS = ('schema', 'model', 'revision', 'arm', 'penalty', 'stage', 'report_mode', 'taskset', 'variant', 'template', 'run')
# The row files, by the stage their rows belong to.
ROW_FILES = {'training-rollouts.jsonl': 'training', 'before-evaluation.jsonl': 'before', 'after-evaluation.jsonl': 'after',
             'probe-evaluation.jsonl': 'probe', 'rows.rescored.jsonl': 'untrained'}
# The outcomes the prior is fitted to, as each row records them.
OUTCOMES = {'genuine': lambda row: bool(row['genuine']), 'refutation': lambda row: bool(row['verified_refutation']),
            'exploit': lambda row: bool(row['exploit'])}


def meta(*, model, revision, arm, penalty, stage, report_mode, taskset, variant, template, run):
    """The provenance a row carries. `arm` and `penalty` are given exactly when the stage's model is trained under an arm;
    `variant` is the prompt's permission variant (prompts.VARIANTS), a condition of its own."""
    trained = STAGES[stage]['trained']
    if trained != (arm is not None) or trained != (penalty is not None):
        raise ValueError(f'a {stage} row has an arm and a penalty exactly when its model is trained here')
    if trained and arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}')
    if variant not in VARIANTS:
        raise ValueError(f'unknown prompt variant {variant!r}')
    if not (model and revision and run and report_mode and taskset and isinstance(template, dict)):
        raise ValueError('model, revision, run, report mode, task set and the template settings are required')
    return dict(schema=SCHEMA, model=model, revision=revision, arm=arm, penalty=float(penalty) if trained else None,
                stage=stage, report_mode=report_mode, taskset=taskset, variant=variant, template=template, run=run)


def stamp(rows, provenance):
    """`rows` with `provenance` added to each."""
    return [dict(row, meta=provenance) for row in rows]


def flag(argv, name):
    """The value given for `--name` in `argv`, or None when it was not given."""
    for index, word in enumerate(argv):
        if word == f'--{name}':
            return argv[index+1]
        if word.startswith(f'--{name}='):
            return word.split('=', 1)[1]
    return None


def stated(summary, field, name):
    """What `summary` states for one setting: its own field, else the flag in its argv; None when it states neither."""
    if field in summary:
        return summary[field]
    return flag(summary['argv'] if 'argv' in summary else [], name)


def variant_stated(summary):
    """The prompt variant `summary` states: the trainer records it as prompt_variant, the probes as variant."""
    found = [value for value in (stated(summary, 'prompt_variant', 'prompt-variant'), stated(summary, 'variant', 'variant')) if value is not None]
    return found[0] if found else None


def from_summary(path, stage):
    """Provenance for a pre-schema row file, from what its summary.json states; raises ValueError naming what is missing."""
    summary_path = path.parent/'summary.json'
    if not summary_path.is_file():
        raise ValueError('no summary.json beside it')
    summary = json.loads(summary_path.read_text())
    trained = STAGES[stage]['trained']
    mode = stated(summary, 'report_mode', 'report-mode')
    found = dict(model=stated(summary, 'model', 'model'), revision=stated(summary, 'revision', 'revision'),
                 # fallback-ok: a run that states no report mode predates them, and every such run was prose (reporting.HISTORICAL_MODE)
                 report_mode=reporting.HISTORICAL_MODE if mode is None else mode, taskset=stated(summary, 'taskset', 'taskset'),
                 template=summary['template_kwargs'] if 'template_kwargs' in summary else None,
                 variant=variant_stated(summary),
                 arm=stated(summary, 'arm', 'arm') if trained else None, penalty=stated(summary, 'penalty', 'penalty') if trained else None)
    needed = [name for name in found if name not in ('arm', 'penalty') or trained]
    missing = [name for name in needed if found[name] is None]
    if missing:
        raise ValueError(f"its summary.json does not state {', '.join(missing)}")
    return meta(stage=stage, run=str(path.parent), **found)


def count_file(path, stage):
    """(key, counts) of one row file, or None when it has no row of its own to count. Raises when its provenance is unclear
    or a count cannot be read, so that a file refused part way adds nothing."""
    rows = [row for row in jsonl.read(path) if 'copied_from' not in row and 'partial_batch' not in row]
    if not rows:
        return None
    # Every row of one file has the same provenance: stamped rows carry it, older files take it from their summary.
    stamped = [row['meta'] for row in rows if 'meta' in row]
    if stamped and (len(stamped) != len(rows) or any(other != stamped[0] for other in stamped)):
        raise ValueError('its rows do not all carry the same provenance')
    provenance = stamped[0] if stamped else from_summary(path, stage)
    # One entry per run and stage: runs differ in length and batch shape, so the fit treats each as its own unit inside its
    # (model, arm, λ) cell instead of the counts being pooled here.
    key = tuple((name, json.dumps(provenance[name], sort_keys=True)) for name in FIELDS if name != 'schema')
    found = dict(answers=len(rows), **{outcome: sum(read(row) for row in rows) for outcome, read in OUTCOMES.items()})
    if STAGES[stage]['steps']:
        lines = [line for line in jsonl.read(path.parent/'steps.jsonl') if 'copied_from' not in line]
        found.update(groups=sum(line['mixed_groups'] + line['zero_advantage_groups'] for line in lines),
                     mixed_groups=sum(line['mixed_groups'] for line in lines))
    return key, found


def cells(root):
    """Counts per run and stage (with its model, revision, arm, penalty, report mode, task set and template) of every row
    file under `root`, and the files refused with why. A cell counts answers and each of OUTCOMES; training cells also count prompt
    groups and the mixed ones (reward variation) from the run's steps.jsonl. Rows copied from another run, and rows of a
    batch that was cut short, are counted where they belong: in the original run, and in no step."""
    counts = defaultdict(lambda: defaultdict(int))
    refused = []
    for path in sorted(Path(root).rglob('*.jsonl')):
        if path.name not in ROW_FILES:
            continue
        try:
            counted = count_file(path, ROW_FILES[path.name])
        except (KeyError, ValueError, OSError) as error:  # reported to the caller in `refused`, never counted
            refused.append((str(path), f'{type(error).__name__}: {error}'))
            continue
        if counted is None:
            continue  # only rows counted in another run, or none at all
        key, found = counted
        for name, value in found.items():
            counts[key][name] += value
    table = [dict({name: json.loads(value) for name, value in key}, **cell) for key, cell in counts.items()]
    return table, refused
