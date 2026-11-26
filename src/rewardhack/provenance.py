"""What every scored or raw row says about where it came from, and the loader that turns results into counts per cell.

The prior for the next run's stopping rule is a hierarchical fit across models and λ (owner, 26 Nov; notes section 77):
logit(rate) = model effect + λ effect + per-cell noise, one fit per outcome. Its input is counts per cell. So that results
can be read without joins or guessing, every row a run writes carries `meta` (see `meta`), stated once here.

Rows written before schema 1 have no `meta`, and schema 1 lacks `tasks` and `part`. `cells` takes what is missing from the
`summary.json` beside them, using only what that summary states (its own fields, or a flag present in its `argv`, whose
absence means the entrypoint's default, see HISTORICAL_DEFAULTS); a field it does not state is never assumed, and the file
is reported as refused instead.
"""
import json
from pathlib import Path

from . import jsonl, reporting
from .outcomes import ARMS
from .prompts import VARIANTS
from .tasks import PARTS, SELECTIONS

SCHEMA = 2
# Where a row was sampled, and what that says: whether the model was trained under an arm (so the row has an arm and λ),
# whether the run's steps.jsonl counts its prompt groups, and, for the trainer's stages, which statements and split it
# used. The trainer has trained on the training split and evaluated on the evaluation split, every statement of each,
# since its first commit (e36d7ea; the probe stage since b396791, which records `probe_split: train`), so these columns
# also hold for its rows written before schema 2. `untrained` is a model never trained here (vLLM or API probes), whose
# run states its own selection.
STAGES = {
    'training': dict(trained=True, steps=True, tasks='all', part='train'),
    'before': dict(trained=True, steps=False, tasks='all', part='evaluation'),
    'after': dict(trained=True, steps=False, tasks='all', part='evaluation'),
    'probe': dict(trained=True, steps=False, tasks='all', part='train'),
    'untrained': dict(trained=False, steps=False, tasks=None, part=None),
}
SELECTION = ('tasks', 'part')
FIELDS = ('schema', 'model', 'revision', 'arm', 'penalty', 'stage', 'report_mode', 'taskset', *SELECTION, 'variant',
          'template', 'run')
# What a probe used when its recorded argv did not give --tasks or --part: these defaults are unchanged in every probe
# entrypoint and in `tasks.select_tasks` since the first commit (e36d7ea), so an argv without the flag states them.
HISTORICAL_DEFAULTS = {'tasks': 'all', 'part': 'train'}


def views(stem):
    """A stream's row files in order of preference: the rescored view over the raw one."""
    return (f'{stem}.rescored.jsonl', f'{stem}.jsonl')


# The row files of each stage, in order of preference; one file per run folder and stage is read. API probes also write
# `rows.ok.*`, their rows without failed requests. The transcript export (export_transcripts.row_files) prefers the same.
VIEWS = {'training': views('training-rollouts'), 'before': views('before-evaluation'), 'after': views('after-evaluation'),
         'probe': views('probe-evaluation'),
         'untrained': ('rows.ok.rescored.jsonl', 'rows.rescored.jsonl', 'rows.ok.jsonl', 'rows.jsonl')}
# The outcomes the prior is fitted to, as each row records them.
OUTCOMES = {'genuine': lambda row: bool(row['genuine']), 'refutation': lambda row: bool(row['verified_refutation']),
            'exploit': lambda row: bool(row['exploit'])}


def preferred(run_dir, names):
    """The first of `names` present in `run_dir`, or None."""
    return next((run_dir/name for name in names if (run_dir/name).exists()), None)


def meta(*, model, revision, arm, penalty, stage, report_mode, taskset, tasks, part, variant, template, run):
    """The provenance a row carries. `arm` and `penalty` are given exactly when the stage's model is trained under an arm;
    `variant` is the prompt's permission variant (prompts.VARIANTS), a condition of its own; `tasks` and `part` are which
    statements and which split were sampled (tasks.select_tasks), fixed by the stage for the trainer's stages. `run` is the
    run folder's name, a label only: `cells` identifies a run by its path."""
    trained = STAGES[stage]['trained']
    if trained != (arm is not None) or trained != (penalty is not None):
        raise ValueError(f'a {stage} row has an arm and a penalty exactly when its model is trained here')
    if trained and arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}')
    if variant not in VARIANTS:
        raise ValueError(f'unknown prompt variant {variant!r}')
    if tasks not in SELECTIONS or part not in PARTS:
        raise ValueError(f'unknown task selection {tasks!r} or split {part!r}')
    if STAGES[stage]['part'] is not None and (tasks, part) != (STAGES[stage]['tasks'], STAGES[stage]['part']):
        raise ValueError(f"a {stage} row samples tasks={STAGES[stage]['tasks']!r} part={STAGES[stage]['part']!r}, not {tasks!r} {part!r}")
    if not (model and revision and run and report_mode and taskset and isinstance(template, dict)):
        raise ValueError('model, revision, run, report mode, task set and the template settings are required')
    return dict(schema=SCHEMA, model=model, revision=revision, arm=arm, penalty=float(penalty) if trained else None,
                stage=stage, report_mode=report_mode, taskset=taskset, tasks=tasks, part=part, variant=variant,
                template=template, run=run)


def stamp(rows, provenance):
    """`rows` with `provenance` added to each."""
    return [dict(row, meta=provenance) for row in rows]


def flag(argv, name):
    """The value given for `--name` in `argv`, or None when it was not given. The last one counts, as with argparse: the
    pod scripts append overrides to a base argv (review round 31)."""
    found = None
    for index, word in enumerate(argv):
        given = word.split('=', 1)[0]
        if len(given) > 2 and given != f'--{name}' and f'--{name}'.startswith(given):
            raise ValueError(f'its argv abbreviates --{name} as {given}, which argparse may have read as it or as another option')
        if word == f'--{name}':
            if index + 1 == len(argv):
                raise ValueError(f'its argv ends with --{name} and no value')
            found = argv[index+1]
        elif word.startswith(f'--{name}='):
            found = word.split('=', 1)[1]
    return found


def stated(summary, field, name):
    """What `summary` states for one setting: its own field, else the flag in its argv; None when it states neither."""
    if field in summary:
        return summary[field]
    return flag(summary['argv'] if 'argv' in summary else [], name)


def variant_stated(summary):
    """The prompt variant `summary` states: the trainer records it as prompt_variant, the probes as variant."""
    found = [value for value in (stated(summary, 'prompt_variant', 'prompt-variant'), stated(summary, 'variant', 'variant')) if value is not None]
    return found[0] if found else None


def read_summary(path):
    """The summary.json beside a row file."""
    summary_path = path.parent/'summary.json'
    if not summary_path.is_file():
        raise ValueError('no summary.json beside it')
    return json.loads(summary_path.read_text())


def selection(path, stage):
    """`tasks` and `part` of a row file written before schema 2: fixed by the stage for the trainer's stages, else what the
    run's summary.json states. `tasks` is read from argv only: a probe summary's own `tasks` field is a count of tasks."""
    if STAGES[stage]['part'] is not None:
        return {name: STAGES[stage][name] for name in SELECTION}
    summary = read_summary(path)
    if 'argv' not in summary:
        raise ValueError('its summary.json has no argv, so which statements and split it sampled is not stated')
    given = dict(tasks=flag(summary['argv'], 'tasks'), part=stated(summary, 'part', 'part'))
    # fallback-ok: a flag absent from a recorded argv took the entrypoint's default, unchanged since the first commit
    return {name: HISTORICAL_DEFAULTS[name] if value is None else value for name, value in given.items()}


def from_summary(path, stage):
    """Provenance for a pre-schema row file, from what its summary.json states; raises ValueError naming what is missing."""
    summary = read_summary(path)
    trained = STAGES[stage]['trained']
    if not trained and stated(summary, 'adapter', 'adapter') is not None:
        raise ValueError('its summary.json names an adapter: a trained model, not an untrained probe')
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
    return meta(stage=stage, run=path.parent.name, **found, **selection(path, stage))


def out_named(path):
    """The folder name a pre-schema row file's summary gives as its --out, or None when its argv names none."""
    summary = read_summary(path)
    out = flag(summary['argv'], 'out') if 'argv' in summary else None
    return None if out is None else Path(out).name


def upgraded(provenance, path, stage):
    """A stamped row's provenance in the current schema: schema 1 had no `tasks` or `part`."""
    if provenance['schema'] == 1:
        kept = {name: provenance[name] for name in FIELDS if name != 'schema' and name not in SELECTION}
        return meta(**kept, **selection(path, stage))
    if provenance['schema'] != SCHEMA:
        raise ValueError(f"unknown provenance schema {provenance['schema']!r}")
    return provenance


def count_file(path, stage):
    """(provenance, counts) of one row file, or None when it has no row of its own to count. Raises when its provenance
    is unclear or a count cannot be read, so that a file refused part way adds nothing."""
    rows = [row for row in jsonl.read(path) if 'copied_from' not in row and 'partial_batch' not in row]
    left_out = set()  # training steps whose rows were scored but never trained on
    if STAGES[stage]['steps']:
        summary = read_summary(path)
        if 'untrained_step' in summary:  # a failed run's step in progress (hf_grpo): its rows and steps line say nothing of training
            left_out.add(summary['untrained_step'])
            rows = [row for row in rows if row['step'] not in left_out]  # older rows without `step` are refused here
    if not rows:
        return None
    if any('scored' in row and row['scored'] is False for row in rows):
        raise ValueError('it holds unscored rows: rescore them (entrypoints/rescore.py) and the rescored file is read')
    # Every row of one file has the same provenance: stamped rows carry it, older files take it from their summary.
    stamped = [row['meta'] for row in rows if 'meta' in row]
    if stamped and (len(stamped) != len(rows) or any(other != stamped[0] for other in stamped)):
        raise ValueError('its rows do not all carry the same provenance')
    provenance = upgraded(stamped[0], path, stage) if stamped else from_summary(path, stage)
    # A copy of another run's rows (once a parent folder held its child's rows.jsonl) would be counted twice; the folder a
    # run wrote into is the one its rows or its summary name (review round 31).
    wrote = provenance['run'] if stamped else out_named(path)
    if wrote is not None and wrote != path.parent.name:
        raise ValueError(f'its rows were written in a run folder named {wrote!r}, not here: a copy is not counted again')
    if provenance['stage'] != stage:
        raise ValueError(f"its rows say stage {provenance['stage']!r}, its name says {stage!r}")
    found = dict(answers=len(rows), **{outcome: sum(read(row) for row in rows) for outcome, read in OUTCOMES.items()})
    if STAGES[stage]['steps']:
        lines = [line for line in jsonl.read(path.parent/'steps.jsonl') if 'copied_from' not in line and line['step'] not in left_out]
        found.update(groups=sum(line['mixed_groups'] + line['zero_advantage_groups'] for line in lines),
                     mixed_groups=sum(line['mixed_groups'] for line in lines))
    return provenance, found


def cells(root):
    """Counts per run and stage of every row file under `root`, and every other .jsonl file there with why it was not
    counted (`refused`). A run is its folder's path under `root` (`run`): a folder name alone is shared by runs of
    different jobs. Each entry has the run's provenance (FIELDS but the schema and the folder's name), answers and each of
    OUTCOMES; training entries also count prompt groups and the mixed ones (reward variation) from the run's steps.jsonl.
    Rows copied from another run, and rows of a batch that was cut short, are counted where they belong: in the original
    run, and in no step. Runs are never pooled here: they differ in length and batch shape, so the fit treats each as its
    own unit inside its (model, arm, λ) cell."""
    root = Path(root)
    files = sorted(root.rglob('*.jsonl'))
    chosen = {}  # path -> stage, one per run folder and stage
    for run_dir in sorted({path.parent for path in files}):
        for stage, names in VIEWS.items():
            path = preferred(run_dir, names)
            if path is not None:
                chosen[path] = stage
    table, refused = [], []
    for path in files:
        if path in chosen:
            try:
                counted = count_file(path, chosen[path])
            except (KeyError, ValueError, TypeError, OSError) as error:  # reported to the caller in `refused`, never counted
                refused.append((str(path), f'{type(error).__name__}: {error}'))
                continue
            if counted is None:
                refused.append((str(path), 'nothing of its own to count: no rows, or only rows copied from another run, '
                                           'of a batch cut short, or of a step never trained on'))
            else:
                provenance, found = counted
                kept = {name: provenance[name] for name in FIELDS if name not in ('schema', 'run')}
                table.append(dict(kept, run=path.parent.relative_to(root).as_posix(), **found))
        elif path.name == 'steps.jsonl' and preferred(path.parent, VIEWS['training']) in chosen:
            continue  # read with the training rows beside it
        else:
            refused.append((str(path), unread(path)))
    return table, refused


def unread(path):
    """Why `cells` does not read `path`."""
    for names in VIEWS.values():
        if path.name in names:
            return f'not read: {preferred(path.parent, names).name} is the preferred view of the same rows'
    return 'not read: not a row file of any stage (provenance.VIEWS)'
