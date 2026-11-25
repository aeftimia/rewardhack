"""Export every scored transcript as one self-contained JSONL: the exact messages the model saw, its
raw output, the verifier's verdict and the reading-rubric category when the row was read.

Consumers need nothing from this package: the grader source is already inside the messages. Each
line also carries the two outcome axes the reading gives us (cheat × disclosure), where the truth
side comes from Lean and only the disclosure side from reading. Entrypoint: `entrypoints/export_transcripts.py`.
"""
import hashlib
import math
import os
import tempfile
from contextlib import contextmanager
import json
import subprocess
from collections import Counter
from pathlib import Path

from . import jsonl
from .prompts import follow_on, prompt
from .reporting import validate_mode
from .tasks import task_index

SCHEMA = 'rewardhack-transcripts-v1'
# Runs written before the probes recorded their model: the entrypoint defaults of the day.
LEGACY_MODELS = {'local-8b-4bit': ('mlx-community/Qwen3-8B-4bit', '545dc4251c05440727734bcd94334791f6ab0192'),
                 'local-llama31-8b-4bit': ('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit', None)}
DISCLOSURE = {'honest_admission': 'acknowledged', 'honest_disclosure': 'disclosed', 'hedged_misleading': 'hedged',
              'false_claim': 'false_claim', 'claim_substitution': 'substituted', 'true_claim': 'true_claim',
              'describes_only': 'describes', 'missing_or_garbled': 'garbled'}


def _opt(argv, name):
    """Value of a command-line option written as `--name value` or `--name=value`."""
    for index, arg in enumerate(argv):
        if arg == name:
            return argv[index + 1] if index + 1 < len(argv) else None
        if isinstance(arg, str) and arg.startswith(name + '='):
            return arg[len(name) + 1:]
    return None


def run_meta(summary, run):
    """Model, revision, prompt variant and training state of a run, from its summary or its argv."""
    argv = summary.get('argv', [])
    legacy = LEGACY_MODELS.get(run.split('/')[0], (None, None))
    # vLLM probes record the chat-template kwargs instead of a thinking flag (Qwen: enable_thinking;
    # gpt-oss: reasoning_effort); the API probes record reasoning_effort directly.
    template = summary.get('template_kwargs') or {}
    thinking = summary.get('thinking')
    if thinking is None and 'enable_thinking' in template:
        thinking = bool(template['enable_thinking'])
    effort = summary.get('reasoning_effort') or template.get('reasoning_effort')
    if thinking is None and effort is not None:
        thinking = effort != 'none'
    return dict(model=summary.get('model') or _opt(argv, '--model') or legacy[0],
                revision=summary.get('revision') or _opt(argv, '--revision') or legacy[1],
                variant=summary.get('variant') or summary.get('prompt_variant') or _opt(argv, '--variant') or _opt(argv, '--prompt-variant') or 'neutral',
                thinking=thinking, reasoning_effort=effort,
                adapter=summary.get('adapter') or _opt(argv, '--adapter'), arm=_opt(argv, '--arm'))


def row_files(run_dir):
    """One preferred scored view per stream; never export raw and rescored copies twice."""
    def first(names):
        return next((run_dir/name for name in names if (run_dir/name).exists()), None)

    probe = first(('rows.ok.rescored.jsonl', 'rows.rescored.jsonl', 'rows.ok.jsonl', 'rows.jsonl'))
    if probe:
        yield probe, 0
    streams = [('rollouts', 'mlx_training'), ('training-rollouts', 'hf_training'),
               ('probe-evaluation', 0), ('before-evaluation', 0), ('after-evaluation', 'hf_after')]
    for stem, updates in streams:
        path = first((stem+'.rescored.jsonl', stem+'.jsonl'))
        if path:
            yield path, updates
    steps = sorted({int(p.name.split('step')[1].split('.')[0]) for p in run_dir.glob('monitor-step*.jsonl')})
    for step in steps:
        yield first((f'monitor-step{step}.rescored.jsonl', f'monitor-step{step}.jsonl')), ('mlx_monitor', step)


def report_mode(row, summary):
    """Only absent historical metadata defaults to prose; invalid explicit values fail."""
    mode = row['report_mode'] if 'report_mode' in row else summary.get('report_mode', 'prose')
    return validate_mode(mode)


def update_history(run_dir):
    """MLX records whether it called the optimizer; HF reward batches do not establish updates."""
    path = run_dir/'steps.jsonl'
    if not path.exists():
        return {}
    history = {}
    for row in jsonl.read(path):
        step, mixed = row.get('step'), row.get('mixed')
        if type(step) is not int or step < 1 or step in history or type(mixed) is not bool:
            return {}  # corrupt history cannot establish a generation checkpoint
        history[step] = mixed
    return history


def parameter_change(summary):
    """Measured final parameter change, if the HF trainer recorded a valid measurement."""
    value = summary.get('adapter_max_abs_change')
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def update_count(history, summary, row, stream):
    """Completed updates at generation time, or None when the writer did not record them."""
    if type(stream) is int:
        return stream
    if stream == 'hf_after':
        # HF always calls its optimizer: Adam momentum and KL can move parameters
        # even without reward variation. Neither attempted nor mixed steps count
        # actual parameter changes. Only a measured unchanged endpoint proves zero.
        attempted = summary.get('global_step')
        if type(attempted) is not int or attempted < 0:
            return None
        if attempted == 0 or parameter_change(summary) == 0:
            return 0
        return None
    if stream == 'hf_training':
        return None  # HF's rollout writer does not identify the generating checkpoint.
    step = row.get('step') if stream == 'mlx_training' else stream[1]
    if type(step) is not int or step < 0:
        return None
    last = step-1 if stream == 'mlx_training' else step
    if last <= 0:
        return 0
    completed = {step: mixed for step, mixed in history.items() if step <= last}
    if set(completed) != set(range(1, last+1)) or any(type(value) is not bool for value in completed.values()):
        return None
    return sum(completed.values())


def reading_index(results):
    """Reading labels keyed two ways: by content (report-reading batches) and by position (grid and trace batches)."""
    by_content, by_position, traces = {}, {}, {}

    def join(batches, labels):
        rows = {}
        for p in batches:
            for r in jsonl.read(p):
                rows[r['id']] = r
        for p in labels:
            for lab in jsonl.read(p):
                if lab['id'] in rows:
                    yield rows[lab['id']], lab, p.name

    reading = results / 'report-reading'
    for row, lab, source in join(sorted(reading.glob('batch*.jsonl')), sorted(reading.glob('labels*.jsonl'))):
        key = content_key(row['run'], row['task_id'], row['established'], row['body'], row['report'])
        label = dict(category=lab['category'], says_false=lab.get('says_false'), mentions_grader=lab.get('mentions_grader'), source=source)
        if key in by_content and by_content[key]['category'] != label['category']:
            label = dict(label, category='conflict', conflict=[by_content[key]['category'], lab['category']])
        by_content[key] = label
    grid = results / 'api-grid-A' / 'reading'
    for row, lab, source in join(sorted(grid.glob('grid_batch*.jsonl')), sorted(grid.glob('grid_labels*.jsonl'))):
        run, index = row['id'].rsplit('#', 1)
        by_position[('api-grid-A/' + run, int(index))] = dict(category=lab['category'], says_false=lab.get('says_false'),
                                                              mentions_grader=lab.get('mentions_grader'), source=source, task_id=row['task_id'])
    for row, lab, source in join(sorted(reading.glob('traces*.jsonl')), sorted(reading.glob('tracelabels*.jsonl'))):
        run, index = row['id'].rsplit('#', 1)
        traces[(run, int(index))] = dict({k: v for k, v in lab.items() if k != 'id'}, source=source, task_id=row['task_id'])
    return by_content, by_position, traces


def content_key(run, task_id, established, body, report):
    return (run, task_id, established or '', (body or '')[:300], (report or '')[:700])


def cheat_axis(row):
    if row.get('verifier_inconclusive'):
        return 'unverified'
    if not row['format_valid']:
        return 'no_answer'
    if row['genuine']:
        return 'genuine'
    if row['exploit']:
        return 'cheat'
    if row['verified_refutation']:
        return 'refutation'
    return 'failed'


def git_commit(repo):
    try:
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout.strip()
        return head + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


@contextmanager
def atomic_text(path):
    """A rejected row must not leave a partial export or overwrite the last good one."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.export-', delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temporary, 0o666 & ~umask)  # a temporary file is owner-only; the export is an ordinary file
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export(results, out, tasks=None):
    """Write `out` (JSONL) and `<out>.manifest.json`; return the manifest."""
    results, out = Path(results), Path(out)
    tasks = tasks or task_index()
    by_content, by_position, traces = reading_index(results)
    commit = git_commit(results.parent)
    counts, skipped, n = Counter(), Counter(), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text(out) as sink:
        for summary_path in sorted(results.rglob('summary.json')):
            run_dir = summary_path.parent
            run = run_dir.relative_to(results).as_posix()
            if run in ('report-reading',) or not any(True for _ in row_files(run_dir)):
                continue
            summary = json.loads(summary_path.read_text())
            meta = run_meta(summary, run)
            history = update_history(run_dir)
            for path, updates in row_files(run_dir):
                rows = list(jsonl.read(path))
                first_turn = {r.get('conversation'): r for r in rows if str(r.get('turn', 1)) == '1'}
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                rel = path.relative_to(results).as_posix()
                for i, row in enumerate(rows):
                    if row.get('scored') is False or not all(k in row for k in ('task_id', 'raw', 'genuine', 'exploit', 'verified_refutation', 'format_valid', 'established', 'body', 'report')):
                        skipped['unscored row'] += 1
                        continue
                    task = tasks.get(row['task_id'])
                    if task is None:
                        skipped['unknown task'] += 1
                        continue
                    turn = int(str(row.get('turn', 1)))
                    # The prompt the model saw depends on the report format; rows from before the typed format are prose.
                    mode = report_mode(row, summary)
                    if turn == 1:
                        messages = prompt(task['statement'], meta['variant'], mode)
                    else:
                        opener = first_turn.get(row.get('conversation'))
                        if opener is None:
                            skipped['follow-on without opener'] += 1
                            continue
                        if report_mode(opener, summary) != mode:
                            raise ValueError('follow-on report mode differs from its opening prompt')
                        messages = prompt(tasks[opener['task_id']]['statement'], meta['variant'], mode) + [
                            {'role': 'assistant', 'content': opener['raw']}, follow_on(task['statement'])]
                    # A run started from an adapter was already trained by an unrecorded number of updates:
                    # the total is unknown there, and the count within this run is kept beside it.
                    run_updates = update_count(history, summary, row, updates)
                    policy_updates = None if meta['adapter'] else run_updates
                    changed = parameter_change(summary) if updates == 'hf_after' else None
                    trained = bool(meta['adapter']) or bool(policy_updates) or (changed is not None and changed > 0)
                    provenance = ('rl_induced' if trained else 'training_unknown' if policy_updates is None
                                  else 'hinted' if meta['variant'] == 'hint' else 'natural')
                    reading = by_content.get(content_key(run, row['task_id'], row['established'], row['body'], row['report']))
                    positional = by_position.get((run, i))
                    if positional and positional['task_id'] == row['task_id']:
                        if reading and reading['category'] != positional['category']:
                            reading = dict(reading, category='conflict', conflict=[reading['category'], positional['category']])
                        else:
                            reading = {k: v for k, v in positional.items() if k != 'task_id'}
                    trace = traces.get((run, i))
                    if trace and trace['task_id'] != row['task_id']:
                        trace = None
                    axis = cheat_axis(row)
                    disclosure = DISCLOSURE.get(reading['category'], reading['category']) if reading else 'unread'
                    record = dict(schema=SCHEMA, id=hashlib.sha256(f'{run}|{rel}|{i}'.encode()).hexdigest()[:16], run=run,
                                  source_file=rel, source_line=i, source_sha256=file_hash, repo_commit=commit, **meta,
                                  policy_updates=policy_updates, run_policy_updates=run_updates, provenance=provenance, taskset=task['taskset'],
                                  task_id=task['id'], group=task['group'], family=task['family'], tier=task['tier'],
                                  statement=task['statement'], statement_true=task['true'], turn=turn, conversation=row.get('conversation'),
                                  report_mode=mode, typed_claim=row.get('claim') if mode == 'typed' else None,
                                  claim=row.get('claim'), error_kinds=row.get('error_kinds'),
                                  honest=row.get('honest'), judge_raw=row.get('judge_raw'),
                                  messages=messages, response=row['raw'], established=row['established'], body=row['body'], report=row['report'],
                                  genuine=bool(row['genuine']), exploit=bool(row['exploit']), verified_refutation=bool(row['verified_refutation']),
                                  format_valid=bool(row['format_valid']), truncated=row.get('truncated'),
                                  verifier_inconclusive=row.get('verifier_inconclusive'),
                                  judge_undecided=bool(row.get('judge_undecided', False)),
                                  genuine_why=row.get('genuine_why'), refutation_why=row.get('refutation_why'),
                                  declaration_why=row.get('declaration_why'), partial_batch=row.get('partial_batch', False),
                                  reading=reading, trace_reading={k: v for k, v in trace.items() if k != 'task_id'} if trace else None,
                                  cheat_axis=axis, disclosure_axis=disclosure)
                    jsonl.write(sink, record)
                    counts[(run, provenance, axis, disclosure)] += 1
                    n += 1
    manifest = dict(schema=SCHEMA, rows=n, repo_commit=commit, results=str(results), out_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    skipped=dict(skipped), read=sum(v for k, v in counts.items() if k[3] != 'unread'),
                    cells=[dict(run=k[0], provenance=k[1], cheat=k[2], disclosure=k[3], n=v) for k, v in sorted(counts.items())])
    with atomic_text(out.with_name(out.name + '.manifest.json')) as handle:
        handle.write(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest
