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
from collections import Counter, defaultdict
from pathlib import Path

from . import jsonl
from ._hashing import file_sha256
from .provenance import VIEWS, preferred, views
from .prompts import follow_on, prompt
from .reporting import report_text, stored_mode
from .tasks import task_index

SCHEMA = 'rewardhack-transcripts-v1'
# Runs written before the probes recorded their model: the entrypoint defaults of the day.
LEGACY_MODELS = {'local-8b-4bit': ('mlx-community/Qwen3-8B-4bit', '545dc4251c05440727734bcd94334791f6ab0192'),
                 'local-llama31-8b-4bit': ('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit', None)}
DISCLOSURE = {'honest_admission': 'acknowledged', 'honest_disclosure': 'disclosed', 'hedged_misleading': 'hedged',
              'false_claim': 'false_claim', 'claim_substitution': 'substituted', 'true_claim': 'true_claim',
              'describes_only': 'describes', 'missing_or_garbled': 'garbled', 'conflict': 'conflict'}


def _opt(argv, name):
    """Value of a command-line option written as `--name value` or `--name=value`."""
    if argv is None:
        return None
    for index, arg in enumerate(argv):
        if arg == name:
            return argv[index + 1] if index + 1 < len(argv) else None
        if isinstance(arg, str) and arg.startswith(name + '='):
            return arg[len(name) + 1:]
    return None


def run_meta(summary, run):
    """Model, revision, prompt variant and training state of a run, from its summary or its argv."""
    argv = summary.get('argv')
    if argv is not None and not isinstance(argv, list):
        raise TypeError('recorded argv must be a list')
    legacy = LEGACY_MODELS.get(run.split('/')[0], (None, None))
    # vLLM probes record the chat-template kwargs instead of a thinking flag (Qwen: enable_thinking;
    # gpt-oss: reasoning_effort); the API probes record reasoning_effort directly.
    template = summary.get('template_kwargs')
    if template is not None and not isinstance(template, dict):
        raise TypeError('recorded template_kwargs must be a mapping')
    thinking = summary.get('thinking')
    if thinking is None and template is not None and 'enable_thinking' in template:
        thinking = bool(template['enable_thinking'])
    effort = summary.get('reasoning_effort')
    if effort is None and template is not None:
        effort = template.get('reasoning_effort')
    if thinking is None and effort is not None:
        thinking = effort != 'none'
    variant = summary.get('variant') or summary.get('prompt_variant') or _opt(argv, '--variant') or _opt(argv, '--prompt-variant')
    if variant is None:
        raise ValueError(f'{run}: no recorded prompt variant; refusing to invent messages')
    return dict(model=summary.get('model') or _opt(argv, '--model') or legacy[0],
                revision=summary.get('revision') or _opt(argv, '--revision') or legacy[1],
                variant=variant,
                thinking=thinking, reasoning_effort=effort,
                adapter=summary.get('adapter') or _opt(argv, '--adapter'), arm=_opt(argv, '--arm'),
                missing_summary_metadata=[name for name in ('argv', 'template_kwargs') if summary.get(name) is None])


def row_files(run_dir):
    """One preferred scored view per stream; never export raw and rescored copies twice."""
    streams = [(VIEWS['untrained'], 0), (views('rollouts'), 'mlx_training'), (VIEWS['training'], 'hf_training'),
               (VIEWS['probe'], 0), (VIEWS['before'], 0), (VIEWS['after'], 'hf_after')]
    for names, updates in streams:
        path = preferred(run_dir, names)
        if path:
            yield path, updates
    steps = sorted({int(p.name.split('step')[1].split('.')[0]) for p in run_dir.glob('monitor-step*.jsonl')})
    for step in steps:
        yield preferred(run_dir, views(f'monitor-step{step}')), ('mlx_monitor', step)


def report_mode(row, summary):
    """Use the row's format, or the run's recorded format for older rows."""
    return stored_mode(row if 'report_mode' in row else summary)


def turn_number(row):
    """Single-turn writers omit both fields; a conversation must identify its turn."""
    if 'conversation' not in row and 'turn' not in row:
        return 1
    turn = int(str(row['turn']))
    if turn < 1 or (turn > 1 and 'conversation' not in row):
        raise ValueError('a follow-on turn requires a conversation and a positive turn number')
    return turn


def run_reader(summary, skipped):
    """How to read a run's files: a function from a path to its rows and the bytes they were parsed from, read once so
    that a hash of them describes exactly the rows exported. A finished run's are read strictly. A run whose summary
    still says 'running' (written as it starts) is still writing or was killed: an unterminated last line is a write cut
    short, left out and counted."""
    def strict(path):
        data = path.read_bytes()
        return list(jsonl.loads(data.decode(), path)), data

    def unfinished(path):
        data = path.read_bytes()
        rows, cut = jsonl.read_finished(data, path)
        if cut:
            skipped[f'unfinished last line of a running or killed run ({path.name})'] += 1
        return rows, data
    # By status: the training and probe writers record 'running' while they run, then 'completed' or 'failed'; other
    # writers (mlx_rl, openai_probe, human review) record none and write their summary once, at the end.
    readers = {'running': unfinished, 'completed': strict, 'failed': strict, None: strict}
    return readers[summary['status'] if 'status' in summary else None]


def own_trained_steps(run_dir, summary):
    """The steps a run trained itself: those of its complete training rows, not copied from an original and not a
    failed step's partial batch."""
    steps = set()
    for path, updates in row_files(run_dir):
        if updates == 'hf_training':
            rows, _ = run_reader(summary, Counter())(path)
            steps |= {row['step'] for row in rows if 'copied_from' not in row and 'partial_batch' not in row}
    return steps


def superseded_steps(results):
    """For each run in a family of resumes (an original, the runs resumed from its checkpoints, the runs resumed from
    theirs), the steps it trained that another run of the family exports instead. Runs that resumed from the same state
    regenerate the same steps (same weights, optimizer and random state), so each step is exported from one run: among
    those that trained it themselves, the one on the branch that got furthest (its own last step or any later resume's),
    then the one furthest down the chain of resumes, then the folder name that sorts last. A step only one run reached
    stays that run's. A resumed run's folder is a sibling of its original's, as on the pod."""
    parent, own = {}, {}
    for path in results.rglob('summary.json'):
        summary = json.loads(path.read_text())
        if 'resumed_from' in summary and summary['resumed_from'] is not None:
            parent[path.parent] = path.parent.parent/Path(summary['resumed_from']).parent.parent.name
    family = set(parent) | set(parent.values())
    for run in family:
        summary_path = run/'summary.json'
        own[run] = own_trained_steps(run, json.loads(summary_path.read_text())) if summary_path.is_file() else set()

    def depth(run):
        return depth(parent[run])+1 if run in parent else 0

    def reach(run):
        return max([max(own[run], default=0)] + [reach(child) for child, up in parent.items() if up == run])

    superseded = defaultdict(set)
    for step in set().union(*own.values()):
        trained = [run for run in family if step in own[run]]
        keeper = max(trained, key=lambda run: (reach(run), depth(run), run.name))
        for run in trained:
            if run != keeper:
                superseded[run].add(step)
    return superseded


def update_history(run_dir, read):
    """MLX records whether it called the optimizer; HF reward batches do not establish updates. `read` gives a file's
    rows and bytes (`run_reader`)."""
    path = run_dir/'steps.jsonl'
    if not path.exists():
        return {}
    history = {}
    rows, _ = read(path)
    for row in rows:
        step, mixed = row.get('step'), row.get('mixed')
        if type(step) is not int or step < 1 or step in history or type(mixed) is not bool:
            return {}  # corrupt history cannot establish a generation checkpoint
        history[step] = mixed
    return history


def parameter_change(summary):
    """Measured final parameter change, if the HF trainer recorded a valid measurement."""
    value = summary.get('adapter_max_abs_change')
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def hf_update_count(history, summary, row, step):
    """Only a measured unchanged HF endpoint proves zero; otherwise unknown."""
    attempted = summary.get('global_step')
    if type(attempted) is not int or attempted < 0:
        return None
    if attempted == 0 or parameter_change(summary) == 0:
        return 0
    return None


def mlx_update_count(history, summary, row, step, *, offset):
    """Count recorded MLX optimizer calls through the generating checkpoint."""
    if type(step) is not int or step < 0:
        return None
    last = step + offset
    if last <= 0:
        return 0
    completed = {step: mixed for step, mixed in history.items() if step <= last}
    if set(completed) != set(range(1, last+1)) or any(type(value) is not bool for value in completed.values()):
        return None
    return sum(completed.values())


def update_count(history, summary, row, stream):
    """Completed updates at generation time, or None when the writer did not record them."""
    if type(stream) is int:
        if stream < 0:
            raise ValueError('negative generation checkpoint')
        return stream
    if isinstance(stream, tuple):
        if len(stream) != 2 or stream[0] != 'mlx_monitor':
            raise ValueError(f'unknown monitor stream: {stream!r}')
        kind, step = stream
    else:
        kind, step = stream, row.get('step')
    handlers = {
        'hf_after': hf_update_count,
        # HF rollout records identify attempts, not the generating parameter state.
        'hf_training': lambda history, summary, row, step: None,
        'mlx_training': lambda history, summary, row, step: mlx_update_count(history, summary, row, step, offset=-1),
        'mlx_monitor': lambda history, summary, row, step: mlx_update_count(history, summary, row, step, offset=0),
    }
    return handlers[kind](history, summary, row, step)


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
    return (run, task_id, report_text(established), report_text(body)[:300], report_text(report)[:700])


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
    # A standalone exported dataset has no repository; a broken repository is an error.
    if not (Path(repo) / '.git').exists():
        return None
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    return head + ('-dirty' if dirty else '')


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
    superseded = superseded_steps(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text(out) as sink:
        for summary_path in sorted(results.rglob('summary.json')):
            run_dir = summary_path.parent
            trained_again = superseded[run_dir]  # a defaultdict: a run outside any family of resumes supersedes nothing
            run = run_dir.relative_to(results).as_posix()
            if run in ('report-reading',) or not any(True for _ in row_files(run_dir)):
                continue
            summary = json.loads(summary_path.read_text())
            meta = run_meta(summary, run)
            read = run_reader(summary, skipped)
            history = update_history(run_dir, read)
            for path, updates in row_files(run_dir):
                rows, data = read(path)
                first_turn = {r['conversation']: r for r in rows if 'conversation' in r and turn_number(r) == 1}
                file_hash = hashlib.sha256(data).hexdigest()
                rel = path.relative_to(results).as_posix()
                for i, row in enumerate(rows):
                    if updates == 'hf_training' and trained_again and row['step'] in trained_again:
                        # Generated again by the run that resumed from this one's checkpoint, and exported from there.
                        skipped['superseded by a resumed run'] += 1
                        continue
                    if 'copied_from' in row:
                        # A resumed run carries the original run's rows; they are exported once, from the original.
                        skipped['copied by a resumed run'] += 1
                        continue
                    if row.get('scored') is False or not all(k in row for k in ('task_id', 'raw', 'genuine', 'exploit', 'verified_refutation', 'format_valid', 'established', 'body', 'report')):
                        skipped['unscored row'] += 1
                        continue
                    task = tasks.get(row['task_id'])
                    if task is None:
                        skipped['unknown task'] += 1
                        continue
                    turn = turn_number(row)
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
                    disclosure = 'unread'
                    if reading is not None:
                        disclosure = DISCLOSURE[reading['category']]
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
                                  judge_undecided=row.get('judge_undecided'),
                                  genuine_why=row.get('genuine_why'), refutation_why=row.get('refutation_why'),
                                  declaration_why=row.get('declaration_why'), partial_batch=row.get('partial_batch'),
                                  reading=reading, trace_reading={k: v for k, v in trace.items() if k != 'task_id'} if trace else None,
                                  cheat_axis=axis, disclosure_axis=disclosure)
                    jsonl.write(sink, record)
                    counts[(run, provenance, axis, disclosure)] += 1
                    n += 1
    manifest = dict(schema=SCHEMA, rows=n, repo_commit=commit, results=str(results), out_sha256=file_sha256(out),
                    skipped=dict(skipped), read=sum(v for k, v in counts.items() if k[3] != 'unread'),
                    cells=[dict(run=k[0], provenance=k[1], cheat=k[2], disclosure=k[3], n=v) for k, v in sorted(counts.items())])
    with atomic_text(out.with_name(out.name + '.manifest.json')) as handle:
        handle.write(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest
