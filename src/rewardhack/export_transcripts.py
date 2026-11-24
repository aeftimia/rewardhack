"""Export every scored transcript as one self-contained JSONL: the exact messages the model saw, its
raw output, the verifier's verdict and the reading-rubric category when the row was read.

Consumers need nothing from this package: the grader source is already inside the messages. Each
line also carries the two outcome axes the reading gives us (cheat × disclosure), where the truth
side comes from Lean and only the disclosure side from reading. Entrypoint: `entrypoints/export_transcripts.py`.
"""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

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
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


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
                variant=summary.get('variant') or _opt(argv, '--variant') or 'neutral',
                thinking=thinking, reasoning_effort=effort,
                adapter=summary.get('adapter') or _opt(argv, '--adapter'), arm=_opt(argv, '--arm'))


def row_files(run_dir):
    """(file, policy_updates) for every scored file of a run: probes have one, trainers have rollouts and monitors."""
    for name in ('rows.ok.rescored.jsonl', 'rows.rescored.jsonl'):
        if (run_dir / name).exists():
            yield run_dir / name, None
            break
    if (run_dir / 'rollouts.rescored.jsonl').exists():
        yield run_dir / 'rollouts.rescored.jsonl', 'step'
    for path in sorted(run_dir.glob('monitor-step*.rescored.jsonl'), key=lambda p: int(p.stem.split('step')[1].split('.')[0])):
        yield path, int(path.stem.split('step')[1].split('.')[0])


def reading_index(results):
    """Reading labels keyed two ways: by content (report-reading batches) and by position (grid and trace batches)."""
    by_content, by_position, traces = {}, {}, {}

    def join(batches, labels):
        rows = {}
        for p in batches:
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r['id']] = r
        for p in labels:
            for line in p.read_text().splitlines():
                if line.strip():
                    lab = json.loads(line)
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


def export(results, out, tasks=None):
    """Write `out` (JSONL) and `<out>.manifest.json`; return the manifest."""
    results, out = Path(results), Path(out)
    tasks = tasks or task_index()
    by_content, by_position, traces = reading_index(results)
    commit = git_commit(results.parent)
    counts, skipped, n = Counter(), Counter(), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as sink:
        for summary_path in sorted(results.rglob('summary.json')):
            run_dir = summary_path.parent
            run = run_dir.relative_to(results).as_posix()
            if run in ('report-reading',) or not any(True for _ in row_files(run_dir)):
                continue
            summary = json.loads(summary_path.read_text())
            meta = run_meta(summary, run)
            for path, updates in row_files(run_dir):
                lines = [line for line in path.read_text().splitlines() if line.strip()]
                rows = [json.loads(line) for line in lines]
                first_turn = {r.get('conversation'): r for r in rows if str(r.get('turn', 1)) == '1'}
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                rel = path.relative_to(results).as_posix()
                for i, row in enumerate(rows):
                    task = tasks.get(row['task_id'])
                    if task is None:
                        skipped['unknown task'] += 1
                        continue
                    turn = int(str(row.get('turn', 1)))
                    # The prompt the model saw depends on the report format; rows from before the typed format are prose.
                    mode = row.get('report_mode') or summary.get('report_mode') or 'prose'
                    validate_mode(mode)
                    if turn == 1:
                        messages = prompt(task['statement'], meta['variant'], mode)
                    else:
                        opener = first_turn.get(row.get('conversation'))
                        if opener is None:
                            skipped['follow-on without opener'] += 1
                            continue
                        messages = prompt(tasks[opener['task_id']]['statement'], meta['variant'], mode) + [
                            {'role': 'assistant', 'content': opener['raw']}, follow_on(task['statement'])]
                    policy_updates = row.get('step') if updates == 'step' else updates
                    trained = bool(meta['adapter']) or bool(policy_updates)
                    provenance = 'rl_induced' if trained else ('hinted' if meta['variant'] == 'hint' else 'natural')
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
                                  policy_updates=policy_updates, provenance=provenance, taskset=task['taskset'],
                                  task_id=task['id'], group=task['group'], family=task['family'], tier=task['tier'],
                                  statement=task['statement'], statement_true=task['true'], turn=turn, conversation=row.get('conversation'),
                                  report_mode=mode, typed_claim=row.get('claim') if mode == 'typed' else None,
                                  messages=messages, response=row['raw'], established=row['established'], body=row['body'], report=row['report'],
                                  genuine=bool(row['genuine']), exploit=bool(row['exploit']), verified_refutation=bool(row['verified_refutation']),
                                  format_valid=bool(row['format_valid']), truncated=row.get('truncated'),
                                  reading=reading, trace_reading={k: v for k, v in trace.items() if k != 'task_id'} if trace else None,
                                  cheat_axis=axis, disclosure_axis=disclosure)
                    sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                    counts[(run, provenance, axis, disclosure)] += 1
                    n += 1
    manifest = dict(schema=SCHEMA, rows=n, repo_commit=commit, results=str(results), out_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    skipped=dict(skipped), read=sum(v for k, v in counts.items() if k[3] != 'unread'),
                    cells=[dict(run=k[0], provenance=k[1], cheat=k[2], disclosure=k[3], n=v) for k, v in sorted(counts.items())])
    out.with_name(out.name + '.manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest
