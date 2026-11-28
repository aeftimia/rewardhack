"""Look up a run's recorded generations, scored answers and phase metrics."""
import hashlib
import io
import json
import re
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

from . import archive, jsonl, outcomes
from .json_objects import unique_object
from .provenance import VIEWS
from .run_registry import CONFIG_FILE, run_id_directory


def _output_index(directory, *, verified=False):
    run_id = directory.name
    path = directory / 'outputs.json'
    local_index = path.is_file()
    if local_index:
        index = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if not local_index or (verified and index['backup_status'] == 'pending'):
        local_index = False
        text = archive.run('rclone', 'cat', f's3:{archive.BUCKET}/experiments/{run_id}/outputs.json',
                           timeout=archive.REQUEST_TIMEOUT)
        if not text.strip():
            raise FileNotFoundError(f'no published output index for experiment {run_id}')
        index = json.loads(text, object_pairs_hook=unique_object)
    if index['run_id'] != run_id:
        raise ValueError(f'output index identifies a different experiment: {run_id}')
    if index['backup_status'] == 'pending' and local_index and index['uri'] is None and index['archive_id'] is None:
        return index
    archive_id = index['archive_id']
    if (index['backup_status'] != 'verified' or not isinstance(archive_id, str) or not archive.RUN_ID.fullmatch(archive_id)
            or index['uri'] != f's3://{archive.BUCKET}/{archive_id}/'):
        raise ValueError(f'output index names an invalid archive or a different experiment: {run_id}')
    return index


@contextmanager
def _verified_file(local, remote, digest):
    with TemporaryDirectory() as temporary:
        if not local.is_file():
            if remote is None:
                raise FileNotFoundError(f'{local}: local output missing while S3 backup is pending')
            local = Path(temporary) / 'rows.jsonl'
            archive.run('rclone', 'copyto', remote, str(local), timeout=archive.TRANSFER_TIMEOUT)
        source = local if remote is None else remote
        with local.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != digest:
                raise ValueError(f'output digest differs from the run index: {source}')
            handle.seek(0)
            with io.TextIOWrapper(handle, encoding='utf-8') as text:
                yield text


def registered_checkpoint(directory, step):
    """Resolve an experiment ID and optimizer checkpoint from their verified records."""
    index = _output_index(directory, verified=True)
    name = index['effective_config']
    if not name or not re.fullmatch(r'train-[a-zA-Z0-9_-]+/' + re.escape(CONFIG_FILE), name):
        raise ValueError(f'{directory.name}: missing or invalid archived effective configuration')
    arm = Path(name).parent.name
    checkpoints = index['checkpoints']
    if 'trainer' not in checkpoints or str(step) not in checkpoints['trainer']:
        raise ValueError(f'{directory.name}: no archived trainer checkpoint at step {step}')
    result = directory / 'checkout/results' / directory.name
    local_receipt = result / archive.RECORD
    receipt = archive.archive_entry(str(local_receipt) if local_receipt.is_file() else index['archive_id'])
    digest = index['effective_config_sha256']
    prefix = f'{arm}/trainer/checkpoint-{step}/'
    members = {p: sha for p, sha in receipt['sha256'].items() if p.startswith(prefix)}
    if (receipt['run_id'] != index['archive_id'] or receipt['commit'] != index['source_commit']
            or receipt['sha256'][name] != digest or checkpoints['trainer'][str(step)] != members):
        raise ValueError(f'{directory.name}: archive receipt differs from the registered checkpoint or config')
    local = directory / 'effective.json'
    if not local.is_file():
        local = result / name
    with _verified_file(local, f"s3:{archive.BUCKET}/{index['archive_id']}/{name}", digest) as handle:
        parent = json.load(handle, object_pairs_hook=unique_object)
    if parent['run_id'] != directory.name or parent['output_arm'] != arm:
        raise ValueError('archived configuration identifies a different experiment or training arm')
    return parent, arm, receipt


def _selection(run_id, phase, steps, state):
    if phase not in VIEWS:
        raise ValueError(f'unknown experiment phase: {phase!r}')
    if steps is not None:
        first, last = steps
        if phase != 'training' or type(first) is not int or type(last) is not int or not 1 <= first <= last:
            raise ValueError('steps must be an inclusive positive training-step range')
    directory = run_id_directory(Path.cwd() / 'experiments' if state is None else state, run_id)
    return directory, _output_index(directory)


def _rows(directory, index, phase, stream, steps):
    run_id = index['run_id']
    files = index[stream][phase]
    if not files:
        raise FileNotFoundError(f'{run_id}: no archived {stream} for phase {phase}')
    names = VIEWS[phase] if stream == 'outputs' else (f'{phase}-generations.jsonl',)
    selected = {}
    for name in files:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name or path.name not in names:
            raise ValueError(f'invalid {phase} artifact path: {name}')
        if path.parent not in selected or names.index(path.name) < names.index(selected[path.parent].name):
            selected[path.parent] = path
    result = directory / 'checkout/results' / run_id
    for relative in sorted(selected.values()):
        local = result / relative
        if not local.resolve().is_relative_to(result.resolve()):
            raise ValueError(f'output path escapes the result directory: {relative}')
        remote = f"s3:{archive.BUCKET}/{index['archive_id']}/{relative}" if index['backup_status'] == 'verified' else None
        with _verified_file(local, remote, files[str(relative)]) as handle:
            for row in jsonl.parse(handle, relative):
                if stream == 'generations' and (row['run_id'] != run_id or row['phase'] != phase):
                    raise ValueError(f'generation belongs to a different run or phase: {relative}')
                if steps is not None and not steps[0] <= row['step'] <= steps[1]:
                    continue
                yield row


def experiment_rows(run_id, phase, *, stream='outputs', steps=None, include_inherited=False, state=None):
    """Yield examples from one phase, selecting an inclusive training-step range if given.

    The outputs stream uses the preferred scored view per run folder. The generations stream
    contains the actual inputs and sampled outputs as recorded before scoring. Copied predecessor
    rows are included only on request. An absent phase raises rather than implying zero samples.
    """
    if stream not in ('outputs', 'generations'):
        raise ValueError(f'unknown output stream: {stream!r}')
    directory, index = _selection(run_id, phase, steps, state)
    for row in _rows(directory, index, phase, stream, steps):
        if include_inherited or 'copied_from' not in row:
            yield row


def summarize_experiment(run_id, phase, *, steps=None, include_inherited=False, state=None):
    """Return protocol outcome rates, denominators and mean scored reward for one phase.

    Inherited, unscored and partial-batch counts make the sample population explicit. A
    scored answer from an interrupted training batch is an observation, even if no update ran.
    """
    directory, index = _selection(run_id, phase, steps, state)
    if not index['outputs'][phase] and not index['generations'][phase]:
        raise FileNotFoundError(f'{run_id}: no archived rows for phase {phase}')
    rows = []
    inherited = unscored = partial = 0
    scored_stream = _rows(directory, index, phase, 'outputs', steps) if index['outputs'][phase] else ()
    for row in scored_stream:
        if 'copied_from' in row:
            inherited += 1
            if not include_inherited:
                continue
        if row.get('scored') is False or ('error' in row and 'raw' not in row):
            unscored += 1
            continue
        partial += bool(row.get('partial_batch'))
        rows.append(row)
    generation = None
    if index['generations'][phase]:
        prompt_lengths, answer_lengths = [], []
        for row in _rows(directory, index, phase, 'generations', steps):
            if include_inherited or 'copied_from' not in row:
                prompt_lengths.append(len(row['prompt_token_ids']))
                answer_lengths.append(len(row['completion_token_ids']))
        generation = dict(samples=len(answer_lengths), prompt_tokens=sum(prompt_lengths),
                          completion_tokens=sum(answer_lengths),
                          mean_completion_tokens=fmean(answer_lengths) if answer_lengths else None,
                          max_completion_tokens=max(answer_lengths) if answer_lengths else None)
    return dict(run_id=run_id, phase=phase, steps=steps, include_inherited=include_inherited,
                inherited_rows=inherited, unscored_rows=unscored, partial_batch_rows=partial,
                generation=generation,
                mean_reward=fmean(r['reward'] for r in rows) if rows else None,
                **outcomes.summarize(rows))
