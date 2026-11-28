"""Publish run configurations, generated examples and checkpoint locations under an experiment ID."""
import re
import tempfile
from pathlib import Path

from . import archive, gates, jsonl
from .experiment_backups import snapshot_step
from .provenance import VIEWS
from .run_registry import CONFIG_FILE, checked_source, read, run_id_directory
from .state_files import atomic_json


def register_config(directory, source):
    """Publish the configuration and original request before the pod starts billing."""
    config = read(directory / CONFIG_FILE)
    if config['run_id'] != directory.name:
        raise ValueError('prepared configuration does not identify its registry directory')
    request = read(directory / 'request.json')
    return _publish_index(directory, 'config.json', dict(run_id=directory.name, source=source, config=config,
                                                         request=request))


def _publish_index(directory, name, index):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        pending = Path(scratch) / path.name
        atomic_json(pending, index)
        archive.publish_record(pending, f's3:{archive.BUCKET}/experiments/{directory.name}/{name}')
    atomic_json(path, index, exclusive=False)
    return index


def _payload_receipt(folder, commit):
    record = folder / archive.RECORD
    return archive.archive_entry(str(record)) if record.exists() else archive.upload(folder, commit)


def archive_snapshot(directory, name):
    """Publish the verified adapter for this run and step before the final archive is ready."""
    step = snapshot_step(name)
    directory = Path(directory)
    source = checked_source(directory)
    receipt = _payload_receipt(directory / 'snapshots' / f'{directory.name}-{name}', source['commit'])
    index = dict(run_id=directory.name, step=step, kind='adapter',
                 prepared_config_sha256=source['config_sha256'], source_commit=source['commit'],
                 uri=receipt['uri'], archive_id=receipt['run_id'], sha256=receipt['sha256'])
    return _publish_index(directory, f'checkpoints/{name}.json', index)


def validate_generation_links(directory, index):
    """Check successful attempts before the next job can spend. Scoring may finish after sampling."""
    if index['missing_artifacts']:
        raise ValueError(f"{directory.name}: missing experiment artifacts {index['missing_artifacts']}")
    result = directory / 'checkout/results' / directory.name
    for phase in ('before', 'training', 'after'):
        identifiers = set()
        for path in index['generations'][phase]:
            for row in jsonl.read(result / path):
                key = row['generation_id']
                if row['run_id'] != directory.name or row['phase'] != phase or key in identifiers:
                    raise ValueError(f'{path}: wrong run, phase or duplicate generation ID {key}')
                identifiers.add(key)
        for path in index['outputs'][phase]:
            if Path(path).name == VIEWS[phase][-1]:
                for row in jsonl.read(result / path):
                    if 'copied_from' not in row and (row['run_id'] != directory.name or row['generation_id'] not in identifiers):
                        raise ValueError(f'{path}: scored answer has no matching generation in this run')


def collect_outputs(directory):
    """Expose finished local stage files immediately, before the large checkpoint transfer."""
    config = read(directory / CONFIG_FILE)
    source = checked_source(directory)
    run_id = directory.name
    if run_id_directory(directory.parent, config['run_id']) != directory:
        raise ValueError('registry directory differs from the experiment ID')
    result = directory / 'checkout/results' / run_id
    provenance = {'run_id': run_id, 'source': source, 'config': config, 'request': read(directory / 'request.json')}
    atomic_json(result / 'experiment-source.json', provenance, exclusive=False)
    arm = config['output_arm']
    effective = result / arm / CONFIG_FILE
    if effective.is_file() and dict(read(effective), settings=config['settings']) != config:
        raise ValueError('effective config differs from the prepared experiment')
    paths = list(result.glob('*/*.jsonl'))
    def streams(names):
        return {p.relative_to(result).as_posix(): gates.file_sha256(p) for p in paths if p.name in names}
    stages = {stage: streams(names) for stage, names in VIEWS.items()}
    generations = {stage: streams((f'{stage}-generations.jsonl',)) for stage in VIEWS}
    missing = []
    for stage in ('before', 'training', 'after'):
        raw = result / arm / VIEWS[stage][-1]
        if not raw.is_file():
            missing.append(raw.relative_to(result).as_posix())
        elif raw.stat().st_size and not generations[stage] and not (stage == 'before' and config['parent']):
            missing.append(f'{arm}/{stage}-generations.jsonl')
    if not effective.is_file():
        missing.append(f'{arm}/{CONFIG_FILE}')
    exit_path = result / 'EXIT_CODE'
    index = dict(run_id=run_id, backup_status='pending', archive_id=None, uri=None,
                 exit_code=int(exit_path.read_text()) if exit_path.is_file() else None,
                 source_commit=source['commit'], prepared_config_sha256=source['config_sha256'],
                 provenance='experiment-source.json', effective_config=f'{arm}/{CONFIG_FILE}' if effective.is_file() else None,
                 effective_config_sha256=gates.file_sha256(effective) if effective.is_file() else None,
                 outputs=stages, generations=generations, checkpoints={}, missing_artifacts=missing)
    atomic_json(directory / 'outputs.json', index)
    return index


def archive_outputs(directory):
    """Publish the full verified result and its checkpoint inventory under the same run ID."""
    directory = Path(directory)
    index = read(directory / 'outputs.json')
    source = checked_source(directory)
    result = directory / 'checkout/results' / directory.name
    receipt = _payload_receipt(result, source['commit'])
    files = receipt['sha256']
    if (index['run_id'] != directory.name or receipt['commit'] != source['commit']
            or files[index['provenance']] != gates.file_sha256(result / index['provenance'])):
        raise ValueError('archive receipt differs from the experiment provenance')
    expected = {p: digest for stream in ('outputs', 'generations') for group in index[stream].values()
                for p, digest in group.items()}
    if index['effective_config'] is not None:
        expected[index['effective_config']] = index['effective_config_sha256']
    if any(files.get(path) != digest for path, digest in expected.items()):
        raise ValueError('archived output or effective config differs from the local registry')
    arm = read(directory / CONFIG_FILE)['output_arm']
    checkpoints = {}
    for path, digest in files.items():
        match = re.fullmatch(re.escape(arm) + r'/(trainer/checkpoint-|adapter-step)([0-9]+)/(.+)', path)
        if match:
            kind = {'trainer/checkpoint-':'trainer', 'adapter-step':'adapter'}[match[1]]
            checkpoints.setdefault(kind, {}).setdefault(match[2], {})[path] = digest
    index.update(backup_status='verified', archive_id=receipt['run_id'], uri=receipt['uri'], checkpoints=checkpoints)
    return _publish_index(directory, 'outputs.json', index)
