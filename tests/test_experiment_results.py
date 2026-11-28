import hashlib
import json
from pathlib import Path

import pytest

from rewardhack import archive, experiment_results, jsonl, outcomes
from rewardhack.provenance import VIEWS


RUN = 'a' * 32
ARCHIVE = '20261127T230000Z-run-01234567'


@pytest.fixture
def recorded(tmp_path):
    directory = tmp_path / RUN
    result = directory / 'checkout/results' / RUN
    task = dict(id='t', group='g', tier='hard', true=False, statement='False')
    scored = outcomes.score(None, task, 'unfinished', 'baseline', truncated=True)
    rows = [dict(scored, step=1, copied_from='parent'), dict(scored, step=2, partial_batch=True)]
    generated = dict(run_id=RUN, phase='training', step=2, task_id='t', sample_index=0,
                     prompt_token_ids=[2, 3, 4], completion_token_ids=[5, 6], output='unfinished')
    index = dict(run_id=RUN, archive_id=ARCHIVE, uri=f's3://{archive.BUCKET}/{ARCHIVE}/', backup_status='verified',
                 outputs={k:{} for k in VIEWS}, generations={k:{} for k in VIEWS})
    for stream, name, data in [
        ('outputs', 'arm/training-rollouts.jsonl', rows),
        ('outputs', 'arm/training-rollouts.rescored.jsonl', rows),
        ('generations', 'arm/training-generations.jsonl', [generated]),
    ]:
        path = result / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(jsonl.dumps(data))
        index[stream]['training'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / 'outputs.json').write_text(json.dumps(index))
    return directory, result, index


def test_phase_metrics_and_raw_examples(recorded):
    directory, _, _ = recorded
    state = directory.parent
    summary = experiment_results.summarize_experiment(RUN, 'training', state=state)
    assert summary['samples'] == summary['partial_batch_rows'] == summary['inherited_rows'] == 1
    assert summary['verified_refutation_false']['of'] == 1
    assert summary['truncated_rate'] == 1 and summary['mean_reward'] == 0
    assert summary['generation']['completion_tokens'] == 2
    assert summary['generation']['prompt_tokens'] == 3
    assert len(list(experiment_results.experiment_rows(RUN, 'training', include_inherited=True, state=state))) == 2
    tokens, = experiment_results.experiment_rows(RUN, 'training', stream='generations', steps=(2, 2), state=state)
    assert tokens['completion_token_ids'] == [5, 6]
    assert list(experiment_results.experiment_rows(RUN, 'training', steps=(1, 1), state=state)) == []
    with pytest.raises(ValueError, match='training-step'):
        experiment_results.summarize_experiment(RUN, 'after', steps=(1, 1), state=state)
    with pytest.raises(FileNotFoundError, match='no archived'):
        experiment_results.summarize_experiment(RUN, 'probe', state=state)


def test_s3_fetches_only_requested_view_and_detects_corruption(recorded, monkeypatch, tmp_path):
    _, result, index = recorded
    state = tmp_path / 'empty'
    requests = []
    def fetch(_, command, remote, *args, timeout):
        requests.append((command, remote))
        if command == 'cat':
            return json.dumps(index)
        assert command == 'copyto'
        name = remote.split(ARCHIVE + '/')[1]
        Path(args[0]).write_bytes((result / name).read_bytes())
        return ''
    monkeypatch.setattr(archive, 'run', fetch)
    assert len(list(experiment_results.experiment_rows(RUN, 'training', state=state))) == 1
    assert requests == [('cat', f's3:{archive.BUCKET}/experiments/{RUN}/outputs.json'),
                        ('copyto', f's3:{archive.BUCKET}/{ARCHIVE}/arm/training-rollouts.rescored.jsonl')]
    (result / 'arm/training-rollouts.rescored.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='digest'):
        list(experiment_results.experiment_rows(RUN, 'training', state=state))


def test_pending_backup_reads_verified_local_rows(recorded):
    directory, result, index = recorded
    index.update(backup_status='pending', archive_id=None, uri=None)
    (directory / 'outputs.json').write_text(json.dumps(index))
    assert experiment_results.summarize_experiment(RUN, 'training', state=directory.parent)['samples'] == 1
    (result / 'arm/training-rollouts.rescored.jsonl').unlink()
    with pytest.raises(FileNotFoundError, match='backup is pending'):
        list(experiment_results.experiment_rows(RUN, 'training', state=directory.parent))
