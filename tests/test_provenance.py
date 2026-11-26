"""Rows carry their provenance, and the loader turns results into per-run counts for the hierarchical prior (owner, 26 Nov;
notes section 77). Older runs take theirs from what their summary.json states, and nothing it does not state is assumed."""
import json
from pathlib import Path

import pytest

from rewardhack import jsonl, provenance

TRAINED = dict(model='openai/gpt-oss-20b', revision='abc', arm='report_correct', penalty=0.5, stage='training',
               report_mode='prose', taskset='hard', tasks='all', part='train', variant='neutral', template={'reasoning_effort': 'low'}, run='train-lambda05')


def row(**outcome):
    return dict(dict(genuine=False, verified_refutation=None, exploit=False), **outcome)


def test_a_trained_row_needs_its_arm_and_an_untrained_one_has_none():
    assert provenance.meta(**TRAINED)['penalty'] == 0.5
    with pytest.raises(ValueError, match='exactly when its model is trained'):
        provenance.meta(**dict(TRAINED, stage='untrained'))
    with pytest.raises(ValueError, match='exactly when its model is trained'):
        provenance.meta(**dict(TRAINED, arm=None, penalty=None))
    assert provenance.meta(**dict(TRAINED, stage='untrained', arm=None, penalty=None))['arm'] is None
    for bad in (dict(arm='nonsense'), dict(variant='nonsense'), dict(model='')):
        with pytest.raises(ValueError):
            provenance.meta(**dict(TRAINED, **bad))
    with pytest.raises(KeyError):
        provenance.meta(**dict(TRAINED, stage='nonsense'))


def test_stamped_runs_are_counted_per_run_with_their_groups(tmp_path):
    run = tmp_path/'train-lambda05'
    run.mkdir()
    meta = provenance.meta(**TRAINED)
    rows = [row(genuine=True), row(verified_refutation='verified'), row(exploit=True), row(exploit=True)]
    rows += [dict(row(exploit=True), partial_batch=True), dict(row(exploit=True), copied_from='elsewhere')]  # counted elsewhere
    (run/'training-rollouts.jsonl').write_text(jsonl.dumps(provenance.stamp(rows, meta)))
    (run/'steps.jsonl').write_text(jsonl.dumps([dict(step=1, mixed_groups=1, zero_advantage_groups=3, meta=meta)]))
    [entry], refused = provenance.cells(tmp_path)
    assert refused == []
    assert (entry['answers'], entry['genuine'], entry['refutation'], entry['exploit']) == (4, 1, 1, 2)
    assert (entry['groups'], entry['mixed_groups']) == (4, 1)
    assert (entry['model'], entry['arm'], entry['penalty'], entry['run']) == ('openai/gpt-oss-20b', 'report_correct', 0.5, 'train-lambda05')


def test_an_older_run_is_read_from_what_its_summary_states_and_refused_when_it_states_too_little(tmp_path):
    old = tmp_path/'old'/'probe-think'
    old.mkdir(parents=True)
    (old/'rows.rescored.jsonl').write_text(jsonl.dumps([row(exploit=True), row()]))
    summary = dict(argv=['--model', 'Qwen/Qwen3-8B', '--revision', 'r1', '--taskset', 'hard', '--variant', 'neutral'],
                   template_kwargs={'enable_thinking': True})
    (old/'summary.json').write_text(json.dumps(summary))  # no report_mode: older than report modes, so prose
    vague = tmp_path/'vague'/'probe'
    vague.mkdir(parents=True)
    (vague/'rows.rescored.jsonl').write_text(jsonl.dumps([row()]))
    (vague/'summary.json').write_text(json.dumps(dict(argv=['--samples', '2'])))
    [entry], refused = provenance.cells(tmp_path)
    assert (entry['model'], entry['stage'], entry['arm'], entry['report_mode'], entry['exploit']) == ('Qwen/Qwen3-8B', 'untrained', None, 'prose', 1)
    [(path, why)] = refused
    assert path.endswith('vague/probe/rows.rescored.jsonl') and 'model, revision' in why


def test_a_file_that_fails_part_way_adds_nothing(tmp_path):
    run = tmp_path/'train-x'
    run.mkdir()
    (run/'training-rollouts.jsonl').write_text(jsonl.dumps(provenance.stamp([row()], provenance.meta(**TRAINED))))
    # no steps.jsonl: the rows were readable, the groups are not
    table, [(_, why)] = provenance.cells(tmp_path)
    assert table == [] and 'steps.jsonl' in why


def test_rescoring_keeps_a_rows_provenance():
    from rewardhack.rescore import CARRIED
    assert 'meta' in CARRIED


UNTRAINED = dict(TRAINED, stage='untrained', arm=None, penalty=None, tasks='false', part='train', run='probe-120b-low')


def write(path, rows, **summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonl.dumps(rows))
    if summary:
        (path.parent/'summary.json').write_text(json.dumps(summary))


def test_a_run_is_its_path_so_same_named_runs_of_different_jobs_stay_apart(tmp_path):
    meta = provenance.meta(**TRAINED)
    for job in ('job-a', 'job-b'):
        write(tmp_path/job/'train-baseline'/'before-evaluation.jsonl', provenance.stamp([row()], dict(meta, stage='before', part='evaluation')))
    table, refused = provenance.cells(tmp_path)
    assert refused == [] and sorted(entry['run'] for entry in table) == ['job-a/train-baseline', 'job-b/train-baseline']


def test_one_view_per_stage_is_read_and_every_other_jsonl_is_listed(tmp_path):
    probe = tmp_path/'api'/'mini'
    meta = provenance.meta(**UNTRAINED)
    write(probe/'rows.ok.rescored.jsonl', provenance.stamp([row(verified_refutation='verified')], meta))
    write(probe/'rows.rescored.jsonl', provenance.stamp([row(), dict(row(), error='timeout')], meta))
    write(probe/'rows.jsonl', [dict(task_id='x', raw='', scored=False, meta=meta)])
    write(tmp_path/'mlx'/'rollouts.jsonl', [row()])
    write(tmp_path/'raw'/'rows.jsonl', [dict(task_id='x', raw='', scored=False, meta=meta)])
    [entry], refused = provenance.cells(tmp_path)
    assert (entry['run'], entry['answers'], entry['refutation'], entry['tasks'], entry['part']) == ('api/mini', 1, 1, 'false', 'train')
    why = {str(Path(path).relative_to(tmp_path)): reason for path, reason in refused}
    assert why.keys() == {'api/mini/rows.rescored.jsonl', 'api/mini/rows.jsonl', 'mlx/rollouts.jsonl', 'raw/rows.jsonl'}
    assert 'rows.ok.rescored.jsonl is the preferred view' in why['api/mini/rows.jsonl']
    assert 'not a row file of any stage' in why['mlx/rollouts.jsonl']
    assert 'unscored rows' in why['raw/rows.jsonl']


def test_schema_1_rows_take_tasks_and_part_from_the_stage_or_the_summary(tmp_path):
    old = {name: value for name, value in provenance.meta(**TRAINED).items() if name not in provenance.SELECTION}
    old['schema'] = 1
    write(tmp_path/'job'/'train'/'after-evaluation.jsonl', provenance.stamp([row()], dict(old, stage='after')))
    probe_meta = {name: value for name, value in provenance.meta(**UNTRAINED).items() if name not in provenance.SELECTION}
    write(tmp_path/'job'/'probe-120b-low'/'rows.rescored.jsonl', provenance.stamp([row()], dict(probe_meta, schema=1)),
          argv=['--tasks', 'false', '--max-seconds', '1200'], part='train', tasks=80)  # `tasks` here is a count, not the selection
    write(tmp_path/'job'/'probe-default'/'rows.rescored.jsonl', provenance.stamp([row()], dict(probe_meta, schema=1)), argv=[])
    write(tmp_path/'job'/'probe-no-argv'/'rows.rescored.jsonl', provenance.stamp([row()], dict(probe_meta, schema=1)), part='all')
    table, [(path, why)] = provenance.cells(tmp_path)
    found = {entry['run']: (entry['tasks'], entry['part']) for entry in table}
    assert found == {'job/train': ('all', 'evaluation'), 'job/probe-120b-low': ('false', 'train'), 'job/probe-default': ('all', 'train')}
    assert path.endswith('probe-no-argv/rows.rescored.jsonl') and 'no argv' in why


def test_the_trainers_stages_fix_their_split_and_a_probe_of_an_adapter_is_not_untrained(tmp_path):
    with pytest.raises(ValueError, match='a before row samples'):
        provenance.meta(**dict(TRAINED, stage='before', part='train'))
    with pytest.raises(ValueError, match='unknown task selection'):
        provenance.meta(**dict(UNTRAINED, tasks='some'))
    trained_probe = tmp_path/'mlx-probe'
    write(trained_probe/'rows.rescored.jsonl', [row()], argv=['--model', 'm', '--revision', 'r', '--taskset', 'v1', '--variant', 'neutral',
                                                              '--adapter', 'runs/x/adapter.safetensors'], template_kwargs={})
    table, [(_, why)] = provenance.cells(tmp_path)
    assert table == [] and 'names an adapter' in why


def test_the_transcript_export_prefers_the_same_views(tmp_path):
    from rewardhack.export_transcripts import row_files
    for name in ('rows.jsonl', 'rows.rescored.jsonl', 'rows.ok.rescored.jsonl', 'training-rollouts.jsonl', 'training-rollouts.rescored.jsonl'):
        (tmp_path/name).write_text('')
    assert [path.name for path, _ in row_files(tmp_path)] == ['rows.ok.rescored.jsonl', 'training-rollouts.rescored.jsonl']
