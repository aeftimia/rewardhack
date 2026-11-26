"""Rows carry their provenance, and the loader turns results into per-run counts for the hierarchical prior (owner, 26 Nov;
notes section 77). Older runs take theirs from what their summary.json states, and nothing it does not state is assumed."""
import json

import pytest

from rewardhack import jsonl, provenance

TRAINED = dict(model='openai/gpt-oss-20b', revision='abc', arm='report_correct', penalty=0.5, stage='training',
               report_mode='prose', taskset='hard', variant='neutral', template={'reasoning_effort': 'low'}, run='train-lambda05')


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
