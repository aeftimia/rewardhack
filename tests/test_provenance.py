"""Rows carry their provenance, and the loader turns results into per-run counts for the hierarchical prior (owner, 26 Nov;
notes section 77). Older runs take theirs from what their summary.json states, and nothing it does not state is assumed."""
import json
from pathlib import Path

import pytest

from rewardhack import jsonl, provenance

TRAINED = dict(model='openai/gpt-oss-20b', revision='abc', arm='report_correct', penalty=0.5, stage='training',
               report_mode='prose', taskset='hard', tasks='all', part='train', variant='neutral', template={'reasoning_effort': 'low'}, run='train-lambda05')


def row(**outcome):
    return dict(dict(true=False, genuine=False, verified_refutation=None, exploit=False), **outcome)


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
    rows = [row(true=True, genuine=True), row(verified_refutation='verified'), row(true=True, exploit=True), row(exploit=True),
            row(true=True, exploit=False, verifier_inconclusive=True)]  # undecided: not among the exploit rate's rows
    rows += [dict(row(exploit=True), partial_batch=True), dict(row(exploit=True), copied_from='elsewhere')]  # counted elsewhere
    rows = [dict(each, step=1) for each in rows] + [dict(row(verified_refutation='verified'), step=2)]  # step 2: never trained on
    (run/'training-rollouts.jsonl').write_text(jsonl.dumps(provenance.stamp(rows, meta)))
    (run/'steps.jsonl').write_text(jsonl.dumps([dict(step=1, mixed_groups=1, zero_advantage_groups=3, meta=meta),
                                                dict(step=2, mixed_groups=1, zero_advantage_groups=0, meta=meta)]))
    (run/'summary.json').write_text(json.dumps(dict(status='failed', untrained_step=2, global_step=1)))
    [entry], refused = provenance.cells(tmp_path)
    assert refused == []
    assert entry['answers'] == 5
    assert (entry['genuine_true'], entry['genuine_true_of']) == (1, 3)  # over true statements
    assert (entry['verified_refutation_false'], entry['verified_refutation_false_of']) == (1, 2)  # over false ones
    assert (entry['exploit'], entry['exploit_of']) == (2, 4)  # over rows the verifier decided
    assert (entry['groups'], entry['mixed_groups']) == (4, 1)
    assert (entry['model'], entry['arm'], entry['penalty'], entry['run']) == ('openai/gpt-oss-20b', 'report_correct', 0.5, 'train-lambda05')


def test_a_row_file_with_nothing_of_its_own_is_listed_and_an_abbreviated_flag_is_refused(tmp_path):
    meta = provenance.meta(**dict(TRAINED, stage='before', part='evaluation'))
    write(tmp_path/'train-lambda05'/'before-evaluation.jsonl', provenance.stamp([dict(row(), copied_from='x')], meta), status='completed')
    write(tmp_path/'probe'/'rows.rescored.jsonl', [row()], argv=['--model', 'm', '--revision', 'r', '--taskset', 'v1',
                                                                 '--variant', 'neutral', '--task', 'false'], template_kwargs={})
    table, refused = provenance.cells(tmp_path)
    why = {str(Path(path).relative_to(tmp_path)): reason for path, reason in refused}
    assert table == [] and 'nothing of its own to count' in why['train-lambda05/before-evaluation.jsonl']
    assert 'as --task, which argparse may have read' in why['probe/rows.rescored.jsonl']


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
    run = tmp_path/'train-lambda05'
    run.mkdir()
    (run/'training-rollouts.jsonl').write_text(jsonl.dumps(provenance.stamp([dict(row(), step=1)], provenance.meta(**TRAINED))))
    (run/'summary.json').write_text(json.dumps(dict(status='completed')))
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
    meta = provenance.meta(**dict(TRAINED, run='train-baseline'))
    for job in ('job-a', 'job-b'):
        write(tmp_path/job/'train-baseline'/'before-evaluation.jsonl', provenance.stamp([row()], dict(meta, stage='before', part='evaluation')), status='completed')
    table, refused = provenance.cells(tmp_path)
    assert refused == [] and sorted(entry['run'] for entry in table) == ['job-a/train-baseline', 'job-b/train-baseline']


def test_one_view_per_stage_is_read_and_every_other_jsonl_is_listed(tmp_path):
    probe = tmp_path/'api'/'probe-120b-low'
    meta = provenance.meta(**UNTRAINED)
    write(probe/'rows.ok.rescored.jsonl', provenance.stamp([row(verified_refutation='verified')], meta))
    write(probe/'rows.rescored.jsonl', provenance.stamp([row(), dict(row(), error='timeout')], meta))
    write(probe/'rows.jsonl', [dict(task_id='x', raw='', scored=False, meta=meta)])
    write(tmp_path/'mlx'/'rollouts.jsonl', [row()])
    write(tmp_path/'raw'/'rows.jsonl', [dict(task_id='x', raw='', scored=False, meta=dict(meta, run='raw'))])
    [entry], refused = provenance.cells(tmp_path)
    assert (entry['run'], entry['answers'], entry['verified_refutation_false'], entry['tasks'], entry['part']) == ('api/probe-120b-low', 1, 1, 'false', 'train')
    why = {str(Path(path).relative_to(tmp_path)): reason for path, reason in refused}
    assert why.keys() == {'api/probe-120b-low/rows.rescored.jsonl', 'api/probe-120b-low/rows.jsonl', 'mlx/rollouts.jsonl', 'raw/rows.jsonl'}
    assert 'rows.ok.rescored.jsonl is the preferred view' in why['api/probe-120b-low/rows.jsonl']
    assert 'not a row file of any stage' in why['mlx/rollouts.jsonl']
    assert 'unscored rows' in why['raw/rows.jsonl']


def test_schema_1_rows_take_tasks_and_part_from_the_stage_or_the_summary(tmp_path):
    old = {name: value for name, value in provenance.meta(**TRAINED).items() if name not in provenance.SELECTION}
    old['schema'] = 1
    write(tmp_path/'job'/'train-lambda05'/'after-evaluation.jsonl', provenance.stamp([row()], dict(old, stage='after')), status='completed')
    probe_meta = {name: value for name, value in provenance.meta(**UNTRAINED).items() if name not in provenance.SELECTION}
    write(tmp_path/'job'/'probe-120b-low'/'rows.rescored.jsonl', provenance.stamp([row()], dict(probe_meta, schema=1)),
          argv=['--tasks', 'false', '--max-seconds', '1200'], part='train', tasks=80)  # `tasks` here is a count, not the selection
    write(tmp_path/'job'/'probe-default'/'rows.rescored.jsonl', provenance.stamp([row()], dict(probe_meta, schema=1, run='probe-default')), argv=[])
    write(tmp_path/'job'/'probe-no-argv'/'rows.rescored.jsonl', provenance.stamp([row()], dict(probe_meta, schema=1, run='probe-no-argv')), part='all')
    table, [(path, why)] = provenance.cells(tmp_path)
    found = {entry['run']: (entry['tasks'], entry['part']) for entry in table}
    assert found == {'job/train-lambda05': ('all', 'evaluation'), 'job/probe-120b-low': ('false', 'train'), 'job/probe-default': ('all', 'train')}
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


def test_flags_read_as_argparse_reads_them_and_a_copied_run_is_not_counted_twice(tmp_path):
    assert provenance.flag(['--part', 'train', '--samples', '2', '--part', 'all'], 'part') == 'all'  # the last one counts
    assert provenance.flag(['--part=train', '--part', 'evaluation'], 'part') == 'evaluation'
    with pytest.raises(ValueError, match='ends with --tasks'):
        provenance.flag(['--tasks'], 'tasks')
    argv = ['--model', 'm', '--revision', 'r', '--taskset', 'v1', '--variant', 'neutral', '--out', 'results/job/child']
    write(tmp_path/'job'/'child'/'rows.rescored.jsonl', [row()], argv=argv, template_kwargs={})
    write(tmp_path/'job'/'rows.rescored.jsonl', [row()], argv=argv, template_kwargs={})  # a copy in the parent folder
    write(tmp_path/'stamped'/'elsewhere'/'rows.rescored.jsonl', provenance.stamp([row()], provenance.meta(**UNTRAINED)))
    write(tmp_path/'bare'/'rows.rescored.jsonl', [row()], argv=argv[:-2] + ['--part'], template_kwargs={})
    [entry], refused = provenance.cells(tmp_path)
    assert entry['run'] == 'job/child'
    why = {str(Path(path).relative_to(tmp_path)): reason for path, reason in refused}
    assert "named 'child'" in why['job/rows.rescored.jsonl'] and "named 'probe-120b-low'" in why['stamped/elsewhere/rows.rescored.jsonl']
    assert 'ends with --part' in why['bare/rows.rescored.jsonl']


def test_the_flags_read_and_the_whole_options_among_them_are_stated_from_the_code():
    import re
    root = Path(__file__).resolve().parents[1]
    source = (root/'src/rewardhack/provenance.py').read_text()
    # Every read of a flag, whatever expression gives the summary or its argv.
    read = set(re.findall(r"stated\([^,()]+(?:\([^()]*\))?, '[a-z_]+', '([a-z-]+)'\)|flag\([^,]+, '([a-z-]+)'\)", source))
    assert {name for pair in read for name in pair if name} == set(provenance.READ_FLAGS)
    options = {option for path in [*root.glob('entrypoints/*.py'), *root.glob('src/rewardhack/*.py')]
               for option in re.findall(r"add_argument\('(--[a-z0-9-]+)'", path.read_text())}
    prefixes = {option for option in options for name in provenance.READ_FLAGS if option != f'--{name}' and f'--{name}'.startswith(option)}
    assert prefixes == provenance.WHOLE_OPTIONS
    assert provenance.flag(['--tasks', 'false', '--taskset', 'hard'], 'taskset') == 'hard'
    assert provenance.flag(['--tasks', 'false'], 'taskset') is None


def test_a_run_that_did_not_complete_counts_only_the_steps_it_trained_on(tmp_path):
    """A step is scored (rows and steps.jsonl line written) before its update, so an unfinished run's last step was never
    trained on. A run killed from outside writes no final summary: its training-log.json says which steps were trained,
    and without it the file is refused (review round 34)."""
    meta = provenance.meta(**TRAINED)
    for name, summary, log in (('killed-logged', dict(status='running'), [dict(loss=0.1, step=1), dict(loss=0.2, step=2)]),
                               ('killed-silent', dict(status='running'), None),
                               ('failed-old', dict(status='failed', global_step=1), None)):
        run = tmp_path/name
        rows = [dict(row(exploit=True), step=step) for step in (1, 2, 3)]
        write(run/'training-rollouts.jsonl', provenance.stamp(rows, dict(meta, run=name)), **summary)
        (run/'steps.jsonl').write_text(jsonl.dumps([dict(step=step, mixed_groups=1, zero_advantage_groups=0) for step in (1, 2, 3)])
                                       + '{"step": 4, "mixed_')  # a line a kill cut short
        if log is not None:
            (run/'training-log.json').write_text(json.dumps(log + [dict(train_runtime=9.0, step=2)]))
    table, refused = provenance.cells(tmp_path)
    found = {entry['run']: (entry['answers'], entry['groups']) for entry in table}
    assert found == {'killed-logged': (2, 2), 'failed-old': (1, 1)}
    why = {str(Path(path).relative_to(tmp_path)): reason for path, reason in refused}
    assert 'without saying which steps were trained on' in why['killed-silent/training-rollouts.jsonl']
    assert 'training rows beside it that are counted' in why['killed-silent/steps.jsonl']  # listed, not skipped
    assert why.keys() == {'killed-silent/training-rollouts.jsonl', 'killed-silent/steps.jsonl'}


def test_a_trainer_run_started_from_an_adapter_is_refused(tmp_path):
    meta = provenance.meta(**dict(TRAINED, stage='before', part='evaluation'))
    write(tmp_path/'train-lambda05'/'before-evaluation.jsonl', provenance.stamp([row()], meta),
          status='completed', adapter='results/x/train-lambda05/adapter')
    table, [(_, why)] = provenance.cells(tmp_path)
    assert table == [] and 'started from an adapter' in why


def test_a_resumed_run_is_refused_and_a_copy_is_refused_as_a_copy_before_anything_else(tmp_path):
    meta = provenance.meta(**dict(TRAINED, stage='after', part='evaluation'))
    write(tmp_path/'train-lambda05'/'after-evaluation.jsonl', provenance.stamp([row()], meta),
          status='completed', adapter=None, resumed_from='results/x/train-lambda05/trainer/checkpoint-20')
    argv = ['--model', 'm', '--revision', 'r', '--taskset', 'v1', '--variant', 'neutral', '--out', 'results/job/child']
    write(tmp_path/'job'/'rows.jsonl', [dict(row(), scored=False)], argv=argv, template_kwargs={})  # a raw copy
    table, refused = provenance.cells(tmp_path)
    why = {str(Path(path).relative_to(tmp_path)): reason for path, reason in refused}
    assert table == [] and 'resumed another from a checkpoint' in why['train-lambda05/after-evaluation.jsonl']
    assert "named 'child'" in why['job/rows.jsonl']  # not "rescore them": rescoring a copy would not help


def test_follow_on_turns_and_failed_requests_are_refused_with_their_reason(tmp_path):
    meta = provenance.meta(**dict(UNTRAINED, run='follow'))
    write(tmp_path/'follow'/'rows.rescored.jsonl', provenance.stamp([dict(row(), turn=1), dict(row(), turn=2)], meta))
    write(tmp_path/'errors'/'rows.jsonl', [dict(task_id='x', error='timeout', scored=False)], argv=[])
    table, refused = provenance.cells(tmp_path)
    why = {str(Path(path).relative_to(tmp_path)): reason for path, reason in refused}
    assert table == [] and 'follow-on rows' in why['follow/rows.rescored.jsonl']
    assert 'failed requests' in why['errors/rows.jsonl']


def test_a_broken_link_is_listed_not_a_crash(tmp_path):
    (tmp_path/'run').mkdir()
    (tmp_path/'run'/'rows.jsonl').symlink_to(tmp_path/'missing.jsonl')
    table, [(path, why)] = provenance.cells(tmp_path)
    assert table == [] and path.endswith('run/rows.jsonl') and 'cannot be opened' in why
