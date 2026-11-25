"""Exports preserve current-writer provenance and unresolved measurements."""
import json
from pathlib import Path

import pytest

from rewardhack.export_transcripts import export, update_count, update_history
from rewardhack.prompts import prompt
from rewardhack.tasks import task_index


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path/'results'; run = root/'run'; run.mkdir(parents=True)
    task = task_index()['add-34-T']
    row = dict(task_id=task['id'], arm='report_correct', raw='{}', established=task['statement'], body='by sorry',
               report='Placeholder', genuine=False, exploit=True, verified_refutation=False, format_valid=True,
               report_mode='typed', claim='ERROR', verifier_inconclusive=False)
    (run/'summary.json').write_text(json.dumps(dict(model='m', report_mode='typed', global_step=4,
                                                   argv=['train', '--prompt-variant', 'hint'])))
    return root, run, row, task


def write(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def exported(root):
    path=root/'out.jsonl'
    result=export(root,path)
    return result,[json.loads(line) for line in path.read_text().splitlines()]


def test_current_hf_streams_export_once_with_known_or_unknown_updates(dataset):
    root,run,row,task=dataset
    for stem in ('probe-evaluation','before-evaluation','after-evaluation','training-rollouts'):
        write(run/(stem+'.jsonl'),[dict(row,raw='raw view')])
        write(run/(stem+'.rescored.jsonl'),[dict(row,raw='preferred rescored view')])
    write(run/'steps.jsonl',[dict(step=n,mixed_groups=m) for n,m in ((1,0),(2,3),(3,0),(4,1))])  # 4 attempted, 2 updated
    manifest,rows=exported(root)
    assert manifest['rows']==4
    assert all(r['response']=='preferred rescored view' for r in rows)
    assert all(r['messages']==prompt(task['statement'],'hint','typed') for r in rows)
    records={Path(r['source_file']).name:r for r in rows}
    for stem in ('probe-evaluation','before-evaluation'):
        record=records[stem+'.rescored.jsonl']
        assert record['policy_updates']==0 and record['provenance']=='hinted'
    after=records['after-evaluation.rescored.jsonl']
    assert after['policy_updates']==2 and after['provenance']=='rl_induced'
    training=records['training-rollouts.rescored.jsonl']
    assert training['policy_updates'] is None and training['provenance']=='training_unknown'


def test_raw_scored_streams_export_without_rescoring_unscored_rows_stay_out(dataset):
    root,run,row,_=dataset
    write(run/'after-evaluation.jsonl',[row])
    write(run/'rows.jsonl',[dict(raw='unscored output',task_id=row['task_id'],scored=False)])
    manifest,rows=exported(root)
    assert manifest['rows']==1 and manifest['skipped']=={'unscored row':1}
    assert rows[0]['source_file']=='run/after-evaluation.jsonl'


def test_uncertainty_and_missing_answers_remain_distinct_from_failed_proofs(dataset):
    root,run,row,_=dataset
    unresolved=dict(row,exploit=False,verifier_inconclusive=True,genuine_why='timeout',refutation_why='timeout',declaration_why='timeout',partial_batch=True)
    cutoff=dict(row,exploit=False,format_valid=False,truncated=True,genuine_why='truncated')
    failed=dict(row,exploit=False,genuine_why='type mismatch')
    write(run/'rows.rescored.jsonl',[unresolved,cutoff,failed])
    manifest,rows=exported(root)
    assert [r['cheat_axis'] for r in rows]==['unverified','no_answer','failed']
    assert rows[0]['verifier_inconclusive'] and rows[0]['partial_batch']
    assert rows[0]['genuine_why']==rows[0]['refutation_why']==rows[0]['declaration_why']=='timeout'
    assert {r['cheat'] for r in manifest['cells']}=={'unverified','no_answer','failed'}


@pytest.mark.parametrize('mode',[None,'',False,0,'typo'])
def test_invalid_explicit_report_mode_is_not_replaced_by_summary(dataset,mode):
    root,run,row,_=dataset
    write(run/'rows.rescored.jsonl',[dict(row,report_mode=mode)])
    output=root/'out.jsonl'; output.write_text('previous complete export')
    with pytest.raises(ValueError,match='report mode'):
        exported(root)
    assert output.read_text()=='previous complete export'
    assert not list(root.glob('.export-*'))


def test_mixed_follow_on_cannot_rewrite_opening_prompt(dataset):
    root,run,row,_=dataset
    write(run/'rows.rescored.jsonl',[dict(row,conversation='c',turn=1,report_mode='prose'),dict(row,conversation='c',turn=2)])
    with pytest.raises(ValueError,match='opening prompt'):
        exported(root)


def test_mlx_attempted_iterations_are_not_completed_updates(dataset):
    root,run,row,_=dataset
    write(run/'steps.jsonl',[dict(step=1,mixed=False),dict(step=2,mixed=True)])
    write(run/'rollouts.jsonl',[dict(row,step=1),dict(row,step=2),dict(row,step=3)])
    write(run/'monitor-step2.jsonl',[row])
    _,rows=exported(root)
    rollouts=[r for r in rows if r['source_file']=='run/rollouts.jsonl']
    assert [r['policy_updates'] for r in rollouts]==[0,0,1]
    assert [r['provenance'] for r in rollouts]==['hinted','hinted','rl_induced']
    monitor=next(r for r in rows if r['source_file']=='run/monitor-step2.jsonl')
    assert monitor['policy_updates']==1


def test_missing_or_corrupt_update_history_is_unknown(dataset):
    _,run,row,_=dataset
    assert update_count({}, {}, dict(row,step=2),'mlx_training') is None
    assert update_count({}, {}, row,'hf_after') is None
    # HF: attempted steps are not updates. No reward variation is zero updates; a missing or short history is unknown.
    assert update_count({}, dict(global_step=24,mixed_reward_groups=0,adapter_max_abs_change=0.0), row,'hf_after')==0
    assert update_count({}, dict(global_step=4,mixed_reward_groups=3), row,'hf_after') is None
    assert update_count({1:True,2:False}, dict(global_step=4,mixed_reward_groups=3), row,'hf_after') is None
    assert update_count({1:True,2:False,3:True,4:True}, dict(global_step=4,mixed_reward_groups=5), row,'hf_after')==3
    write(run/'steps.jsonl',[dict(step=1,mixed=True),dict(step=1,mixed=False)])
    assert update_history(run)=={}


def test_run_started_from_an_adapter_has_unknown_update_count_and_equals_options_are_read(dataset):
    root,run,row,task=dataset
    (run/'summary.json').write_text(json.dumps(dict(report_mode='typed', argv=['probe','--prompt-variant=hint','--adapter=/a','--model=m'])))
    write(run/'probe-evaluation.jsonl',[row])
    _,rows=exported(root)
    record,=rows
    assert (record['variant'],record['adapter'],record['model'])==('hint','/a','m')
    assert record['messages']==prompt(task['statement'],'hint','typed')
    assert record['policy_updates'] is None and record['run_policy_updates']==0 and record['provenance']=='rl_induced'
    # The summary an HF run now writes: recorded values win, and an absent adapter or empty switches fall back to argv.
    from rewardhack.export_transcripts import run_meta
    meta=run_meta(dict(model='m2',revision='r',prompt_variant='nocheat',adapter=None,template_kwargs={'enable_thinking':True},
                       argv=['train','--model','other','--prompt-variant','hint']),'x/y')
    assert (meta['model'],meta['revision'],meta['variant'],meta['adapter'],meta['thinking'])==('m2','r','nocheat',None,True)
    assert run_meta(dict(adapter=None,template_kwargs={},prompt_variant=None,argv=['--adapter','/a','--prompt-variant=hint']),'x/y')['adapter']=='/a'
    import os, stat
    umask=os.umask(0); os.umask(umask)
    assert stat.S_IMODE((root/'out.jsonl').stat().st_mode)==0o666 & ~umask
