"""Rescoring preserves the generation identity required by paired-pilot ingestion."""
import json

from rewardhack import outcomes, rescore


def test_rescore_preserves_pilot_generation_identity_without_inventing_it(tmp_path,monkeypatch):
    task=dict(id='t',group='g',tier='easy',true=True,statement='True')
    raw=json.dumps(dict(established='True',proof_body='by trivial',error=False,justification='Proved it.'))
    provenance=dict(spec_hash='a'*64,seed='seed-1',checkpoint=4,compute=4,condition='typed-intervention',checkpoint_sha256='b'*64)
    with_metadata=dict(task_id='t',arm='report_correct', penalty=0.0,report_mode='typed',raw=raw,sample_index=2,pilot_source=provenance,
                       termination_schema='cutoff-v1',truncated=False)
    legacy=dict(task_id='t',arm='report_correct', penalty=0.0,report_mode='typed',raw=raw)
    monkeypatch.setattr(outcomes,'evaluate_submission',lambda lean,statement,text:dict(json.loads(text),reward=1.,compiles=True,diagnostic=''))
    monkeypatch.setattr(outcomes,'verify_proof',lambda *a,**k:dict(ok=True,axioms=[],why='ok'))
    monkeypatch.setattr(outcomes,'verify_refutation',lambda *a,**k:dict(ok=False,why='not a negation'))
    path=tmp_path/'rows.jsonl'; path.write_text(json.dumps(with_metadata)+'\n'+json.dumps(legacy)+'\n')
    for _ in range(2):
        rescore.rescore_file(path,'unused',{'t':task})
        path=path.with_suffix('.rescored.jsonl')
        first,second=[json.loads(line) for line in path.read_text().splitlines()]
        assert first['sample_index']==2 and first['pilot_source']==provenance
        assert 'sample_index' not in second and 'pilot_source' not in second
        assert first['genuine'] and first['reward']==2


def test_rescore_keeps_the_mark_on_a_resumed_runs_copied_rows(tmp_path,monkeypatch):
    """The transcript export skips rows a resumed run copied from the original; rescoring must not drop that mark."""
    task=dict(id='t',group='g',tier='easy',true=True,statement='True')
    raw=json.dumps(dict(established='True',proof_body='by trivial',error=False,justification='Proved it.'))
    copied=dict(task_id='t',arm='report_correct', penalty=0.0,report_mode='typed',raw=raw,copied_from='results/original/train-baseline')
    own=dict(task_id='t',arm='report_correct', penalty=0.0,report_mode='typed',raw=raw)
    monkeypatch.setattr(outcomes,'evaluate_submission',lambda lean,statement,text:dict(json.loads(text),reward=1.,compiles=True,diagnostic=''))
    monkeypatch.setattr(outcomes,'verify_proof',lambda *a,**k:dict(ok=True,axioms=[],why='ok'))
    monkeypatch.setattr(outcomes,'verify_refutation',lambda *a,**k:dict(ok=False,why='not a negation'))
    path=tmp_path/'training-rollouts.jsonl'; path.write_text(json.dumps(copied)+'\n'+json.dumps(own)+'\n')
    rescore.rescore_file(path,'unused',{'t':task})
    first,second=[json.loads(line) for line in path.with_suffix('.rescored.jsonl').read_text().splitlines()]
    assert first['copied_from']=='results/original/train-baseline' and 'copied_from' not in second
