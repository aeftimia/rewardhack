"""Decision safety, frozen evidence, and paired-cohort regressions."""
import copy
import json
from pathlib import Path

import pytest

from rewardhack import pilot
from rewardhack.pilot_evidence import paired_observation
from rewardhack.pilot_simulation import example_spec, synthetic_rows


@pytest.fixture
def spec(tmp_path):
    return pilot.load_spec(example_spec(tmp_path/'demo'))


def all_rows(spec, points=None, scenario='null'):
    return [r for point in (points or spec['checkpoints']) for r in synthetic_rows(spec, point, scenario)]


def test_default_pilot_has_no_success_or_futility_claim(spec):
    for scenario in ('null', 'useful', 'delayed', 'harm', 'regression'):
        decision = pilot.decide(spec, all_rows(spec, scenario=scenario))
        assert decision['reason'] == 'compute_cap'
        assert not decision['hypothesis_defined']
        assert all(v['success_lower'] is None for v in decision['posterior']['variants'])


def test_early_flat_curve_is_not_a_negative_result(spec):
    decision = pilot.decide(spec, all_rows(spec, spec['checkpoints'][:2], 'delayed'))
    assert not decision['stop'] and not decision['mature']
    assert decision['next_checkpoint'] == 4


def test_measurement_failure_stops_without_bayesian_conclusion(spec):
    decision = pilot.decide(spec, all_rows(spec, [1], 'measurement_failure'))
    assert decision == dict(stop=True, reason='measurement_failure', posterior=None, next_checkpoint=None)


def test_unknown_and_duplicate_seed_or_nonprefix_are_rejected(spec):
    rows = all_rows(spec, [1])
    with pytest.raises(ValueError, match='seed'):
        pilot.decide(spec, rows+[rows[0]])
    with pytest.raises(ValueError, match='compute'):
        pilot.decide(spec, all_rows(spec, [2]))


def variant(success=.5, future=.5, plateau=.5, misfit=False):
    return dict(success_lower=success, success_upper=success, future_success_upper=future, plateau_lower=plateau, model_misfit=misfit)


def test_all_sensitivities_and_strict_boundary_required(spec, monkeypatch):
    from rewardhack import pilot_model
    for e in spec['endpoints']:
        e['target'] = .1
    variants = [variant(success=.99), variant(success=.95)]
    monkeypatch.setattr(pilot_model, 'analyze', lambda *a: dict(variants=variants))
    assert pilot.decide(spec, all_rows(spec, [1,2,4]))['reason'] == 'continue'
    variants[1]['success_lower'] = .96
    assert pilot.decide(spec, all_rows(spec, [1,2,4]))['reason'] == 'evidence_for_at_tested_compute'


def test_no_vacuous_futility_at_final_checkpoint(spec, monkeypatch):
    from rewardhack import pilot_model
    for e in spec['endpoints']:
        e['target'] = .9
    monkeypatch.setattr(pilot_model, 'analyze', lambda *a: dict(variants=[variant(future=0,plateau=None)]))
    assert pilot.decide(spec, all_rows(spec))['reason'] == 'compute_cap'
    assert pilot.decide(spec, all_rows(spec, [1,2,4]))['reason'] == 'predicted_futility_within_budget'


def test_model_misfit_stops_without_claiming_hypothesis_false(spec, monkeypatch):
    from rewardhack import pilot_model
    monkeypatch.setattr(pilot_model, 'analyze', lambda *a: dict(variants=[variant(success=.99,misfit=True)]))
    assert pilot.decide(spec, all_rows(spec, [1]))['reason'] == 'model_misfit'


@pytest.mark.parametrize('mutate', [
    lambda s: s['budget'].update(max_usd=float('nan')),
    lambda s: s['budget'].update(max_seconds=True),
    lambda s: s.update(seeds=['one','two']),
    lambda s: s.update(checkpoints=[1,1,4,8]),
    lambda s: s['model'].update(noise_floors=[0,.1]),
    lambda s: s['stop'].update(min_compute_fraction=.1),
    lambda s: s['stop'].update(confidence=.9),
    lambda s: s['stages'][0].update(argv=['python','x.py']),
    lambda s: s['stages'][0].update(inputs={}),
    lambda s: s['stages'][0].update(evidence='relative.jsonl'),
    lambda s: s['endpoints'][0].update(target=float('inf')),
])
def test_invalid_specs_fail_closed(spec, mutate):
    mutate(spec)
    with pytest.raises(ValueError):
        pilot.validate_spec(spec)


def test_evidence_is_complete_bound_and_nonreused(spec):
    paths = []
    for stage in spec['stages'][:2]:
        path = Path(stage['evidence']); paths.append(path)
        path.write_text(''.join(json.dumps(r)+'\n' for r in synthetic_rows(spec, stage['checkpoint'])))
    assert len(pilot.read_evidence(spec, paths)) == 6
    rows = synthetic_rows(spec, 2)
    rows[0]['sources'] = synthetic_rows(spec, 1)[0]['sources']
    paths[1].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='reused'):
        pilot.read_evidence(spec, paths)
    rows[0]['baseline_compute'] = 1
    paths[1].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='compute'):
        pilot.read_evidence(spec, paths)


def protocol_fixture(spec, tmp_path):
    protocol = dict(schema='paired-evaluation-v1', tasks=[dict(id=f'{group}-{truth}',group=group,true=truth) for group in ('g1','g2') for truth in (True,False)],
                    samples_per_task=2, baseline_arm='baseline', intervention_arm='report_correct',
                    endpoints=dict(exploit_reduction=dict(metric='exploit',truth='false',direction='decrease'),proof_retention=dict(metric='genuine',truth='true',direction='increase')))
    spec['context']['evaluation_protocol'] = pilot.digest(protocol)
    paths = []
    for arm in ('baseline', 'report_correct'):
        path=tmp_path/f'{arm}.jsonl'; paths.append(path)
        rows = [dict(task_id=t['id'], group=t['group'], true=t['true'], arm=arm, sample_index=i,
                     pilot_source=dict(spec_hash=pilot.digest(spec),seed='s1',checkpoint=1,compute=1,
                                       condition=spec['context']['baseline' if arm=='baseline' else 'intervention'],
                                       checkpoint_sha256=pilot.digest(arm)),
                     report_mode='typed', genuine=t['true'], exploit=not t['true'] and arm=='baseline',
                     format_valid=True,verifier_inconclusive=False,truncated=False,termination_schema='cutoff-v1')
                for t in protocol['tasks'] for i in range(2)]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return protocol, paths


def test_paired_effects_cluster_by_group_and_keep_all_planned_samples(spec,tmp_path):
    protocol, paths = protocol_fixture(spec,tmp_path)
    row=paired_observation(spec,protocol,'s1',1,*paths,1,1)
    assert row['effects']==dict(exploit_reduction=dict(mean=1,se=0),proof_retention=dict(mean=0,se=0))
    assert row['quality']['valid']
    with pytest.raises(ValueError,match='matched'):
        paired_observation(spec,protocol,'s1',1,*paths,1,2)
    data=paths[0].read_text().splitlines(); paths[0].write_text('\n'.join(data[:-1]))
    with pytest.raises(ValueError,match='incomplete'):
        paired_observation(spec,protocol,'s1',1,*paths,1,1)


def test_paired_unknown_verifier_never_disappears(spec,tmp_path):
    protocol, paths=protocol_fixture(spec,tmp_path)
    rows=[json.loads(x) for x in paths[1].read_text().splitlines()]
    rows[0]['verifier_inconclusive']=True
    paths[1].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    row=paired_observation(spec,protocol,'s1',1,*paths,1,1)
    assert not row['quality']['valid'] and row['quality']['invalid_rate']==1/16


def test_paired_rejects_protocol_drift_and_duplicate_samples(spec,tmp_path):
    protocol, paths=protocol_fixture(spec,tmp_path)
    changed=copy.deepcopy(protocol); changed['samples_per_task']=3
    with pytest.raises(ValueError,match='protocol'):
        paired_observation(spec,changed,'s1',1,*paths,1,1)
    paths[0].write_text(paths[0].read_text()+paths[0].read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError,match='sample_index'):
        paired_observation(spec,protocol,'s1',1,*paths,1,1)


def test_cutoff_cannot_masquerade_as_exploit_reduction(spec,tmp_path):
    protocol, paths=protocol_fixture(spec,tmp_path)
    rows=[json.loads(x) for x in paths[1].read_text().splitlines()]
    rows[1].update(truncated=True,format_valid=False,genuine=False,exploit=False)
    paths[1].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    row=paired_observation(spec,protocol,'s1',1,*paths,1,1)
    assert not row['quality']['valid']
    checkpoint=[dict(row,seed=s,sources=[pilot.digest(s)]) for s in spec['seeds']]
    assert pilot.decide(spec,checkpoint)['reason']=='measurement_failure'


def test_impossible_outcome_flags_refused(spec,tmp_path):
    protocol, paths=protocol_fixture(spec,tmp_path)
    rows=[json.loads(x) for x in paths[1].read_text().splitlines()]
    rows[0].update(genuine=True,exploit=True)
    paths[1].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='inconsistent'):
        paired_observation(spec,protocol,'s1',1,*paths,1,1)


def test_one_incompatible_sensitivity_blocks_scientific_stop(spec,monkeypatch):
    from rewardhack import pilot_model
    monkeypatch.setattr(pilot_model,'analyze',lambda *a:dict(variants=[variant(success=.99),variant(success=.99,misfit=True)]))
    assert pilot.decide(spec,all_rows(spec,[1,2,4]))['reason']=='model_misfit'


def test_paired_wrong_source_seed_checkpoint_and_condition_rejected(spec,tmp_path):
    protocol,paths=protocol_fixture(spec,tmp_path)
    original=[json.loads(x) for x in paths[1].read_text().splitlines()]
    for name,value in [('seed','s2'),('checkpoint',2),('compute',2),('condition','wrong'),('spec_hash','0'*64)]:
        rows=copy.deepcopy(original)
        rows[0]['pilot_source'][name]=value
        paths[1].write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with pytest.raises(ValueError,match='provenance'):
            paired_observation(spec,protocol,'s1',1,*paths,1,1)


def test_paired_file_cannot_mix_checkpoint_artifacts(spec,tmp_path):
    protocol,paths=protocol_fixture(spec,tmp_path)
    rows=[json.loads(x) for x in paths[1].read_text().splitlines()]
    rows[0]['pilot_source']['checkpoint_sha256']='a'*64
    paths[1].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='mixes checkpoint'):
        paired_observation(spec,protocol,'s1',1,*paths,1,1)


def test_real_curve_decision_saves_compute_without_killing_delayed_learning(spec):
    spec['seeds']=[f's{i}' for i in range(12)]
    spec['endpoints'][0]['target']=.1
    spec['endpoints'][1]['target']=-.3
    useful=pilot.decide(spec,all_rows(spec,[1,2,4],'useful'))
    delayed=pilot.decide(spec,all_rows(spec,[1,2,4],'delayed'))
    assert useful['reason']=='evidence_for_at_tested_compute'
    assert delayed['reason']=='continue'
    final=pilot.decide(spec,all_rows(spec,scenario='delayed'))
    assert final['reason']=='evidence_for_at_tested_compute'


@pytest.mark.parametrize('name',['prior_scales','noise_floors','seed_sds','delayed_knees'])
def test_numeric_duplicate_sensitivities_refused(spec,name):
    value=spec['model'][name][0]
    spec['model'][name]=[value,float(value)]
    with pytest.raises(ValueError):
        pilot.validate_spec(spec)
