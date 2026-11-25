"""Data must move at least one declared prior across a statistical stop boundary."""
import math

import pytest

from rewardhack import pilot, pilot_model
from rewardhack.pilot_simulation import example_spec, synthetic_rows


@pytest.fixture
def spec(tmp_path):
    return pilot.load_spec(example_spec(tmp_path/'prior-demo'))


def rows(spec, *, se=1., mean=0.):
    observations = [r for c in spec['checkpoints'][:3] for r in synthetic_rows(spec,c,'null')]
    for row in observations:
        for effect in row['effects'].values():
            effect.update(mean=mean,se=se)
    return observations


def test_tiny_priors_cannot_end_noisy_pilot_for_plateau(spec):
    spec['model']['prior_scales']=[.05,.051]
    for endpoint in spec['endpoints']:
        endpoint['plateau_delta']=.2
    decision=pilot.decide(spec,rows(spec))
    assert decision['reason']=='continue'
    assert decision['prior_driven_boundaries']==['practical_plateau_within_budget']
    for variant in decision['posterior']['variants']:
        assert variant['plateau_lower']>.95
        assert variant['prior_plateau_lower']>.95


@pytest.mark.parametrize('target,boundary,key', [
    (-.5,'evidence_for_at_tested_compute','prior_success_lower'),
    (.5,'predicted_futility_within_budget','prior_future_success_upper'),
])
def test_noisy_data_cannot_ratify_prior_only_success_or_futility(spec,target,boundary,key):
    spec['model']['prior_scales']=[.05,.051]
    for endpoint in spec['endpoints']:
        endpoint['target']=target
    decision=pilot.decide(spec,rows(spec))
    assert not decision['stop']
    assert boundary in decision['prior_driven_boundaries']
    assert all(v[key] is not None for v in decision['posterior']['variants'])


def test_prior_bounds_do_not_learn_shape_weights_or_observation_variance(spec):
    before=pilot_model.analyze(spec,rows(spec,mean=0))
    after=pilot_model.analyze(spec,rows(spec,mean=.8,se=.01))
    prior_keys=('prior_success_lower','prior_future_success_upper','prior_plateau_lower')
    for first,second in zip(before['variants'],after['variants'],strict=True):
        assert {k:first[k] for k in prior_keys}=={k:second[k] for k in prior_keys}
    assert before['variants'][0]['shape_weights']!=after['variants'][0]['shape_weights']
    # The population-coefficient prior is independent of observation and seed noise.
    for scale in spec['model']['prior_scales']:
        matches=[v for v in before['variants'] if v['prior_scale']==scale]
        assert all(v['prior_plateau_lower']==matches[0]['prior_plateau_lower'] for v in matches)


def test_prior_success_matches_analytic_equal_weight_shape_mixture(spec):
    spec['endpoints']=spec['endpoints'][:1]
    spec['endpoints'][0]['target']=.1
    result=pilot_model.analyze(spec,rows(spec))
    x=.5
    bases=[0,x,1-math.exp(-4*x)]
    bases += [1/(1+math.exp(-12*(x-k)))-1/(1+math.exp(12*k)) for k in spec['model']['delayed_knees']]
    for variant in result['variants']:
        scale=variant['prior_scale']
        exact=sum(.5*math.erfc(.1/math.sqrt(2*scale**2*(1+f*f))) for f in bases)/len(bases)
        assert variant['prior_success_lower']==pytest.approx(exact)


def test_unknown_targets_and_finished_horizon_have_no_invented_prior_events(spec):
    data=[r for c in spec['checkpoints'] for r in synthetic_rows(spec,c)]
    for variant in pilot_model.analyze(spec,data)['variants']:
        assert variant['prior_success_lower'] is None
        assert variant['prior_future_success_upper'] is None
        assert variant['prior_plateau_lower'] is None
    assert pilot.decide(spec,data)['reason']=='compute_cap'


def test_one_skeptical_prior_is_sufficient_and_equality_is_not_crossing(spec,monkeypatch):
    variants=[dict(success_lower=.99,prior_success_lower=.99,model_misfit=False,
                   future_success_upper=None,prior_future_success_upper=None,
                   plateau_lower=.5,prior_plateau_lower=.5),
              dict(success_lower=.99,prior_success_lower=.95,model_misfit=False,
                   future_success_upper=None,prior_future_success_upper=None,
                   plateau_lower=.5,prior_plateau_lower=.5)]
    monkeypatch.setattr(pilot_model,'analyze',lambda *unused:dict(variants=variants))
    assert pilot.decide(spec,rows(spec))['reason']=='evidence_for_at_tested_compute'


def test_default_useful_and_delayed_learning_decisions_are_preserved(spec):
    spec['seeds']=[f's{i}' for i in range(12)]
    spec['endpoints'][0]['target']=.1
    spec['endpoints'][1]['target']=-.3
    def decision(scenario,points):
        return pilot.decide(spec,[r for c in points for r in synthetic_rows(spec,c,scenario)])
    assert decision('useful',[1,2,4])['reason']=='evidence_for_at_tested_compute'
    assert decision('delayed',[1,2,4])['reason']=='continue'
    assert decision('delayed',[1,2,4,8])['reason']=='evidence_for_at_tested_compute'
