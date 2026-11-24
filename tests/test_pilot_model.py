"""Numerical and dependence regressions for bounded pilot curve inference."""
import math

import pytest

from rewardhack import pilot_model


def specification(target=0.1):
    return dict(checkpoints=[1., 2., 4., 8.], seeds=['s1', 's2', 's3'],
                endpoints=[dict(name='benefit', target=target, plateau_delta=.02)],
                model=dict(prior_scales=[.25, .5], noise_floors=[.05, .1], seed_sds=[.15, .3],
                           delayed_knees=[.5, .9]))


def evidence(checkpoints=(1., 2.), means=(0., 0.)):
    return [dict(seed=seed, checkpoint=checkpoint,
                 effects={'benefit': {'mean': mean+offset, 'se': .03}})
            for seed, offset in [('s1', -.02), ('s2', 0), ('s3', .02)]
            for checkpoint, mean in zip(checkpoints, means, strict=True)]


def test_flat_regression_matches_exact_conjugate_normal_posterior_and_evidence():
    scale, noise, se, seed_sd = .4, .07, .09, .2
    y = .3
    fitted = pilot_model._fit([('one', 0., y, se)], scale, noise, seed_sd)
    measurement_variance = se*se+noise*noise+seed_sd*seed_sd
    variance = 1/(1/(scale*scale)+1/measurement_variance)
    mean = variance*y/measurement_variance
    assert fitted['mean'][0] == pytest.approx(mean)
    assert fitted['covariance'][0] == pytest.approx(variance)
    assert fitted['mean'][1] == 0
    assert fitted['covariance'][2] == pytest.approx(scale*scale)
    predictive_variance = measurement_variance+scale*scale
    log_evidence = -.5*(math.log(2*math.pi*predictive_variance)+y*y/predictive_variance)
    assert fitted['log_evidence'] == pytest.approx(log_evidence)
    assert fitted['max_abs_loo_z'] == pytest.approx(abs(y)/math.sqrt(predictive_variance))


def test_more_checkpoints_do_not_manufacture_independent_seeds():
    repeated = pilot_model._fit([('one', 0., .2, .01)]*100, .5, .01, .2)
    independent = pilot_model._fit([(str(i), 0., .2, .01) for i in range(100)], .5, .01, .2)
    limiting_variance = 1/(1/.5**2+1/.2**2)
    assert repeated['covariance'][0] >= limiting_variance
    assert repeated['covariance'][0] > 50*independent['covariance'][0]


def test_no_seed_variance_reduces_to_independent_observations():
    same = pilot_model._fit([('one', .2, .3, .02), ('one', .7, .6, .03)], .5, .1, 0)
    distinct = pilot_model._fit([('one', .2, .3, .02), ('two', .7, .6, .03)], .5, .1, 0)
    assert same == distinct


def test_many_checkpoints_do_not_erase_between_seed_slope_uncertainty():
    # Three incompatible seed trajectories are three replications, not 300.
    observations = [(str(seed), i/100, slope*i/100, .001)
                    for seed, slope in enumerate((-.5, 0., .5)) for i in range(1, 101)]
    correlated = pilot_model._fit(observations, .5, .01, .2)
    independent = pilot_model._fit(observations, .5, .01, 0)
    limiting_slope_variance = 1/(1/.5**2+3/.2**2)
    assert correlated['covariance'][2] >= limiting_slope_variance
    assert correlated['covariance'][2] > 100*independent['covariance'][2]
    assert pilot_model._inside(0, .5**2*correlated['covariance'][2], .02) < .95


def test_single_observation_seed_slope_matches_conjugate_predictive_density():
    x, y, scale, seed_sd, noise, se = .7, .3, .4, .2, .07, .09
    fit = pilot_model._fit([('s', x, y, se)], scale, noise, seed_sd)
    predictive_variance = (scale**2+seed_sd**2)*(1+x*x)+noise**2+se**2
    exact = -.5*(math.log(2*math.pi*predictive_variance)+y*y/predictive_variance)
    assert fit['log_evidence'] == pytest.approx(exact)


def test_woodbury_matches_dense_joint_gaussian_with_shared_and_distinct_seeds():
    observations = [('a', .1, .2, .03), ('a', .7, .6, .02), ('b', .4, -.1, .04)]
    prior, noise, seed_sd = .4, .05, .2
    size = len(observations)
    covariance = [[((prior**2+(seed_sd**2 if si == sj else 0))*(1+xi*xj)
                    +(sei**2+noise**2 if i == j else 0))
                   for j, (sj, xj, yj, sej) in enumerate(observations)]
                  for i, (si, xi, yi, sei) in enumerate(observations)]
    # Independent dense Cholesky reference, rather than reusing the model's algebra.
    lower = [[0.]*size for _ in range(size)]
    for i in range(size):
        for j in range(i+1):
            residual = covariance[i][j]-sum(lower[i][k]*lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(residual) if i == j else residual/lower[j][j]
    standardized = []
    for i, (_, _, y, _) in enumerate(observations):
        standardized.append((y-sum(lower[i][j]*standardized[j] for j in range(i)))/lower[i][i])
    log_density = -.5*(size*math.log(2*math.pi)+2*sum(math.log(lower[i][i]) for i in range(size))
                       +sum(z*z for z in standardized))
    fitted = pilot_model._fit(observations, prior, noise, seed_sd)
    assert fitted['log_evidence'] == pytest.approx(log_density)


def test_joint_probability_uses_bounds_not_independence_or_separate_thresholds():
    lower, upper = pilot_model._joint([.97, .97])
    assert lower == pytest.approx(.94)
    assert upper == .97
    assert lower < .95


def test_unknown_effect_targets_disable_success_and_futility_probabilities():
    result = pilot_model.analyze(specification(None), evidence())
    assert len(result['variants']) == 8
    assert result['current_checkpoint'] == 2
    for variant in result['variants']:
        assert variant['success_lower'] is None
        assert variant['success_upper'] is None
        assert variant['future_success_upper'] is None
        assert variant['current']['benefit']['p_target'] is None
        assert 0 <= variant['plateau_lower'] <= 1
        assert [r['checkpoint'] for r in variant['future']] == [4., 8.]


def test_early_flat_curve_retains_late_learning_uncertainty():
    result = pilot_model.analyze(specification(), evidence((1., 2.), (0., 0.)))
    for variant in result['variants']:
        assert variant['plateau_lower'] < .95
        assert variant['future_success_upper'] > .05
        delayed = [shape for shape in variant['shape_weights']['benefit'] if shape['kind'] == 'delayed']
        assert len(delayed) == 2
        assert all(shape['weight'] > .05 for shape in delayed)


def test_declining_trajectories_are_allowed_and_predictions_are_bounded_in_compute():
    spec = specification()
    result = pilot_model.analyze(spec, evidence((1., 2., 4.), (.4, .2, -.2)))
    for variant in result['variants']:
        assert variant['future'][0]['checkpoint'] == 8.
        assert variant['future'][0]['benefit']['mean'] < variant['current']['benefit']['mean']
        assert variant['future'][0]['benefit']['p_target'] < .5


def test_completed_horizon_has_no_fictitious_plateau_forecast():
    result = pilot_model.analyze(specification(), evidence((1., 2., 4., 8.), (0., .1, .2, .2)))
    for variant in result['variants']:
        assert variant['future'] == []
        assert variant['future_success_upper'] == 0
        assert variant['plateau_lower'] is None


def test_unmodeled_oscillation_triggers_diagnostic_not_a_confident_plateau():
    result = pilot_model.analyze(specification(), evidence((1., 2., 4., 8.), (.9, -.9, .9, -.9)))
    for variant in result['variants']:
        assert variant['model_misfit']
        assert variant['misfit_endpoints'] == ['benefit']
        assert all(shape['max_abs_loo_z'] > 6 for shape in variant['shape_weights']['benefit'])


def test_abrupt_hump_is_not_explained_by_a_monotonic_shape():
    result = pilot_model.analyze(specification(), evidence((1., 2., 4., 8.), (0., .9, 0., 0.)))
    assert all(variant['model_misfit'] for variant in result['variants'])


def test_consistent_linear_seed_trajectories_pass_model_diagnostic():
    result = pilot_model.analyze(specification(), evidence((1., 2., 4., 8.), (.05, .1, .2, .4)))
    assert all(not variant['model_misfit'] for variant in result['variants'])


def test_forecast_contrast_uses_covariance_and_cancels_intercept_uncertainty():
    # No change in basis has exactly zero uncertainty, regardless of intercept SD.
    assert pilot_model._inside(0, 0, .02) == 1
    fit = pilot_model._fit([('s', 0, .1, .01)], 100, .01, 10)
    assert fit['covariance'][0] > 1
    assert pilot_model._inside(0, 0*fit['covariance'][2], .02) == 1


def test_model_average_is_deterministic_and_weights_normalize():
    first = pilot_model.analyze(specification(), evidence())
    assert first == pilot_model.analyze(specification(), list(reversed(evidence())))
    for variant in first['variants']:
        assert sum(s['weight'] for s in variant['shape_weights']['benefit']) == pytest.approx(1)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.01, True])
def test_invalid_measurement_errors_fail_closed(bad):
    with pytest.raises(ValueError):
        pilot_model._fit([('s', 0, .1, bad)], .5, .1, .1)


def test_two_endpoint_joint_and_future_bounds_are_conservative():
    spec = specification()
    spec['endpoints'].append(dict(name='retention', target=-.1, plateau_delta=.02))
    rows = evidence((1., 2., 4.), (.1, .2, .3))
    for row in rows:
        row['effects']['retention'] = dict(mean=0, se=.03)
    result = pilot_model.analyze(spec, rows)
    for variant in result['variants']:
        p = [value['p_target'] for value in variant['current'].values()]
        assert variant['success_lower'] == pytest.approx(max(0, sum(p)-1))
        assert variant['success_upper'] == min(p)
        future = variant['future'][0]
        assert variant['future_success_upper'] == min(future['benefit']['p_target'], future['retention']['p_target'])
