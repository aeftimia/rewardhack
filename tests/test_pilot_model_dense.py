"""`pilot_model._fit` against an independent dense reference.

The reference builds the full marginal covariance of the observations,

    K = diag(se² + noise_floor²) + seed_sd²·(11ᵀ + xxᵀ within a seed) + prior_scale²·XXᵀ,

and reads every reported quantity from it in function space: the coefficient
posterior by Gaussian conditioning (never through a precision matrix), the
evidence as the N(0, K) log-density of y, and the leave-one-out z-scores as
(K⁻¹y)ᵢ / sqrt((K⁻¹)ᵢᵢ). It shares no algebra with the model.
"""
import math

import numpy as np
import pytest

from rewardhack import pilot_model

# Agreement demanded of `_fit`, relative to the reference with an absolute floor
# for quantities that are exactly or nearly zero.
RELATIVE, ABSOLUTE = 1e-7, 1e-9


def dense_reference(observations, prior_scale, noise_floor, seed_sd):
    seeds = [seed for seed, _, _, _ in observations]
    x, y, se = (np.array(column, dtype=float) for column in list(zip(*observations, strict=True))[1:])
    design = np.column_stack([np.ones_like(x), x])
    gram = design @ design.T
    same_seed = np.array([[a == b for b in seeds] for a in seeds], dtype=float)
    covariance = np.diag(se**2+noise_floor**2) + seed_sd**2*same_seed*gram + prior_scale**2*gram
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    inverse = np.linalg.inv(covariance)
    alpha = np.linalg.solve(covariance, y)
    cross = prior_scale**2*design.T  # cov(coefficients, observations)
    mean = cross @ alpha
    posterior = prior_scale**2*np.eye(2) - cross @ np.linalg.solve(covariance, cross.T)
    return dict(mean=tuple(mean), covariance=(posterior[0, 0], posterior[0, 1], posterior[1, 1]),
                log_evidence=-.5*(len(y)*math.log(2*math.pi)+logdet+y @ alpha),
                max_abs_loo_z=float(np.max(np.abs(alpha)/np.sqrt(np.diag(inverse)))))


def flatten(fit):
    return [*fit['mean'], *fit['covariance'], fit['log_evidence'], fit['max_abs_loo_z']]


def assert_matches_reference(observations, prior_scale, noise_floor, seed_sd):
    fitted = pilot_model._fit(observations, prior_scale, noise_floor, seed_sd)
    assert set(fitted) == {'mean', 'covariance', 'log_evidence', 'max_abs_loo_z'}
    reference = dense_reference(observations, prior_scale, noise_floor, seed_sd)
    # Coefficient moments are compared on the scale of the prior they shrink from.
    floors = [ABSOLUTE*prior_scale]*2+[ABSOLUTE*prior_scale**2]*3+[ABSOLUTE]*2
    for got, expected, floor in zip(flatten(fitted), flatten(reference), floors, strict=True):
        assert got == pytest.approx(expected, rel=RELATIVE, abs=floor)


def random_case(rng):
    """Several seeds of unequal size on the model's own scales, with the edge cases mixed in."""
    sizes = rng.integers(1, 9, size=rng.integers(1, 7))
    kind = ['flat', 'linear', 'saturating', 'delayed'][rng.integers(4)]
    knee = float(rng.uniform(.1, .95))
    observations = []
    for seed, size in enumerate(sizes):
        for _ in range(size):
            se = 0.0 if rng.random() < .25 else float(rng.uniform(.001, .2))
            observations.append((f's{seed}', pilot_model._basis(kind, float(rng.uniform(0, 1)), knee),
                                 float(rng.normal(0, .5)), se))
    seed_sd = 0.0 if rng.random() < .25 else float(rng.uniform(.01, .5))
    return observations, float(rng.uniform(.05, 1)), float(rng.uniform(.01, .2)), seed_sd


@pytest.mark.parametrize('trial', range(400))
def test_fit_matches_dense_reference_on_random_designs(trial):
    assert_matches_reference(*random_case(np.random.default_rng(trial)))


@pytest.mark.parametrize('scale', [1e-3, 1e-2, 1., 1e2, 1e3])
@pytest.mark.parametrize('trial', range(20))
def test_fit_matches_dense_reference_when_every_quantity_is_rescaled(scale, trial):
    """Effects, errors and prior all multiplied by one factor: a change of units."""
    observations, prior_scale, noise_floor, seed_sd = random_case(np.random.default_rng(1000+trial))
    assert_matches_reference([(seed, x, y*scale, se*scale) for seed, x, y, se in observations],
                             prior_scale*scale, noise_floor*scale, seed_sd*scale)


@pytest.mark.parametrize('observations, prior_scale, noise_floor, seed_sd', [
    ([('s', .7, .3, .09)], .4, .07, .2),                                    # one observation
    ([('s', .7, .3, 0.)], .4, .07, 0.),                                     # one observation, no se, no seed effect
    ([('a', 0., .2, 0.), ('a', 0., -.1, 0.), ('b', 0., .4, 0.)], .5, .05, .3),  # flat shape: slope unidentified
    ([('a', .5, .2, .01)]*6+[('b', .5, .1, .01)], .25, .05, .15),            # repeated rows, unequal groups
    ([(f's{i}', i/31, .01*i, .03) for i in range(32)], .5, .1, 0.),          # every row its own seed
    ([(s, i/31, .01*i, .03) for s in 'abc' for i in range(32)], .5, .05, .3),  # the pilot's largest design
    ([('a', .1, .2, .03), ('a', .7, .6, .02), ('b', .4, -.1, .04)], 5., .05, .001),  # wide prior, tiny seed sd
    ([('a', .1, .2, .03), ('a', .7, .6, .02), ('b', .4, -.1, .04)], .01, .2, 2.),    # tight prior, wide seed sd
])
def test_fit_matches_dense_reference_on_named_edge_cases(observations, prior_scale, noise_floor, seed_sd):
    assert_matches_reference(observations, prior_scale, noise_floor, seed_sd)


def test_fit_does_not_depend_on_the_order_of_the_observations():
    observations, *parameters = random_case(np.random.default_rng(7))
    assert pilot_model._fit(observations, *parameters) == pilot_model._fit(observations[::-1], *parameters)


@pytest.mark.parametrize('observations, parameters', [
    ([], (.5, .1, .1)),
    ([('s', float('nan'), .1, .01)], (.5, .1, .1)),
    ([('s', 0., float('inf'), .01)], (.5, .1, .1)),
    ([('s', 0., .1, -.01)], (.5, .1, .1)),
    ([('s', 0., .1, .01)], (0., .1, .1)),
    ([('s', 0., .1, .01)], (.5, 0., .1)),
    ([('s', 0., .1, .01)], (.5, .1, -.1)),
    ([('s', 0., .1, .01)], (float('nan'), .1, .1)),
    ([('s', 0., 1e200, .01), ('s', 1., 1e200, .01)], (.5, .1, .1)),   # y'K⁻¹y overflows
    ([('s', 0., .1, 1e200)], (.5, .1, .1)),                           # se² overflows
    ([('s', 1e200, .1, .01)], (.5, .1, .1)),                          # x² overflows
])
def test_fit_refuses_inputs_it_cannot_integrate(observations, parameters):
    with pytest.raises(ValueError):
        pilot_model._fit(observations, *parameters)
