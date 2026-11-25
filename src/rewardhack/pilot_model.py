"""Small, deterministic Bayesian learning-curve model for a bounded pilot.

Each endpoint is a paired, seed-level effect (higher is better). A proper finite
mixture of Gaussian regressions allows flat, improving, declining and delayed
trajectories. Shared random intercepts and slopes correlate checkpoints from the same seed.
Unknown endpoint dependence is handled with probability bounds, not multiplication.
This is an approximate Gaussian model of effects, not a universal RL scaling law.
Everything is closed-form Gaussian algebra: no optimiser, grid search or quadrature.
"""
import math
from itertools import product

import numpy as np
from scipy import linalg, special, stats


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f'{name} is outside its allowed range')
    return float(value)


# Curve shapes b(x) over the compute fraction x in [0, 1]; each starts at b(0) = 0.
BASES = {
    'flat': lambda x, knee: 0.0,
    'linear': lambda x, knee: x,
    'saturating': lambda x, knee: -math.expm1(-4*x),
    'delayed': lambda x, knee: float(special.expit(12*(x-knee)) - special.expit(-12*knee)),
}


def _basis(kind, x, knee=None):
    if kind not in BASES:
        raise ValueError(f'unknown curve shape {kind!r}')
    return BASES[kind](x, knee)


def _cholesky(matrix, name):
    """Lower Cholesky factor of a symmetric matrix; nonfinite or not positive definite is a ValueError."""
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f'nonfinite {name}')
    try:
        return linalg.cho_factor(matrix, lower=True, check_finite=False)
    except linalg.LinAlgError as error:
        raise ValueError(f'{name} is not positive definite') from error


def _fit(observations, prior_scale, noise_floor, seed_sd):
    """Integrate proper Normal coefficients and correlated seed trajectories exactly.

    `observations` contains (seed, basis, measured effect, standard error). With
    design X = [1, x] and coefficients β ~ N(0, prior_scale² I), each seed adds its
    own random intercept and slope with sd seed_sd, so the observations have covariance

        V = diag(se² + noise_floor²) + seed_sd² (XXᵀ within a seed),
        K = V + prior_scale² XXᵀ          (y with β integrated out).

    Posterior of β: precision P = I/prior_scale² + XᵀV⁻¹X, covariance P⁻¹, mean P⁻¹XᵀV⁻¹y.
    This precision form keeps its digits when the data are informative; the equivalent
    prior_scale²I − prior_scale⁴XᵀK⁻¹X subtracts two nearly equal matrices.
    Log evidence is log N(y; 0, K). For a joint Normal the exact leave-one-out
    z-score is (K⁻¹y)ᵢ / sqrt((K⁻¹)ᵢᵢ).
    """
    prior_scale = _number(prior_scale, 'prior scale', positive=True)
    noise_floor = _number(noise_floor, 'noise floor', positive=True)
    seed_sd = _number(seed_sd, 'seed sd', nonnegative=True)
    if not observations:
        raise ValueError('the learning curve needs observations')
    # A canonical order makes the result bit-for-bit independent of the order rows arrive in.
    rows = sorted((seed, _number(x, 'basis'), _number(y, 'effect'), _number(se, 'standard error', nonnegative=True))
                  for seed, x, y, se in observations)
    seeds = [seed for seed, _, _, _ in rows]
    x, y, se = (np.array(column) for column in list(zip(*rows, strict=True))[1:])
    design = np.column_stack([np.ones_like(x), x])
    gram = design @ design.T
    same_seed = np.array([[a == b for b in seeds] for a in seeds])
    within = np.diag(se**2+noise_floor**2) + seed_sd**2*np.where(same_seed, gram, 0.)
    marginal = within + prior_scale**2*gram

    within_factor = _cholesky(within, 'within-seed covariance')
    weighted_design = linalg.cho_solve(within_factor, design, check_finite=False)  # V⁻¹X
    information = design.T @ weighted_design
    precision = np.eye(2)/prior_scale**2 + (information+information.T)/2
    precision_factor = _cholesky(precision, 'learning-curve posterior precision')
    covariance = linalg.cho_solve(precision_factor, np.eye(2), check_finite=False)
    mean = linalg.cho_solve(precision_factor, weighted_design.T @ y, check_finite=False)

    marginal_factor = _cholesky(marginal, 'learning-curve marginal covariance')
    standardized = linalg.cho_solve(marginal_factor, y, check_finite=False)  # K⁻¹y
    log_determinant = 2*np.sum(np.log(np.diag(marginal_factor[0])))
    log_evidence = -0.5*(len(y)*math.log(2*math.pi) + log_determinant + y @ standardized)
    if not np.all(np.isfinite([*mean, *covariance.ravel(), log_evidence])):
        raise ValueError('nonfinite learning-curve posterior')
    precision_diagonal = np.diag(linalg.cho_solve(marginal_factor, np.eye(len(y)), check_finite=False))
    if not np.all(np.isfinite(precision_diagonal)) or np.any(precision_diagonal <= 0):
        raise ValueError('unstable leave-one-out predictive precision')
    return dict(mean=(float(mean[0]), float(mean[1])),
                covariance=(float(covariance[0, 0]), float(covariance[0, 1]), float(covariance[1, 1])),
                log_evidence=float(log_evidence),
                max_abs_loo_z=float(np.max(np.abs(standardized)/np.sqrt(precision_diagonal))))


def _variance(value):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)) or np.any(value < -1e-10):
        raise ValueError('invalid learning-curve posterior variance')
    return np.maximum(0.0, value)


def _normal_probability(mean, variance, point_mass, probability):
    """Elementwise probability of an event under N(mean, variance); a zero variance is a point mass."""
    shape = np.broadcast_shapes(np.shape(mean), np.shape(variance))
    mean, variance = np.broadcast_arrays(np.array(mean, dtype=float, ndmin=1), np.array(variance, dtype=float, ndmin=1))
    result = point_mass(mean).astype(float)
    spread = variance != 0
    result[spread] = probability(mean[spread], np.sqrt(variance[spread]))
    return result.reshape(shape)


def _above(mean, variance, threshold):
    """P(N(mean, variance) >= threshold)."""
    return _normal_probability(mean, variance, lambda m: m >= threshold,
                               lambda m, sd: stats.norm.sf(threshold, loc=m, scale=sd))


def _inside(mean, variance, delta):
    """P(|N(mean, variance)| < delta)."""
    return _normal_probability(mean, variance, lambda m: np.abs(m) < delta,
                               lambda m, sd: stats.norm.cdf(delta, loc=m, scale=sd)-stats.norm.cdf(-delta, loc=m, scale=sd))


def _distribution(fits, x, target):
    """Moments of the shape mixture for intercept + b(x)·slope, and P(effect >= target)."""
    weights = np.array([fit['weight'] for fit in fits])
    loadings = [np.array([1.0, _basis(fit['kind'], x, fit['knee'])]) for fit in fits]
    means = np.array([loading @ fit['mean'] for fit, loading in zip(fits, loadings, strict=True)])
    variances = _variance([loading @ np.array([[c00, c01], [c01, c11]]) @ loading
                           for (c00, c01, c11), loading in zip((fit['covariance'] for fit in fits), loadings, strict=True)])
    mean = weights @ means
    variance = _variance(weights @ (variances+(means-mean)**2))
    return dict(mean=float(mean), sd=float(np.sqrt(variance)),
                p_target=None if target is None else float(weights @ _above(means, variances, target)))


def _joint(probabilities):
    """Frechet bounds for an intersection without assuming endpoint independence."""
    return max(0.0, 1-math.fsum(1-p for p in probabilities)), min(probabilities)


def _plateau(fits, x_from, x_to, delta):
    """P(|change in effect between two compute fractions| < delta); the intercept cancels."""
    weights = np.array([fit['weight'] for fit in fits])
    change = np.array([_basis(fit['kind'], x_to, fit['knee'])-_basis(fit['kind'], x_from, fit['knee']) for fit in fits])
    slope_means = np.array([fit['mean'][1] for fit in fits])
    slope_variances = np.array([fit['covariance'][2] for fit in fits])
    return float(weights @ _inside(change*slope_means, _variance(change**2*slope_variances), delta))


def _summaries(spec, endpoint_fits, current, horizon, future):
    """Use identical events and probability bounds for prior and posterior curves."""
    targets_known = all(endpoint['target'] is not None for endpoint in spec['endpoints'])
    present = {endpoint['name']: _distribution(endpoint_fits[endpoint['name']], current/horizon, endpoint['target'])
               for endpoint in spec['endpoints']}
    forecasts = []
    plateau_failures = []
    for checkpoint in future:
        forecast = {'checkpoint': checkpoint}
        for endpoint in spec['endpoints']:
            name = endpoint['name']
            forecast[name] = _distribution(endpoint_fits[name], checkpoint/horizon, endpoint['target'])
            probability = _plateau(endpoint_fits[name], current/horizon, checkpoint/horizon, endpoint['plateau_delta'])
            plateau_failures.append(1-min(1.0, max(0.0, probability)))
        forecasts.append(forecast)
    lower, upper = _joint([value['p_target'] for value in present.values()]) if targets_known else (None, None)
    future_upper = (min(1.0, math.fsum(min(forecast[endpoint['name']]['p_target'] for endpoint in spec['endpoints'])
                                     for forecast in forecasts)) if targets_known and future else None)
    return dict(current=present, future=forecasts, success_lower=lower, success_upper=upper,
                future_success_upper=future_upper,
                plateau_lower=max(0.0, 1-math.fsum(plateau_failures)) if future else None)


def analyze(spec, rows):
    """Analyze completed checkpoints; decisions and maturity gates belong to pilot.py.

    The caller validates the frozen design and complete paired evidence. All
    predictions concern planned checkpoints within its finite compute allowance.
    Posterior moments are not clipped to [-1, 1]: Gaussian support is an explicit
    approximation and clipping would hide model mismatch rather than fix it.
    The hyperparameter grid is not integrated over: each (prior scale, noise floor,
    seed sd) is a separate sensitivity variant, and the delayed knees are discrete
    mixture components with equal prior mass.
    """
    if not rows:
        raise ValueError('no completed pilot evidence')
    checkpoints = spec['checkpoints']
    horizon = max(checkpoints)
    current = max(row['checkpoint'] for row in rows)
    future = [checkpoint for checkpoint in checkpoints if checkpoint > current]
    model = spec['model']
    shapes = [('flat', None), ('linear', None), ('saturating', None)]
    shapes += [('delayed', knee) for knee in model['delayed_knees']]
    variants = []
    for prior_scale, noise_floor, seed_sd in product(model['prior_scales'], model['noise_floors'], model['seed_sds']):
        endpoint_fits = {}
        for endpoint in spec['endpoints']:
            name = endpoint['name']
            fits = []
            for kind, knee in shapes:
                observations = [(row['seed'], _basis(kind, row['checkpoint']/horizon, knee),
                                 row['effects'][name]['mean'], row['effects'][name]['se']) for row in rows]
                fit = _fit(observations, prior_scale, noise_floor, seed_sd)
                fits.append(dict(fit, kind=kind, knee=knee))
            # Equal, proper prior mass on the declared shapes: weights are normalised evidences.
            log_evidence = np.array([fit['log_evidence'] for fit in fits])
            for fit, weight in zip(fits, np.exp(log_evidence-special.logsumexp(log_evidence)), strict=True):
                fit['weight'] = float(weight)
            endpoint_fits[name] = fits
        posterior = _summaries(spec, endpoint_fits, current, horizon, future)
        prior_fits = {endpoint['name']: [dict(kind=kind, knee=knee, weight=1/len(shapes),
                                            mean=(0.0, 0.0), covariance=(prior_scale**2, 0.0, prior_scale**2))
                                       for kind, knee in shapes] for endpoint in spec['endpoints']}
        prior = _summaries(spec, prior_fits, current, horizon, future)
        misfit_endpoints = [name for name, fits in endpoint_fits.items()
                            if all(fit['max_abs_loo_z'] > 6 for fit in fits)]
        variants.append(dict(prior_scale=prior_scale, noise_floor=noise_floor, seed_sd=seed_sd,
                             **posterior,
                             prior_success_lower=prior['success_lower'],
                             prior_future_success_upper=prior['future_success_upper'],
                             prior_plateau_lower=prior['plateau_lower'],
                             model_misfit=bool(misfit_endpoints), misfit_endpoints=misfit_endpoints,
                             shape_weights={name: [{'kind': fit['kind'], 'knee': fit['knee'], 'weight': fit['weight'],
                                                   'max_abs_loo_z': fit['max_abs_loo_z']}
                                                   for fit in fits] for name, fits in endpoint_fits.items()}))
    return dict(variants=variants, current_checkpoint=current,
                interpretation='Conditional Gaussian learning-curve posterior over the planned compute horizon; '
                'seed-correlated intercepts/slopes, conservative joint bounds. Six-sigma leave-one-out alarms are '
                'model diagnostics, not evidence against the hypothesis or a universal calibration claim.')
