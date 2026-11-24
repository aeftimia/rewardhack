"""Small, deterministic Bayesian learning-curve model for a bounded pilot.

Each endpoint is a paired, seed-level effect (higher is better). A proper finite
mixture of Gaussian regressions allows flat, improving, declining and delayed
trajectories. Shared random intercepts and slopes correlate checkpoints from the same seed.
Unknown endpoint dependence is handled with probability bounds, not multiplication.
This is an approximate Gaussian model of effects, not a universal RL scaling law.
"""
import math
from collections import defaultdict
from itertools import product


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f'{name} is outside its allowed range')
    return float(value)


def _basis(kind, x, knee=None):
    if kind == 'flat':
        return 0.0
    if kind == 'linear':
        return x
    if kind == 'saturating':
        return -math.expm1(-4*x)
    if kind == 'delayed':
        return 1/(1+math.exp(-12*(x-knee))) - 1/(1+math.exp(12*knee))
    raise ValueError(f'unknown curve shape {kind!r}')


def _fit(observations, prior_scale, noise_floor, seed_sd):
    """Integrate proper Normal coefficients and correlated seed trajectories exactly.

`observations` contains (seed, basis, measured effect, standard error). Within a
    seed, V = diag(se**2 + noise_floor**2) + seed_sd**2 * (11' + xx').
    A two-dimensional Woodbury update integrates both a random intercept and a
    random slope, preserving dependence without a dense checkpoint matrix.
"""
    prior_scale = _number(prior_scale, 'prior scale', positive=True)
    noise_floor = _number(noise_floor, 'noise floor', positive=True)
    seed_sd = _number(seed_sd, 'seed sd', nonnegative=True)
    grouped = defaultdict(list)
    for seed, x, y, se in observations:
        grouped[seed].append((_number(x, 'basis'), _number(y, 'effect'),
                              _number(se, 'standard error', nonnegative=True)))
    if not observations:
        raise ValueError('the learning curve needs observations')
    xx00 = xx01 = xx11 = xy0 = xy1 = yy = logdet = 0.0
    predictive_terms = []
    tau2 = seed_sd**2
    for seed in sorted(grouped):
        group = grouped[seed]
        weighted = [(1/(se*se+noise_floor*noise_floor), x, y) for x, y, se in group]
        sw = math.fsum(w for w, x, y in weighted)
        sx = math.fsum(w*x for w, x, y in weighted)
        sy = math.fsum(w*y for w, x, y in weighted)
        sxx = math.fsum(w*x*x for w, x, y in weighted)
        sxy = math.fsum(w*x*y for w, x, y in weighted)
        # det(A) for A = X'D^-1X is nonnegative; roundoff can put it just below zero.
        determinant_a = max(0.0, sw*sxx-sx*sx)
        determinant_q = 1+tau2*(sw+sxx)+tau2*tau2*determinant_a
        # (I + tau² A)^-1 A, evaluated without subtracting two large matrices.
        xx00 += (sw+tau2*determinant_a)/determinant_q
        xx01 += sx/determinant_q
        xx11 += (sxx+tau2*determinant_a)/determinant_q
        effective_y0 = ((1+tau2*sxx)*sy-tau2*sx*sxy)/determinant_q
        effective_y1 = ((1+tau2*sw)*sxy-tau2*sx*sy)/determinant_q
        xy0 += effective_y0
        xy1 += effective_y1
        yy += math.fsum(w*y*y for w, x, y in weighted)-tau2*(sy*effective_y0+sxy*effective_y1)
        logdet += math.fsum(math.log(se*se+noise_floor*noise_floor) for x, y, se in group)
        logdet += math.log(determinant_q)
        q00, q01, q11 = (1+tau2*sxx)/determinant_q, -tau2*sx/determinant_q, (1+tau2*sw)/determinant_q
        for w, x, y in weighted:
            # Entries of V^-1 X, V^-1 y and diag(V^-1), for exact Gaussian LOO checks.
            r0, r1 = w*(q00+x*q01), w*(q01+x*q11)
            vy = w*(y-tau2*(effective_y0+x*effective_y1))
            diagonal = w-tau2*w*w*(q00+2*x*q01+x*x*q11)
            predictive_terms.append((r0, r1, vy, diagonal))
    precision = 1/(prior_scale*prior_scale)
    p00, p01, p11 = precision+xx00, xx01, precision+xx11
    determinant = p00*p11-p01*p01
    if not math.isfinite(determinant) or determinant <= 0:
        raise ValueError('unstable learning-curve posterior precision')
    c00, c01, c11 = p11/determinant, -p01/determinant, p00/determinant
    m0, m1 = c00*xy0+c01*xy1, c01*xy0+c11*xy1
    log_evidence = -0.5*(len(observations)*math.log(2*math.pi)+logdet
                         +4*math.log(prior_scale)+math.log(determinant)
                         +yy-xy0*m0-xy1*m1)
    if not all(math.isfinite(v) for v in (m0, m1, c00, c01, c11, log_evidence)):
        raise ValueError('nonfinite learning-curve posterior')
    max_loo_z = 0.0
    for r0, r1, vy, diagonal in predictive_terms:
        # K includes both seed effects and the proper population-coefficient prior.
        # For a joint Normal, LOO z_i = (K^-1 y)_i / sqrt((K^-1)_ii).
        precision_diagonal = diagonal-r0*r0*c00-2*r0*r1*c01-r1*r1*c11
        if not math.isfinite(precision_diagonal) or precision_diagonal <= 0:
            raise ValueError('unstable leave-one-out predictive precision')
        z = abs(vy-r0*m0-r1*m1)/math.sqrt(precision_diagonal)
        max_loo_z = max(max_loo_z, z)
    return dict(mean=(m0, m1), covariance=(c00, c01, c11), log_evidence=log_evidence,
                max_abs_loo_z=max_loo_z)


def _variance(value):
    if not math.isfinite(value) or value < -1e-10:
        raise ValueError('invalid learning-curve posterior variance')
    return max(0.0, value)


def _above(mean, variance, threshold):
    if variance == 0:
        return float(mean >= threshold)
    return 0.5*math.erfc((threshold-mean)/math.sqrt(2*variance))


def _inside(mean, variance, delta):
    if variance == 0:
        return float(abs(mean) < delta)
    sd = math.sqrt(2*variance)
    return max(0.0, min(1.0, 0.5*(math.erf((delta-mean)/sd)-math.erf((-delta-mean)/sd))))


def _distribution(fits, x, target):
    components = []
    for fit in fits:
        basis = _basis(fit['kind'], x, fit['knee'])
        m0, m1 = fit['mean']
        c00, c01, c11 = fit['covariance']
        components.append((fit['weight'], m0+basis*m1, _variance(c00+2*basis*c01+basis*basis*c11)))
    mean = math.fsum(w*m for w, m, v in components)
    variance = _variance(math.fsum(w*(v+(m-mean)**2) for w, m, v in components))
    return dict(mean=mean, sd=math.sqrt(variance),
                p_target=None if target is None else math.fsum(w*_above(m, v, target) for w, m, v in components))


def _joint(probabilities):
    """Frechet bounds for an intersection without assuming endpoint independence."""
    return max(0.0, 1-math.fsum(1-p for p in probabilities)), min(probabilities)


def analyze(spec, rows):
    """Analyze completed checkpoints; decisions and maturity gates belong to pilot.py.

    The caller validates the frozen design and complete paired evidence. All
    predictions concern planned checkpoints within its finite compute allowance.
    Posterior moments are not clipped to [-1, 1]: Gaussian support is an explicit
    approximation and clipping would hide model mismatch rather than fix it.
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
    targets_known = all(endpoint['target'] is not None for endpoint in spec['endpoints'])
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
            maximum = max(fit['log_evidence'] for fit in fits)
            weights = [math.exp(fit['log_evidence']-maximum) for fit in fits]
            total = math.fsum(weights)
            for fit, weight in zip(fits, weights, strict=True):
                fit['weight'] = weight/total  # equal, proper prior mass on the declared shapes
            endpoint_fits[name] = fits
        present = {endpoint['name']: _distribution(endpoint_fits[endpoint['name']], current/horizon, endpoint['target'])
                   for endpoint in spec['endpoints']}
        forecasts = []
        plateau_failures = []
        for checkpoint in future:
            forecast = {'checkpoint': checkpoint}
            for endpoint in spec['endpoints']:
                name = endpoint['name']
                forecast[name] = _distribution(endpoint_fits[name], checkpoint/horizon, endpoint['target'])
                probability = 0.0
                for fit in endpoint_fits[name]:
                    difference = (_basis(fit['kind'], checkpoint/horizon, fit['knee'])
                                  - _basis(fit['kind'], current/horizon, fit['knee']))
                    mean = difference*fit['mean'][1]
                    variance = _variance(difference*difference*fit['covariance'][2])
                    probability += fit['weight']*_inside(mean, variance, endpoint['plateau_delta'])
                plateau_failures.append(1-min(1.0, max(0.0, probability)))
            forecasts.append(forecast)
        lower, upper = _joint([value['p_target'] for value in present.values()]) if targets_known else (None, None)
        future_upper = (min(1.0, math.fsum(min(forecast[endpoint['name']]['p_target'] for endpoint in spec['endpoints'])
                                         for forecast in forecasts)) if targets_known else None)
        misfit_endpoints = [name for name, fits in endpoint_fits.items()
                            if all(fit['max_abs_loo_z'] > 6 for fit in fits)]
        variants.append(dict(prior_scale=prior_scale, noise_floor=noise_floor, seed_sd=seed_sd, current=present, future=forecasts,
                             success_lower=lower, success_upper=upper, future_success_upper=future_upper,
                             plateau_lower=max(0.0, 1-math.fsum(plateau_failures)) if future else None,
                             model_misfit=bool(misfit_endpoints), misfit_endpoints=misfit_endpoints,
                             shape_weights={name: [{'kind': fit['kind'], 'knee': fit['knee'], 'weight': fit['weight'],
                                                   'max_abs_loo_z': fit['max_abs_loo_z']}
                                                   for fit in fits] for name, fits in endpoint_fits.items()}))
    return dict(variants=variants, current_checkpoint=current,
                interpretation='Conditional Gaussian learning-curve posterior over the planned compute horizon; '
                'seed-correlated intercepts/slopes, conservative joint bounds. Six-sigma leave-one-out alarms are '
                'model diagnostics, not evidence against the hypothesis or a universal calibration claim.')
