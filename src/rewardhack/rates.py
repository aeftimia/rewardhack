"""Bayesian error and outcome rates: Beta posteriors for counts, and the posterior of a difference between two arms.

Every rate here is k events in n readings under a Jeffreys prior, Beta(1/2, 1/2). Why Jeffreys and not the uniform
Beta(1, 1): docs/notes-2026-11-21.md (27 Nov). In short, the rates that matter (concealed exploits, judge and verifier
failures, false honest readings) sit near zero, where a uniform prior adds a whole pseudo-event and Jeffreys half of
one, and Jeffreys intervals keep close to their stated coverage at small n.
"""
import numpy as np
from scipy import stats

PRIOR = (0.5, 0.5)  # Jeffreys: Beta(a, b), pseudo-counts of events and non-events
CREDIBLE = 0.95  # mass of every reported equal-tailed credible interval
DRAWS = 20_000  # Monte Carlo draws for a difference of rates: the interval ends move by about 0.002 between seeds


def posterior(k, n):
    """Beta posterior of a rate after k events in n readings. Takes numbers or arrays."""
    k, n = np.asarray(k), np.asarray(n)
    if np.any(k < 0) or np.any(k > n):
        raise ValueError(f'event counts must lie between 0 and the number of readings: {k} of {n}')
    return stats.beta(k + PRIOR[0], n - k + PRIOR[1])


def rate(k, n):
    """`{n, of}` (k of n), plus the posterior mean and the equal-tailed 95% credible interval `low`..`high`.

    With no readings (`of` 0) the posterior is the prior, and so is its interval: it says nothing, and `of` shows why.
    """
    belief = posterior(k, n)
    low, high = belief.interval(CREDIBLE)
    return {'n': int(k), 'of': int(n), 'mean': float(belief.mean()), 'low': float(low), 'high': float(high)}


def above(k, n, limit):
    """Posterior probability that the rate is above `limit`. Takes numbers or arrays (the stop simulation passes arrays)."""
    return posterior(k, n).sf(limit)


def exceeds(k, n, limit, confidence):
    """Whether the rate is above `limit` with posterior probability above `confidence`; None with no readings to judge by."""
    if n == 0:
        return None
    return bool(above(k, n, limit) > confidence)


def group_counts(rows, test, groups):
    """`{group: (k, n)}` for `difference`: in each of `groups`, k of the n rows pass `test`. A group with no rows here
    is (0, 0), so two arms filtered differently still share one set of groups; a row from an unlisted group is refused."""
    counts = dict.fromkeys(groups, (0, 0))
    for row in rows:
        k, n = counts[row['group']]
        counts[row['group']] = (k + bool(test(row)), n + 1)
    return counts


def difference(before, after, *, draws=DRAWS, seed=0):
    """Posterior of rate(after) - rate(before), clustered by task group. A posterior, not a p-value.

    `before` and `after` map each task group to `(k, n)` for one arm (`group_counts`). Two sources of uncertainty are
    drawn together. Within a group, each arm's rate is drawn from its Beta posterior. Across groups, which groups the task
    set happened to hold is a Bayesian bootstrap: Dirichlet(1, ..., 1) weights on the groups, the same weights for both
    arms, since both answered the same groups. An arm's pooled rate is its group rates weighted by weight times readings.
    With one group this is the difference of two Beta posteriors. With two or three groups the bootstrap can only reweight
    the groups seen, so the interval understates how far an unseen group could differ.
    """
    if before.keys() != after.keys() or not before:
        raise ValueError(f'both arms need the same, nonempty set of task groups: {sorted(before)} and {sorted(after)}')
    groups = sorted(before)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(groups)), size=draws)

    def pooled(counts):
        k, n = (np.array([counts[g][i] for g in groups]) for i in (0, 1))
        if not n.sum():
            raise ValueError('an arm has no readings in any group')
        rates = posterior(k, n).rvs(size=(draws, len(groups)), random_state=rng)
        return (weights*n*rates).sum(axis=1)/(weights*n).sum(axis=1)

    delta = pooled(after) - pooled(before)
    low, high = np.quantile(delta, [(1-CREDIBLE)/2, (1+CREDIBLE)/2])
    return {'mean': float(delta.mean()), 'low': float(low), 'high': float(high), 'p_increase': float((delta > 0).mean()),
            'groups': len(groups), 'draws': draws, 'seed': seed}
