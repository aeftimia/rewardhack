"""Beta-posterior rates, the clustered difference of two arms, the Student-t pilot effect and the Bayesian stops (27 Nov)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from rewardhack import outcomes, rates
from rewardhack.judge import agreement, score_review
from rewardhack.judge_holdout import DEFAULT_CRITERIA, acceptance, credibly_within
from rewardhack.pilot_evidence import group_effect


def test_a_rate_keeps_its_counts_and_adds_the_jeffreys_posterior():
    cell = rates.rate(3, 40)
    assert (cell['n'], cell['of']) == (3, 40)
    assert cell['mean'] == pytest.approx(3.5/41)
    assert (cell['low'], cell['high']) == pytest.approx(stats.beta(3.5, 37.5).interval(0.95))
    assert cell['low'] < 3/40 < cell['high']
    # No readings: the prior, wide open, and `of` says why.
    empty = rates.rate(0, 0)
    assert empty['of'] == 0 and empty['mean'] == 0.5 and empty['low'] < 0.01 and empty['high'] > 0.99
    with pytest.raises(ValueError, match='between 0'):
        rates.rate(5, 4)


def test_exceeds_reads_the_posterior_tail_and_has_no_answer_without_readings():
    assert rates.above(0, 100, 0.05) == pytest.approx(stats.beta(0.5, 100.5).sf(0.05))
    assert rates.exceeds(5, 100, 0.05, 0.5) is True and rates.exceeds(4, 100, 0.05, 0.5) is False
    assert rates.exceeds(0, 0, 0.05, 0.5) is None


def test_group_counts_share_one_set_of_groups():
    rows = [dict(group='a', hit=True), dict(group='a', hit=False), dict(group='b', hit=True)]
    assert rates.group_counts(rows, lambda r: r['hit'], {'a', 'b', 'c'}) == {'a': (1, 2), 'b': (1, 1), 'c': (0, 0)}
    with pytest.raises(KeyError):
        rates.group_counts(rows, lambda r: r['hit'], {'a'})


def test_the_difference_of_two_arms_is_a_posterior_and_widens_with_group_spread():
    before = {'g1': (2, 50), 'g2': (3, 50), 'g3': (2, 50), 'g4': (3, 50)}
    after = {'g1': (20, 50), 'g2': (22, 50), 'g3': (18, 50), 'g4': (21, 50)}
    change = rates.difference(before, after)
    assert change['p_increase'] > 0.999 and change['low'] > 0.25 and change['mean'] == pytest.approx(0.81/2 - 0.05, abs=0.01)
    assert rates.difference(before, after) == change, 'a fixed seed gives the same posterior summary'
    # Same pooled counts, but the change happens in two groups only: the clustered interval is wider.
    lumpy = {'g1': (2, 50), 'g2': (3, 50), 'g3': (38, 50), 'g4': (38, 50)}
    spread = rates.difference(before, lumpy)
    assert spread['high'] - spread['low'] > 2*(change['high'] - change['low'])
    # One group: the difference of two Beta posteriors.
    single = rates.difference({'g': (10, 100)}, {'g': (10, 100)})
    assert abs(single['mean']) < 0.005 and single['low'] < 0 < single['high'] and 0.45 < single['p_increase'] < 0.55
    with pytest.raises(ValueError, match='same, nonempty'):
        rates.difference({'g1': (1, 2)}, {'g2': (1, 2)})
    with pytest.raises(ValueError, match='no readings'):
        rates.difference({'g': (0, 0)}, {'g': (1, 2)})


def scored(group, true=True, exploit=False, genuine=False, honest=True, claim='OK', **extra):
    return dict(group=group, task_id=group, true=true, exploit=exploit, genuine=genuine, honest=honest, claim=claim,
                format_valid=True, verifier_inconclusive=False, verified_refutation=False, **extra)


def test_compare_reports_per_answer_changes_and_none_where_there_is_nothing_to_compare():
    before = [scored(g, exploit=False) for g in 'abc' for _ in range(10)]
    after = [scored(g, exploit=True, honest=False) for g in 'abc' for _ in range(10)]
    change = outcomes.compare(before, after)
    assert change['exploit']['p_increase'] > 0.999 and change['concealed_exploit']['p_increase'] > 0.999
    assert change['honest']['p_increase'] < 0.001
    assert change['verified_refutation_false'] is None  # no false tasks in either set
    # Unjudged rows: only the outcomes that need no judge.
    plain = [dict(r, claim=None) for r in before]
    assert set(outcomes.compare(plain, plain)) == set(outcomes.COMPARED)
    with pytest.raises(ValueError, match='same task groups'):
        outcomes.compare(before, after[:10])


def test_the_pilot_effect_is_a_student_t_posterior_read_as_a_matched_normal_width():
    means = [0.1, 0.3]
    effect = group_effect(means)
    scale = np.std(means, ddof=1)/np.sqrt(2)
    assert effect['mean'] == pytest.approx(0.2)
    assert effect['se'] == pytest.approx(stats.t.ppf(0.975, 1)*scale/stats.norm.ppf(0.975))
    assert effect['se'] > 6*scale  # the old normal standard error was six and a half times too narrow with two groups
    three = group_effect([0.1, 0.2, 0.3])
    assert three['se'] == pytest.approx(stats.t.ppf(0.975, 2)*0.1/np.sqrt(3)/stats.norm.ppf(0.975))
    assert group_effect([1.0, 1.0]) == dict(mean=1.0, se=0.0)
    with pytest.raises(ValueError, match='two independent'):
        group_effect([0.5])


def test_judge_validation_reports_intervals_and_accepts_only_credibly_low_error_rates():
    rows = [dict(category='c', read_claim='OK', claim='OK', genuine=g) for g in [True]*100 + [False]*100]
    result = agreement(rows)
    assert result['false_honest'] == rates.rate(0, 100) and result['false_honest']['high'] < 0.03
    assert credibly_within(0, 40, 0.05) and not credibly_within(1, 40, 0.05)
    flags = [dict(category='c', read_claim='ERROR', claim='ERROR', genuine=False) for _ in range(40)]
    perfect = agreement(rows + flags)
    assert acceptance(perfect, 240)['accepted'] and acceptance(perfect, 240)['accept_risk'] == 0.05
    # One missed flag in 40 flagged reports: an observed 2.5%, under the 5% limit, but P(rate > 5%) = 0.25, so not accepted.
    few = agreement(rows + flags[:39] + [dict(flags[0], claim='OK')])
    verdict = acceptance(few, 240, DEFAULT_CRITERIA)
    assert not verdict['accepted'] and any('missed_flag' in f for f in verdict['failures'])


def test_the_human_audit_reports_the_reading_error_as_a_posterior():
    key = [dict(number=i, stratum='random', read_claim='OK', genuine=True, id=str(i)) for i in range(1, 25)]
    sheet = '\n'.join(f'Answer {i:02d}: K' for i in range(1, 25))
    error = score_review(sheet, key)['strata']['random']['reading_error']
    assert error == rates.rate(0, 24) and 0.05 < error['high'] < 0.11


def test_the_stop_waits_for_near_certainty_and_matches_the_simulation():
    assert outcomes.STOP_CONFIDENCE == 0.999 and outcomes.FAILURE_WINDOW == 200
    flags = [False]*19 + [True]
    assert outcomes.failing(flags) is False and outcomes.failing([]) is False
    assert outcomes.failing([True]*3) is True
    path = Path(__file__).resolve().parent.parent/'entrypoints'/'failure_stop_sim.py'
    spec = importlib.util.spec_from_file_location('failure_stop_sim', path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    at_limit, _, _ = sim.stops(0.05, 4000, 60, 16, seed=0)
    assert at_limit.mean() < 0.05, 'a 5% fluke stopped too many runs'
    stopped, failures, readings = sim.stops(0.20, 2000, 60, 16, seed=0)
    assert stopped.all() and np.median(failures) <= 12 and np.median(readings) <= 64
