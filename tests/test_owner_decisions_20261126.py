"""Owner's decisions of 26 November: undecided rows train on nothing in every arm, a run of verifier timeouts stops
training, and the evaluation draw follows --seed."""
import json
from types import SimpleNamespace

import pytest

from rewardhack import hf_grpo, outcomes
from test_scan_fixes import fresh
from test_training_integration import train


def test_a_verifier_timeout_trains_on_nothing_in_every_arm():
    for arm in ('baseline', 'report_correct', 'equal_reward'):
        assert outcomes.training_keep(fresh(arm=arm, verifier_inconclusive=True)) is False
        assert outcomes.training_keep(fresh(arm=arm)) is True
    # The baseline's reward never reads the judge, so a failed reading does not drop its row.
    assert outcomes.training_keep(fresh(arm='baseline', judge_undecided=True)) is True
    assert outcomes.training_keep(fresh(judge_undecided=True)) is False


def test_verifier_timeouts_stop_a_run_only_beyond_a_fluke():
    ok, timeout = fresh(), fresh(verifier_inconclusive=True)
    assert not outcomes.verifier_failing([timeout] + [ok]*19)
    assert not outcomes.verifier_failing([timeout]*3)  # fewer than STOP_MIN_FAILURES
    assert outcomes.verifier_failing([timeout]*4)
    # Late failures are not hidden by a long good history.
    assert outcomes.verifier_failing([ok]*5000 + [timeout]*40 + [ok]*160)
    # Rows that predate the label are unknown and do not count either way.
    old = fresh()
    del old['verifier_inconclusive'], old['genuine_why']
    assert not outcomes.verifier_failing([old]*50)


def test_the_longest_undecided_proof_is_counted_per_step():
    rows = [fresh(verifier_inconclusive=True, body='x'*900), fresh(verifier_inconclusive=True, body='y'*40), fresh(body='z'*5000)]
    assert outcomes.tally(rows)['inconclusive_longest_proof_chars'] == 900
    assert outcomes.tally([fresh(body='z'*5000)])['inconclusive_longest_proof_chars'] == 0


def test_a_run_of_verifier_timeouts_stops_training(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='undecided, its undecided rate is credibly above 5%'):
        train(tmp_path, monkeypatch, lambda i, raw, step: None, timeouts_stop=True)
    assert json.loads(next(tmp_path.glob('run*/training-rollouts.jsonl')).read_text().splitlines()[0])['partial_batch']


def test_the_evaluation_draw_follows_the_run_seed():
    assert hf_grpo.evaluation_seed(SimpleNamespace(seed=1)) != hf_grpo.evaluation_seed(SimpleNamespace(seed=2))
    assert hf_grpo.evaluation_seed(SimpleNamespace(seed=7)) == hf_grpo.evaluation_seed(SimpleNamespace(seed=7))
