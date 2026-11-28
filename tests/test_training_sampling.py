"""Stratification balances task draws, while preserving GRPO repetition and resume order."""
from collections import Counter
from itertools import islice

import pytest

from rewardhack.training_sampling import FamilyTruthSampler


def tasks():
    return [dict(id=f'{family}-{truth}-{index}', family=family, true=truth)
            for family, size in [('a', 2), ('b', 4), ('c', 8), ('d', 16), ('e', 32)]
            for truth in (False, True) for index in range(size)]


def sampler(rows, epoch=0, seed=42):
    return FamilyTruthSampler(rows, seed=seed, batch_size=4, mini_repeat_count=4, repeat_count=4, epoch=epoch)


def test_every_five_update_block_balances_family_and_truth():
    rows = tasks()
    draw = sampler(rows)
    assert draw.block_size == 20
    order = draw.task_indices(0)
    for start in range(0, len(order), 20):
        counts = Counter((rows[i]['family'], rows[i]['true']) for i in order[start:start+20])
        assert len(counts) == 10 and set(counts.values()) == {2}
    assert set(order) <= set(range(len(rows)))


def test_grpo_groups_and_batch_reuse_stay_contiguous():
    draw = sampler(tasks())
    result = list(draw)
    assert len(result) == len(draw)
    for start in range(0, len(result), 64):
        block = result[start:start+64]
        assert block[:16] == block[16:32] == block[32:48] == block[48:64]
        for group in range(0, 16, 4):
            assert len(set(block[group:group+4])) == 1


@pytest.mark.parametrize('epoch,completed_updates', [(0, 2), (1, 3), (2, 6)])
def test_resume_reconstructs_mid_block_and_later_epoch_order(epoch, completed_updates):
    uninterrupted = sampler(tasks())
    for _ in range(epoch):
        list(uninterrupted)
    expected = list(uninterrupted)[completed_updates*64:]
    resumed = sampler(tasks(), epoch=epoch)
    assert list(islice(iter(resumed), completed_updates*64, None)) == expected
    assert list(resumed) == list(uninterrupted)


def test_schedule_depends_on_seed_not_model_arm_or_format():
    rows = tasks()
    changed_metadata = [dict(row, model='other', arm='baseline', report_mode='typed') for row in rows]
    assert sampler(rows).task_indices(0) == sampler(changed_metadata).task_indices(0)
    assert sampler(rows).task_indices(0) != sampler(rows, seed=43).task_indices(0)
    assert sampler(rows).task_indices(0) != sampler(rows).task_indices(1)


def test_incomplete_stratum_is_refused():
    with pytest.raises(ValueError, match='both truth'):
        sampler([row for row in tasks() if row['true']])
