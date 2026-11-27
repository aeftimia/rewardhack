"""Family/truth-balanced training draws, separate from token sampling and evaluation."""
import math
import random
from collections import defaultdict

MODES = ('uniform', 'family-truth')


class FamilyTruthSampler:
    """Equal stratum mass per block; contiguous GRPO groups and deterministic epochs.

    Minority strata are intentionally oversampled. A block is the smallest whole
    number of batches divisible by the number of strata. Epochs round the task
    pool size up to whole blocks. Transformers skips already-consumed batches
    within the reconstructed epoch when resuming a checkpoint.
    """

    def __init__(self, tasks, *, seed, batch_size, mini_repeat_count, repeat_count, epoch=0):
        if min(batch_size, mini_repeat_count, repeat_count) < 1 or epoch < 0 or not tasks:
            raise ValueError('nonempty tasks, positive batch/repetition counts and nonnegative epoch required')
        if len({task['id'] for task in tasks}) != len(tasks):
            raise ValueError('training task IDs must be unique')
        self.strata = defaultdict(list)
        for index, task in enumerate(tasks):
            if type(task['true']) is not bool:
                raise ValueError('task truth status must be Boolean')
            self.strata[(task['family'], task['true'])].append(index)
        families = {family for family, _ in self.strata}
        if set(self.strata) != {(family, truth) for family in families for truth in (False, True)}:
            raise ValueError('each training family must contain both truth statuses')
        self.keys = sorted(self.strata)
        self.seed = seed
        self.batch_size = batch_size
        self.mini_repeat_count = mini_repeat_count
        self.repeat_count = repeat_count
        self.epoch = epoch
        self.block_size = math.lcm(len(self.keys), batch_size)
        self.draws_per_epoch = math.ceil(len(tasks) / self.block_size) * self.block_size

    def task_indices(self, epoch):
        rng = random.Random(f'{self.seed}:{epoch}')
        remaining = {key: [] for key in self.keys}
        order = []
        for _ in range(self.draws_per_epoch // self.block_size):
            block = self.keys * (self.block_size // len(self.keys))
            rng.shuffle(block)
            for key in block:
                if not remaining[key]:
                    remaining[key] = list(self.strata[key])
                    rng.shuffle(remaining[key])
                order.append(remaining[key].pop())
        return order

    def __iter__(self):
        indexes = self.task_indices(self.epoch)
        self.epoch += 1
        for start in range(0, len(indexes), self.batch_size):
            chunk = indexes[start:start + self.batch_size]
            for _ in range(self.repeat_count):
                for index in chunk:
                    yield from [index] * self.mini_repeat_count

    def __len__(self):
        return self.draws_per_epoch * self.mini_repeat_count * self.repeat_count
