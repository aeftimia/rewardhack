"""Exercise MLX training orchestration without an Apple GPU or model downloads."""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class Array(list):
    """Only the vector arithmetic used to normalize rewards in the training loop."""
    def mean(self):
        return sum(self) / len(self)

    def std(self):
        return math.sqrt(sum((x - self.mean()) ** 2 for x in self) / len(self))

    def __sub__(self, other):
        return Array(x - other for x in self)

    def __truediv__(self, other):
        return Array(x / other for x in self)

    def tolist(self):
        return list(self)


@pytest.fixture
def train_stub(monkeypatch, tmp_path):
    gradient_calls, updates = [], []

    def loss_and_grad(model, ids, tokens, advantage):
        gradient_calls.append((tokens[0], advantage))
        return 0.0, advantage

    optimizer = SimpleNamespace(state={}, update=lambda model, gradient: updates.append(gradient))
    for name, attrs in {
        'mlx': {},
        'mlx.core': dict(array=Array, eval=lambda *a: None, get_peak_memory=lambda: 0,
                         save_safetensors=lambda *a: None),
        'mlx.nn': dict(value_and_grad=lambda *a: loss_and_grad),
        'mlx.optimizers': dict(Adam=lambda **kw: optimizer),
        'mlx.utils': dict(tree_flatten=lambda x: [], tree_map=lambda fn, a, b: fn(a, b)),
        'mlx_lm': dict(load=None, stream_generate=None),
        'mlx_lm.sample_utils': dict(make_sampler=None),
        'mlx_lm.tuner': {},
        'mlx_lm.tuner.utils': dict(linear_to_lora_layers=None),
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[1] / 'src/rewardhack/mlx_rl.py'
    spec = importlib.util.spec_from_file_location('rewardhack._verifier_test_mlx', source)
    mlx_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mlx_rl)
    # Six groups are reserved for monitoring; leave one for the training step.
    tasks = [dict(id=str(i), group=str(i), statement='True') for i in range(7)]
    monkeypatch.setattr(mlx_rl, 'split', lambda rows: {'train': tasks})
    monkeypatch.setattr(mlx_rl, 'taskset_rows', lambda name: [])
    monkeypatch.setattr(mlx_rl, 'attach_lora', lambda *a: None)
    monkeypatch.setattr(mlx_rl, 'encode', lambda *a: [1])
    model = SimpleNamespace(trainable_parameters=lambda: {}, parameters=lambda: {})

    def run(rewards, undecided, arm='report_correct'):
        rows = [dict(arm=arm, reward=reward, verifier_inconclusive=unverified, body='',
                     exploit=False, genuine=False, true=True, exploit_and_claims=False, format_valid=True)
                for reward, unverified in zip(rewards, undecided, strict=True)]
        samples = iter((str(i), [10+i, 2], 0.0, False) for i in range(len(rows)))
        monkeypatch.setattr(mlx_rl, 'sample', lambda *a: next(samples))
        monkeypatch.setattr(mlx_rl, 'score', lambda lean, task, raw, *a, **kw: dict(rows[int(raw)]))
        monkeypatch.setattr(mlx_rl, 'attempt_rate', lambda *a: [dict(row) for row in rows])
        args = SimpleNamespace(out=tmp_path, taskset='v1', seed=0, adapter=None, learning_rate=.01,
                               monitor_samples=1, steps=1, max_seconds=100, variant='neutral', thinking=False,
                               max_tokens=8, group=len(rows), lean='must-not-run', arm=arm, penalty=.5,
                               judge_fn=None, monitor_every=1)
        mlx_rl.rl(args, model, None)
        records = {name: [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
                   for name in ('steps.jsonl', 'monitor.jsonl', 'rollouts.jsonl')}
        records['summary'] = json.loads((tmp_path / 'summary.json').read_text())
        return records, gradient_calls, updates

    return run


def test_all_undecided_rows_skip_training_and_warn_in_every_summary(train_stub):
    records, gradients, updates = train_stub([1.0, 0.0], [True, True])
    assert gradients == updates == []
    assert records['steps.jsonl'][0]['mixed'] is False
    assert records['summary']['updates'] == 0
    for record in records['steps.jsonl'] + records['monitor.jsonl'] + [records['summary']]:
        assert record['unverified_rate'] == 1.0
        assert record['check_verifier'] is True
    assert [row['training_reward'] for row in records['rollouts.jsonl']] == [0.0, 0.0]


@pytest.mark.parametrize('arm', ['report_correct', 'equal_reward'])
def test_mixed_group_skips_undecided_gradient_pass(train_stub, arm):
    records, gradients, updates = train_stub([1.0, 0.0, 1.0], [False, False, True], arm)
    assert [token for token, _ in gradients] == [10, 11]
    assert gradients[0][1] > 0 and gradients[1][1] < 0
    assert len(updates) == 1 and records['summary']['updates'] == 1
    assert records['summary']['unverified_rate'] == pytest.approx(1/3)
    assert records['summary']['check_verifier'] is True
    assert [row['training_reward'] for row in records['rollouts.jsonl']] == [1.0, 0.0, .5]


def test_baseline_keeps_grader_rewards_and_gradients_for_undecided_rows(train_stub):
    records, gradients, updates = train_stub([1.0, 0.0], [True, True], 'baseline')
    assert [token for token, _ in gradients] == [10, 11]
    assert gradients[0][1] > 0 and gradients[1][1] < 0
    assert len(updates) == 1
    assert [row['training_reward'] for row in records['rollouts.jsonl']] == [1.0, 0.0]
    assert records['summary']['unverified_rate'] == 1.0
    assert records['summary']['check_verifier'] is True


def test_decided_rows_train_without_verifier_warning(train_stub):
    records, gradients, updates = train_stub([1.0, 0.0], [False, False])
    assert [token for token, _ in gradients] == [10, 11]
    assert len(updates) == 1
    for record in records['steps.jsonl'] + records['monitor.jsonl'] + [records['summary']]:
        assert record['unverified_rate'] == 0.0
        assert record['check_verifier'] is False
