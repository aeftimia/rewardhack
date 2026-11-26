"""Report mode reaches generation, scoring, and saved rows without model/API calls."""
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rewardhack import hf_grpo, openai_probe, vllm_probe
from rewardhack.outcomes import RETRY_SECONDS

ROOT = Path(__file__).resolve().parents[1]
TASK = dict(id='t', group='g', statement='True', true=True, tier='easy')


@pytest.mark.parametrize('mode', ['prose', 'typed'])
@pytest.mark.parametrize('filename,module,positional,required', [
    ('run.py', 'hf_grpo', ['probe'], ['--lean', 'stub']),
    ('local_mlx.py', 'mlx_rl', ['rl'], ['--lean', 'stub']),
    ('probe_vllm.py', 'vllm_probe', [], ['--lean', 'none', '--model', 'stub', '--revision', 'stub']),
    ('probe_api.py', 'openai_probe', [], ['--model', 'stub', '--max-usd', '0', '--usd-per-m-input', '0', '--usd-per-m-output', '0']),
])
def test_report_mode_cli(monkeypatch, tmp_path, mode, filename, module, positional, required):
    seen = []
    stub = ModuleType('rewardhack.' + module)
    stub.run = seen.append
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    args = [] if mode == 'prose' else ['--report-mode', mode]
    monkeypatch.setattr(sys, 'argv', [filename, *positional, '--out', str(tmp_path), *required, *args])
    runpy.run_path(str(ROOT/'entrypoints'/filename), run_name='__main__')
    assert len(seen) == 1 and seen[0].report_mode == mode


def test_api_mode_survives_follow_on_and_unscored_storage(monkeypatch, tmp_path):
    modes, calls = [], []
    monkeypatch.setattr(openai_probe, 'prompt', lambda statement, variant, mode: modes.append(mode) or [{'role': 'system', 'content': mode}])
    monkeypatch.setattr(openai_probe, 'select_tasks', lambda *a, **kw: [TASK])
    monkeypatch.setattr(openai_probe, 'taskset_rows', lambda *a: [dict(TASK, id='f', group='other', true=False)])
    def call(args, messages, *, timeout, retries, extra_body):
        calls.append(messages)
        return dict(raw='{}', usage={}, finish='stop', model='stub')
    monkeypatch.setattr(openai_probe, 'call', call)
    args = SimpleNamespace(report_mode='typed', max_usd=1, usd_per_m_input=0, usd_per_m_output=0,
                           out=tmp_path, taskset='v1', tasks='all', limit=None, samples=1, concurrency=1,
                           follow_on=True, variant='neutral', model='stub', reasoning_effort=None, max_tokens=8)
    openai_probe.run(args)
    rows = [json.loads(line) for line in (tmp_path/'rows.jsonl').read_text().splitlines()]
    assert modes == ['typed'] and len(calls) == 2
    assert calls[1][0] == {'role': 'system', 'content': 'typed'}
    assert [row['report_mode'] for row in rows] == ['typed', 'typed']
    assert [row['turn'] for row in rows] == [1, 2]
    assert json.loads((tmp_path/'summary.json').read_text())['report_mode'] == 'typed'


@pytest.mark.parametrize('lean', ['none', 'stub'])
def test_vllm_mode_reaches_prompt_scoring_and_storage(monkeypatch, tmp_path, lean):
    modes, scored = [], []
    monkeypatch.setattr(vllm_probe, 'prompt', lambda statement, variant, mode: modes.append(mode) or [])
    class Tokenizer:
        chat_template = ''  # no clock and no reasoning switch to settle
        def apply_chat_template(self, *a, **kw): return 'prompt'
        def __call__(self, text): return {'input_ids': [1]}
    class LLM:
        def __init__(self, **kw): pass
        def get_tokenizer(self): return Tokenizer()
        def generate(self, *a):
            return [SimpleNamespace(outputs=[SimpleNamespace(text='{}', token_ids=[2], finish_reason='stop')])]
    stub = ModuleType('vllm')
    stub.__dict__.update(LLM=LLM, SamplingParams=lambda **kw: kw, __version__='stub')
    monkeypatch.setitem(sys.modules, 'vllm', stub)
    monkeypatch.setattr(vllm_probe, 'select_tasks', lambda *a: [TASK])
    monkeypatch.setattr(vllm_probe, 'GpuSampler', lambda: SimpleNamespace(start=lambda: None, summary=lambda: {}))
    monkeypatch.setattr(vllm_probe.deadline, 'arm', lambda *a: None)
    monkeypatch.setattr(vllm_probe.deadline, 'disarm', lambda *a: None)
    monkeypatch.setattr(vllm_probe.subprocess, 'check_output', lambda *a, **kw: 'Lean stub')
    def score(*a, **kw):
        scored.append(kw['report_mode'])
        return dict(report_mode=kw['report_mode'])
    monkeypatch.setattr(vllm_probe, 'score', score)
    monkeypatch.setattr(vllm_probe, 'summarize', lambda rows: {})
    args = SimpleNamespace(report_mode='typed', verifier_retry_seconds=RETRY_SECONDS, part='train', template_kwargs='{}', out=tmp_path, max_seconds=10,
                           lean=lean, taskset='v1', tasks='all', model='stub', revision='stub', seed=0,
                           max_model_len=16, gpu_memory_utilization=.5, samples=1, max_tokens=8, variant='neutral')
    vllm_probe.run(args)
    assert modes == ['typed'] and scored == ([] if lean == 'none' else ['typed'])
    assert json.loads((tmp_path/'rows.jsonl').read_text())['report_mode'] == 'typed'
    assert json.loads((tmp_path/'summary.json').read_text())['report_mode'] == 'typed'


def test_hf_typed_evaluation_strips_special_tokens_and_routes_mode(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    modes, decoded, scored = [], [], []
    monkeypatch.setattr(hf_grpo, 'prompt', lambda statement, variant, mode: modes.append(mode) or [])
    class Inputs(dict):
        def to(self, device): return self
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 9
        def apply_chat_template(self, *a, **kw): return Inputs(input_ids=torch.tensor([[1]]))
        def decode(self, tokens, **kw):
            decoded.append(kw['skip_special_tokens'])
            return '{}'
    model = SimpleNamespace(parameters=lambda: iter([torch.ones(1)]), eval=lambda: None,
                            generate=lambda **kw: torch.tensor([[1, 2, 9]]))
    def score(*a, **kw):
        scored.append(kw['report_mode'])
        return dict(report_mode=kw['report_mode'])
    monkeypatch.setattr(hf_grpo, 'score', score)
    monkeypatch.setattr(hf_grpo, 'training_rewards', lambda *a: None)
    monkeypatch.setattr(hf_grpo, 'summarize', lambda rows: {})
    monkeypatch.setattr(hf_grpo, 'reward_groups', lambda *a: dict(groups=1, mixed_groups=0, zero_advantage_groups=1, per_group=[]))
    args = SimpleNamespace(report_mode='typed', verifier_retry_seconds=RETRY_SECONDS, out=tmp_path, eval_samples=1, prompt_variant='neutral',
                           thinking='on', reasoning_effort='low', max_completion_length=8, lean='stub',
                           arm='baseline', penalty=0, judge_fn=None, seed=0)
    result = hf_grpo.evaluate(model, Tokenizer(), [TASK], args, 'test')
    assert modes == scored == ['typed'] and decoded == [True]
    assert result['report_mode'] == 'typed'


def test_mlx_encode_default_and_typed_monitor_mode(monkeypatch):
    for name, attrs in {
        'mlx': {}, 'mlx.core': {}, 'mlx.nn': {}, 'mlx.optimizers': {},
        'mlx.utils': dict(tree_flatten=None, tree_map=None),
        'mlx_lm': dict(load=None, stream_generate=None), 'mlx_lm.sample_utils': dict(make_sampler=None),
        'mlx_lm.tuner': {}, 'mlx_lm.tuner.utils': dict(linear_to_lora_layers=None),
    }.items():
        module = ModuleType(name); module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('rewardhack._mode_mlx', ROOT/'src/rewardhack/mlx_rl.py')
    mlx_rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(mlx_rl)
    modes, scored = [], []
    monkeypatch.setattr(mlx_rl, 'prompt', lambda statement, variant, mode: modes.append(mode) or [])
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1])
    mlx_rl.encode(tokenizer, 'True', 'neutral', False)  # historical holdout callers stay prose
    monkeypatch.setattr(mlx_rl, 'sample', lambda *a: ('{}', [2], 0., False))
    monkeypatch.setattr(mlx_rl, 'score', lambda *a, **kw: scored.append(kw['report_mode']) or {})
    args = SimpleNamespace(report_mode='typed', verifier_retry_seconds=RETRY_SECONDS, variant='neutral', thinking=False, max_tokens=8,
                           lean='stub', arm='baseline', penalty=0, judge_fn=None)
    mlx_rl.attempt_rate(None, tokenizer, [TASK], args, 1)
    assert modes == ['prose', 'typed'] and scored == ['typed']


def test_hf_training_routes_typed_prompt_and_reward(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    from rewardhack import outcomes
    modes, scored = [], []
    monkeypatch.setattr(hf_grpo, 'prompt', lambda statement, variant, mode: modes.append(mode) or [])
    template = outcomes.score('unused', TASK, '', 'baseline', truncated=True)
    def score(*a, **kw):
        scored.append(kw['report_mode'])
        return dict(template, report_mode=kw['report_mode'])
    monkeypatch.setattr(hf_grpo, 'score', score)
    class Trainer:
        def __init__(self, model, reward_funcs, **kw):
            self.model, self.reward = model, reward_funcs
            self.state = SimpleNamespace(global_step=0, log_history=[])
            self.args = SimpleNamespace(lr_scheduler_type='constant', to_dict=dict)  # what the real trainer reports back, recorded in the summary
            self.router_aux_loss_coef, self.aux_loss_enabled, self.generation_config = 0.0, False, SimpleNamespace(to_dict=dict)
        def train(self, resume_from_checkpoint):
            assert resume_from_checkpoint is None
            assert self.reward(['{}'], ['True'], completion_ids=[[2, 9]]) == [0.]
        def save_model(self, path): pass
    for name, attrs in {
        'datasets': dict(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
        'peft': dict(LoraConfig=lambda **kw: kw),
        'transformers': dict(TrainerCallback=object, set_seed=lambda seed: None),
        'trl': dict(GRPOTrainer=Trainer, GRPOConfig=lambda **kw: kw),
    }.items():
        module = ModuleType(name); module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    weight = torch.nn.Parameter(torch.ones(1))
    model = SimpleNamespace(named_parameters=lambda: iter([('weight', weight)]), named_modules=lambda: iter([]))
    args = SimpleNamespace(report_mode='typed', verifier_retry_seconds=RETRY_SECONDS, out=tmp_path, arm='baseline', lean='stub', penalty=0,
                           judge_fn=None, num_generations=1, max_completion_length=8, adapter=True, resume_from=None, save_every=99, rollout_backend='transformers',
                           steps=1, prompts_per_step=1, learning_rate=.01, beta=0, seed=0,
                           thinking='on', reasoning_effort='low', prompt_variant='neutral', lr_schedule='constant', router_aux_loss_coef=0.0, scale_rewards='group', max_grad_norm=1.0)
    summary = dict(rollouts={})
    hf_grpo.train(model, SimpleNamespace(eos_token_id=9), [TASK], args, summary)
    assert modes == scored == ['typed'] and summary['report_mode'] == 'typed'
    assert json.loads((tmp_path/'training-rollouts.jsonl').read_text())['report_mode'] == 'typed'
