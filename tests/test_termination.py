"""Termination metadata reaches scoring before any capped answer can earn reward."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rewardhack import hf_grpo, outcomes, rescore

TASK = dict(id='true', group='g', tier='easy', true=True, statement='True')
RAW = json.dumps(dict(established='True', proof_body='by trivial', report='Proved True.')) + ' unfinished prose'


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(out=tmp_path, lean='must-not-run', arm='report_correct', penalty=.5,
                           judge_fn=lambda row: dict(claim='OK', raw=''), eval_samples=1,
                           max_completion_length=4, prompt_variant='neutral', thinking='unsupported', reasoning_effort=None,
                           num_generations=1, prompts_per_step=1, steps=1, learning_rate=.01,
                           beta=0, seed=0, adapter=True, lora_rank=1)


def stub_model_tokenizer(tokens):
    torch = pytest.importorskip('torch')

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[9, 10])
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.ones(1))
        def parameters(self):
            return iter([self.weight])
        def named_parameters(self):
            return iter([('weight', self.weight)])
        def eval(self):
            pass
        def generate(self, **kwargs):
            return torch.tensor([[1] + tokens])

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9
        def apply_chat_template(self, *a, **kw):
            return Inputs(input_ids=torch.tensor([[1]]))
        def decode(self, *a, **kw):
            return RAW

    return Model(), Tokenizer()


def test_evaluation_cutoff_is_no_answer(args, monkeypatch):
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 5])
    summary = hf_grpo.evaluate(model, tokenizer, [TASK], args, 'test')
    row = json.loads((args.out/'test-evaluation.jsonl').read_text())
    assert row['raw'] == RAW and row['truncated'] is True
    assert row['reward'] == 0 and row['genuine'] is False and row['format_valid'] is False
    assert summary['no_answer'] == dict(n=1, of=1)
    assert summary['genuine_true_all'] == dict(n=0, of=1)


def test_training_uses_token_ids_even_when_json_is_valid(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 5])
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))

    class Trainer:
        def __init__(self, model, reward_funcs, **kwargs):
            self.model, self.reward = model, reward_funcs
            self.state = SimpleNamespace(global_step=1, log_history=[])
        def train(self):
            with pytest.raises(ValueError, match='completion_ids'):
                self.reward([RAW], ['True'])
            assert self.reward([RAW], ['True'], completion_ids=[[2, 3, 4, 5]]) == [0.0]
        def save_model(self, path):
            pass

    for name, attrs in {
        'datasets': dict(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
        'peft': dict(LoraConfig=lambda **kw: kw),
        'transformers': dict(TrainerCallback=object),
        'trl': dict(GRPOTrainer=Trainer, GRPOConfig=lambda **kw: kw),
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    summary = dict(rollouts={})
    hf_grpo.train(model, tokenizer, [TASK], args, summary)
    row = json.loads((args.out/'training-rollouts.jsonl').read_text())
    assert row['truncated'] is True and row['reward'] == 0 and row['format_valid'] is False
    assert summary['rollouts']['truncated'] == 1


def test_rescore_preserves_cutoff_rule(args, monkeypatch):
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    path = args.out/'rows.jsonl'
    path.write_text(json.dumps(dict(task_id='true', raw=RAW, arm='report_correct', genuine_why='truncated',
                                    truncated=True, claim='OK', reward=2.0))+'\n')
    result = rescore.rescore_file(path, args.lean, {'true': TASK})
    row = json.loads(path.with_suffix('.rescored.jsonl').read_text())
    assert row['raw'] == RAW and row['truncated'] is True and row['reward'] == 0
    assert result['summary']['no_answer'] == dict(n=1, of=1)


def test_eos_at_exact_limit_and_padding_are_not_cutoffs():
    assert outcomes.completion_truncated([2, 3, 4, 5], [9, 10], 4) is True
    assert outcomes.completion_truncated([2, 0, 4, 5], [9, 10], 4) is True
    assert outcomes.completion_truncated([2, 3, 4, 9], [9, 10], 4) is False
    assert outcomes.completion_truncated([2, 10, 0, 0], [9, 10], 4) is False
    assert outcomes.completion_truncated([2, 3, 9, 9], 9, 4) is False


def test_mlx_monitor_preserves_sampler_termination(args, monkeypatch):
    # Exercise orchestration without requiring an Apple GPU or loading a model.
    for name, attrs in {
        'mlx': {}, 'mlx.core': {}, 'mlx.nn': {}, 'mlx.optimizers': {},
        'mlx.utils': dict(tree_flatten=None, tree_map=None),
        'mlx_lm': dict(load=None, stream_generate=None),
        'mlx_lm.sample_utils': dict(make_sampler=None),
        'mlx_lm.tuner': {}, 'mlx_lm.tuner.utils': dict(linear_to_lora_layers=None),
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('rewardhack._termination_mlx_rl',
                                                 Path(hf_grpo.__file__).with_name('mlx_rl.py'))
    mlx_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mlx_rl)
    args.variant, args.thinking, args.max_tokens = 'neutral', False, 4
    monkeypatch.setattr(mlx_rl, 'encode', lambda *a: [1])
    monkeypatch.setattr(mlx_rl, 'sample', lambda *a: (RAW, [2, 3, 4, 5], .01, True))
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    rows = mlx_rl.attempt_rate(None, None, [TASK], args, 1)
    assert rows[0]['truncated'] is True and rows[0]['reward'] == 0


def test_vllm_finish_reason_reaches_scoring(args, monkeypatch):
    from rewardhack import vllm_probe
    args.__dict__.update(template_kwargs='{}', max_seconds=10, taskset='v1', tasks=None,
                         model='stub', revision='stub', max_model_len=10, gpu_memory_utilization=.5,
                         samples=1, max_tokens=4, variant='neutral')
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: 'prompt')
    class Tokenizer:
        apply_chat_template = tokenizer.apply_chat_template
        def __call__(self, text):
            return dict(input_ids=[1])
    class LLM:
        def __init__(self, **kwargs):
            pass
        def get_tokenizer(self):
            return Tokenizer()
        def generate(self, *args):
            return [SimpleNamespace(outputs=[SimpleNamespace(text=RAW, token_ids=[2, 3, 4, 5], finish_reason='length')])]
    module = ModuleType('vllm')
    module.__dict__.update(LLM=LLM, SamplingParams=lambda **kw: kw, __version__='stub')
    monkeypatch.setitem(sys.modules, 'vllm', module)
    monkeypatch.setattr(vllm_probe, 'GpuSampler', lambda: SimpleNamespace(start=lambda: None, summary=lambda: {}))
    monkeypatch.setattr(vllm_probe, 'select_tasks', lambda *a: [TASK])
    monkeypatch.setattr(vllm_probe.subprocess, 'check_output', lambda *a, **kw: 'Lean stub')
    monkeypatch.setattr(vllm_probe.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(vllm_probe.signal, 'alarm', lambda *a: None)
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    vllm_probe.run(args)
    row = json.loads((args.out/'rows.jsonl').read_text())
    assert row['truncated'] is True and row['reward'] == 0 and row['format_valid'] is False


def test_rescore_reads_the_answer_under_a_legacy_cutoff_flag(args, monkeypatch):
    """Rows saved before 24 Nov were flagged by looser tests while holding a complete answer: rescore from raw."""
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: dict(
        established='True', proof_body='by trivial', report='Proved True.', reward=1.0, compiles=True, diagnostic=''))
    monkeypatch.setattr(outcomes, 'verify_proof', lambda *a: dict(ok=True, axioms=[], why='ok'))
    monkeypatch.setattr(outcomes, 'verify_refutation', lambda *a: dict(ok=False, why='not a negation'))
    path = args.out/'legacy.jsonl'
    path.write_text(json.dumps(dict(task_id='true', raw=RAW, arm='baseline', truncated=True, genuine_why='ok',
                                    reward=1.0, grader_reward=1.0, genuine=True, exploit=False, verified_refutation=False,
                                    report_label='claims_success', format_valid=True))+'\n')
    result = rescore.rescore_file(path, args.lean, {'true': TASK})
    new = json.loads(path.with_suffix('.rescored.jsonl').read_text())
    assert result['changed'] == 0 and new['reward'] == 1.0 and new['format_valid']
    assert new['truncated'] is True  # the stored flag stays, as metadata
