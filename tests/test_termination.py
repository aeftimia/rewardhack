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
            assert self.reward([RAW], ['True'], completion_ids=[[2, 10, 3, 4]]) == [0.0]
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


@pytest.mark.parametrize('scored', [False, True])
def test_rescore_reads_the_answer_under_a_legacy_cutoff_flag(args, monkeypatch, scored):
    """Rows saved before 24 Nov were flagged by looser tests while holding a complete answer: rescore from raw."""
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: dict(
        established='True', proof_body='by trivial', report='Proved True.', reward=1.0, compiles=True, diagnostic=''))
    monkeypatch.setattr(outcomes, 'verify_proof', lambda *a: dict(ok=True, axioms=[], why='ok'))
    monkeypatch.setattr(outcomes, 'verify_refutation', lambda *a: dict(ok=False, why='not a negation'))
    path = args.out/'legacy.jsonl'
    path.write_text(json.dumps(dict(task_id='true', raw=RAW, arm='baseline', truncated=True, genuine_why='ok', scored=scored,
                                    reward=1.0, grader_reward=1.0, genuine=True, exploit=False, verified_refutation=False,
                                    report_label='claims_success', format_valid=True))+'\n')
    result = rescore.rescore_file(path, args.lean, {'true': TASK})
    new = json.loads(path.with_suffix('.rescored.jsonl').read_text())
    assert result['changed'] == 0 and new['reward'] == 1.0 and new['format_valid']
    assert new['truncated'] is True  # the stored flag stays, as metadata


def test_fresh_unscored_cutoff_stays_zero_on_repeated_rescore(args, monkeypatch):
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    row = dict(task_id='true', raw=RAW, arm='baseline', scored=False, truncated=True)
    row['termination_schema'] = 'cutoff-v1'
    path = args.out/'fresh.jsonl'
    path.write_text(json.dumps(row)+'\n')
    for _ in range(2):
        result = rescore.rescore_file(path, args.lean, {'true': TASK})
        path = path.with_suffix('.rescored.jsonl')
        new = json.loads(path.read_text())
        assert new['reward'] == 0 and not new['format_valid'] and new['truncated']
        assert result['summary']['no_answer'] == dict(n=1, of=1)


def test_evaluation_overrides_model_only_stop_tokens(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 10, 3, 4])
    generate = model.generate
    def generate_checked(**kwargs):
        assert kwargs['eos_token_id'] == tokenizer.eos_token_id == 9
        return generate(**kwargs)
    model.generate = generate_checked
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    hf_grpo.evaluate(model, tokenizer, [TASK], args, 'test')
    row = json.loads((args.out/'test-evaluation.jsonl').read_text())
    assert row['truncated'] and row['reward'] == 0


def test_real_trl_generation_uses_same_stop_as_cutoff_detector(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, LogitsProcessorList
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    backend = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2, 'hello': 3, '<other>': 4}, unk_token='<unk>'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_layer=1, n_head=1, n_embd=8,
                                     bos_token_id=3, eos_token_id=[1, 4], pad_token_id=0))
    config = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to='none',
                        num_generations=2, per_device_train_batch_size=2, max_completion_length=4)
    trainer = GRPOTrainer(model=model, args=config, reward_funcs=lambda completions, **kw: [0.] * len(completions),
                          train_dataset=Dataset.from_list([{'prompt': 'hello'}]), processing_class=tokenizer)
    assert trainer.generation_config.eos_token_id == outcomes.stop_ids(model, tokenizer) == 1

    class EmitModelOnlyStop:
        def __call__(self, ids, scores):
            scores.fill_(-float('inf'))
            scores[:, 4] = 0
            return scores
    with torch.no_grad():
        output = model.generate(input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 1), dtype=torch.long),
                                generation_config=trainer.generation_config,
                                logits_processor=LogitsProcessorList([EmitModelOnlyStop()]))
    generated = output[0, 1:].tolist()
    assert generated == [4, 4, 4, 4]
    assert outcomes.completion_truncated(generated, outcomes.stop_ids(model, tokenizer), 4)


def test_api_raw_row_marks_termination_provenance():
    from rewardhack.openai_probe import unscored_row
    row = unscored_row(TASK, 0, dict(raw=RAW, usage={}, finish='length', model='stub'), 1, 'c', 0)
    assert row['termination_schema'] == 'cutoff-v1'
    assert rescore.scored_as_cut_off(row) is True


def test_rescore_reports_transport_errors_without_counting_them(args, monkeypatch):
    from rewardhack.openai_probe import error_row, unscored_row
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    error = error_row(TASK, 0, {'error': 'HTTP 429'}, 1, 'c')
    answer = unscored_row(TASK, 1, dict(raw=RAW, usage={}, finish='length', model='stub'), 1, 'd', 0)
    path = args.out/'mixed.jsonl'
    payload = ''.join(json.dumps(row)+'\n' for row in [error, answer])
    path.write_text(payload)
    result = rescore.rescore_file(path, args.lean, {'true': TASK})
    assert result['input_rows'] == 2 and result['rows'] == 1
    assert result['transport_errors'] == [dict(row=1, task_id='true', error='HTTP 429')]
    assert result['summary']['samples'] == 1
    assert path.read_text() == payload


def test_rescore_does_not_silently_drop_malformed_rows(args):
    path = args.out/'invalid.jsonl'
    path.write_text(json.dumps(dict(task_id='true', arm='baseline'))+'\n')
    with pytest.raises(ValueError, match='neither raw output nor a transport error'):
        rescore.rescore_file(path, args.lean, {'true': TASK})
    assert not path.with_suffix('.rescored.jsonl').exists()


def test_verifier_mask_removes_actual_trl_kl_gradient_without_changing_attention(tmp_path, monkeypatch):
    """Zero advantages alone leave TRL's KL active; the loss mask must remove both terms."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    backend = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2, 'hello': 3}, unk_token='<unk>'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_layer=1, n_head=1, n_embd=8,
                                     bos_token_id=3, eos_token_id=1, pad_token_id=0))
    config = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to='none',
                        num_generations=2, per_device_train_batch_size=2, max_completion_length=2, beta=0, loss_type='grpo')

    class MaskedTrainer(hf_grpo.VerifierMaskMixin, GRPOTrainer):
        pass

    trainer = MaskedTrainer(model=model, args=config, reward_funcs=lambda completions, **kw: [0.] * len(completions),
                            train_dataset=Dataset.from_list([{'prompt': 'hello'}]), processing_class=tokenizer)
    # Supply reference log probabilities directly, so the real KL path needs no checkpoint download.
    trainer.beta = .1
    trainer.current_gradient_accumulation_steps = 1
    batch = dict(prompt_ids=torch.tensor([[3], [3]]), prompt_mask=torch.ones((2, 1), dtype=torch.long),
                 completion_ids=torch.tensor([[3, 1], [3, 1]]), completion_mask=torch.ones((2, 2), dtype=torch.long),
                 advantages=torch.tensor([0., 0.]), ref_per_token_logps=torch.full((2, 2), -1.),
                 tool_mask=torch.tensor([[1, 1], [1, 0]]))
    seen_attention = []
    logps = torch.full((2, 2), -2., requires_grad=True)

    def log_probabilities(model, ids, attention_mask, *a, **kw):
        seen_attention.append(attention_mask.clone())
        return logps, torch.zeros_like(logps), None

    monkeypatch.setattr(trainer, '_get_per_token_logps_and_entropies', log_probabilities)
    loss = trainer._compute_loss(model, batch)
    loss.backward()
    assert logps.grad[0].abs().sum() > 0  # actual pinned TRL KL acts despite zero advantage
    logps.grad.zero_()

    def generated(self, inputs):
        self._verifier_keep = [False, True]
        return dict(batch, advantages=torch.tensor([1e-7, 0.]))  # normalization roundoff must also be removed

    monkeypatch.setattr(GRPOTrainer, '_generate_and_score_completions', generated)
    masked = trainer._generate_and_score_completions([])
    assert masked['advantages'].tolist() == [0., 0.]
    assert masked['tool_mask'].tolist() == [[0, 0], [1, 0]]
    assert torch.equal(masked['completion_mask'], batch['completion_mask'])
    loss = trainer._compute_loss(model, masked)
    loss.backward()
    assert logps.grad[0].tolist() == [0., 0.]
    assert logps.grad[1, 0].abs() > 0 and logps.grad[1, 1] == 0
    assert torch.equal(seen_attention[0], seen_attention[1])


def test_probe_reward_variation_matches_dropped_training_rows(args, monkeypatch):
    torch = pytest.importorskip('torch')
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    model.generate = lambda **kw: torch.tensor([[1, 2, 3, 4, 9], [1, 2, 3, 4, 9]])
    args.eval_samples = 2
    template = outcomes.score('unused', TASK, RAW, 'baseline', truncated=True)
    rows = [dict(template, arm='report_correct', reward=2., verifier_inconclusive=False),
            dict(template, arm='report_correct', reward=1., verifier_inconclusive=True)]
    pending = iter(rows)
    monkeypatch.setattr(hf_grpo, 'score', lambda *a, **kw: next(pending))
    summary = hf_grpo.evaluate(model, tokenizer, [TASK], args, 'probe')
    assert summary['mixed_reward_groups'] == 0 and summary['zero_advantage_groups'] == 1
    assert outcomes.training_rewards(rows, 2) == [2., 2.]
    saved = [json.loads(line) for line in (args.out/'probe-evaluation.jsonl').read_text().splitlines()]
    assert [r['reward'] for r in saved] == [2., 1.]
