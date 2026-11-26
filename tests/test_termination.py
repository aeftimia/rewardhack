"""Termination metadata reaches scoring before any capped answer can earn reward."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from conftest import run_options

from rewardhack import hf_grpo, outcomes, rescore
from rewardhack.outcomes import RETRY_SECONDS
from rewardhack.rates import rate

TASK = dict(id='true', group='g', tier='easy', true=True, statement='True')
RAW = json.dumps(dict(established='True', proof_body='by trivial', report='Proved True.')) + ' unfinished prose'


@pytest.fixture
def args(tmp_path):
    return run_options(out=tmp_path, lean='must-not-run', arm='report_correct', penalty=.5, report_mode='prose', verifier_retry_seconds=RETRY_SECONDS,
                           judge_fn=lambda row: dict(claim='OK', raw=''), eval_samples=1, rollout_backend='transformers',
                           max_completion_length=4, prompt_variant='neutral', thinking='unsupported', reasoning_effort=None,
                           num_generations=1, prompts_per_step=1, steps=1, learning_rate=.01,
                           beta=0, seed=0, adapter=True, resume_from=None, save_every=99, stop_saturated=True, lora_rank=1, lr_schedule='constant', router_aux_loss_coef=0.0, scale_rewards='group', max_grad_norm=1.0)


def stub_model_tokenizer(tokens):
    torch = pytest.importorskip('torch')

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[9, 10])
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.ones(1))
        def parameters(self):
            return iter([self.weight])
        def named_modules(self):
            return iter([])
        def named_parameters(self):
            return iter([('weight', self.weight)])
        def modules(self):  # no adapter layers to merge for evaluation
            return iter([])
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
    summary = hf_grpo.evaluate(model, tokenizer, [TASK], args, 'before')
    row = json.loads((args.out/'before-evaluation.jsonl').read_text())
    assert row['raw'] == RAW and row['truncated'] is True
    assert row['reward'] == 0 and row['genuine'] is False and row['format_valid'] is False
    assert summary['no_answer'] == rate(1, 1)
    assert summary['genuine_true_all'] == rate(0, 1)


def test_training_uses_token_ids_even_when_json_is_valid(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 5])
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))

    class Trainer:
        def __init__(self, model, reward_funcs, **kwargs):
            # Token-level loss: per-answer length normalisation let the 26 Nov baseline drift to the token limit.
            assert kwargs['args']['loss_type'] == 'dapo'
            self.model, self.reward = model, reward_funcs
            self.state = SimpleNamespace(global_step=0, log_history=[])
            self.callback_handler = SimpleNamespace(callbacks=[SimpleNamespace()])  # the TensorBoard callback report_to adds
            self.args = SimpleNamespace(lr_scheduler_type='constant', to_dict=dict)  # what the real trainer reports back, recorded in the summary
            self.router_aux_loss_coef, self.aux_loss_enabled, self.generation_config = 0.0, False, SimpleNamespace(to_dict=dict)
        def train(self, resume_from_checkpoint):
            assert resume_from_checkpoint is None
            with pytest.raises(ValueError, match='completion_ids'):
                self.reward([RAW], ['True'])
            assert self.reward([RAW], ['True'], completion_ids=[[2, 10, 3, 4]]) == [0.0]
        def save_model(self, path):
            pass

    for name, attrs in {
        'datasets': dict(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
        'peft': dict(LoraConfig=lambda **kw: kw),
        'transformers': dict(TrainerCallback=object, set_seed=lambda seed: None), 'transformers.integrations': dict(TensorBoardCallback=SimpleNamespace),
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
    assert result['summary']['no_answer'] == rate(1, 1)


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
                         samples=1, max_tokens=4, variant='neutral', part='train', chunk_prompts=1)
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: 'prompt')
    class Tokenizer:
        chat_template = ''  # no clock and no reasoning switch to settle
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
    monkeypatch.setattr(vllm_probe.deadline, 'arm', lambda *a: None)
    monkeypatch.setattr(vllm_probe.deadline, 'disarm', lambda *a: None)
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    real, allowances = vllm_probe.score, []
    monkeypatch.setattr(vllm_probe, 'score', lambda *a, **kw: allowances.append(kw.get('retries')) or real(*a, **kw))
    vllm_probe.run(args)
    assert isinstance(allowances[0], outcomes.RetryBudget)  # second verifier checks are bounded on this path too
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
        assert result['summary']['no_answer'] == rate(1, 1)


def test_evaluation_overrides_model_only_stop_tokens(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 10, 3, 4])
    generate = model.generate
    def generate_checked(**kwargs):
        assert kwargs['eos_token_id'] == tokenizer.eos_token_id == 9
        return generate(**kwargs)
    model.generate = generate_checked
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    hf_grpo.evaluate(model, tokenizer, [TASK], args, 'before')
    row = json.loads((args.out/'before-evaluation.jsonl').read_text())
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
                        num_generations=2, per_device_train_batch_size=2, max_completion_length=4,
                        generation_kwargs={'eos_token_id': outcomes.stop_ids(model, tokenizer)})
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
                        num_generations=2, per_device_train_batch_size=2, max_completion_length=2, beta=0, loss_type='dapo')

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
                 tool_mask=torch.tensor([[1, 1], [1, 0]]), num_items_in_batch=torch.tensor(3))
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
    kept_gradient = logps.grad[1].clone()
    # Removing the dropped row entirely must produce the same kept-row gradient.
    # The step's token count must leave the dropped row's tokens out, so it is carried over as the masked batch has it.
    only_kept = {key: value if key == 'num_items_in_batch' else value[1:] for key, value in masked.items()}
    logps = torch.full((1, 2), -2., requires_grad=True)
    trainer._compute_loss(model, only_kept).backward()
    torch.testing.assert_close(logps.grad[0], kept_gradient)


def test_probe_out_of_time_keeps_whole_groups_and_says_incomplete(args, monkeypatch):
    """Measured on the first pod run: 12 groups needed 25 minutes of a 20-minute limit and the summary was lost."""
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    clock = [0.0]
    monkeypatch.setattr(hf_grpo.time, 'monotonic', lambda: clock[0])
    template = outcomes.score('unused', TASK, RAW, 'baseline', truncated=True)
    def slow_score(*a, **kw):
        clock[0] += 100  # one task takes 100 s
        return dict(template, reward=0., verifier_inconclusive=False)
    monkeypatch.setattr(hf_grpo, 'score', slow_score)
    tasks = [dict(TASK, id=f't{i}') for i in range(5)]
    summary = hf_grpo.evaluate(model, tokenizer, tasks, args, 'probe', stop_at=330)  # a 4th task would end at 400
    assert (summary['tasks_planned'], summary['tasks_completed'], summary['incomplete']) == (5, 3, True)
    assert len((args.out/'probe-evaluation.jsonl').read_text().splitlines()) == 3
    clock[0] = 0.0
    whole = hf_grpo.evaluate(model, tokenizer, tasks, args, 'probe', stop_at=10_000)
    assert whole['tasks_completed'] == 5 and not whole['incomplete']
    clock[0] = 0.0
    assert hf_grpo.evaluate(model, tokenizer, tasks, args, 'before')['tasks_completed'] == 5  # no limit: unchanged


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
    assert outcomes.training_rewards(rows, 2) == [2., None]
    saved = [json.loads(line) for line in (args.out/'probe-evaluation.jsonl').read_text().splitlines()]
    assert [r['reward'] for r in saved] == [2., 1.]


def trl_stub(monkeypatch, drive):
    """The trainer's reward function and step callback, driven by `drive(reward_fn, callback)` in place of TRL."""
    class Trainer:
        def __init__(self, model, reward_funcs, callbacks, **kwargs):
            self.model, self.reward, self.callbacks = model, reward_funcs, callbacks
            self.state = SimpleNamespace(global_step=0, log_history=[])
            self.callback_handler = SimpleNamespace(callbacks=[SimpleNamespace()])  # the TensorBoard callback report_to adds
            self.args = SimpleNamespace(lr_scheduler_type='constant', to_dict=dict)  # what the real trainer reports back, recorded in the summary
            self.router_aux_loss_coef, self.aux_loss_enabled, self.generation_config = 0.0, False, SimpleNamespace(to_dict=dict)
        def train(self, resume_from_checkpoint):
            assert resume_from_checkpoint is None
            drive(self.reward, self.callbacks[0])
        def save_model(self, path):
            pass
    for name, attrs in {'datasets': dict(Dataset=SimpleNamespace(from_list=lambda rows: rows)), 'peft': dict(LoraConfig=lambda **kw: kw),
                        'transformers': dict(TrainerCallback=object, set_seed=lambda seed: None), 'transformers.integrations': dict(TensorBoardCallback=SimpleNamespace), 'trl': dict(GRPOTrainer=Trainer, GRPOConfig=lambda **kw: kw)}.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)


def scripted_scores(monkeypatch, verdicts):
    """`hf_grpo.score` returns scripted (reward, undecided) rows, or raises a scripted exception; records its keywords."""
    template, seen, pending = outcomes.score('unused', TASK, RAW, 'baseline', truncated=True), [], iter(verdicts)
    def score(*a, **kw):
        seen.append(kw)
        verdict = next(pending)
        if isinstance(verdict, BaseException):
            raise verdict
        return dict(template, arm='report_correct', reward=verdict[0], verifier_inconclusive=verdict[1], truncated=False)
    monkeypatch.setattr(hf_grpo, 'score', score)
    return seen


def test_training_records_what_the_verifier_cost_it(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    args.num_generations, args.stop_no_variation, args.save_every, args.train_deadline = 2, 1, 99, float('inf')
    seen = scripted_scores(monkeypatch, [(0.0, False), (1.5, True)])
    def drive(reward, callback):
        assert reward([RAW, RAW], ['True', 'True'], completion_ids=[[2, 9], [2, 9]]) == [0.0, None]
        control = callback.on_step_end(None, SimpleNamespace(global_step=1, log_history=[]), SimpleNamespace(should_training_stop=False, should_save=False), model=None)
        assert control.should_training_stop and control.should_save  # a stopped run keeps a checkpoint of its last step
    trl_stub(monkeypatch, drive)
    summary = dict(rollouts={})
    hf_grpo.train(model, tokenizer, [TASK], args, summary)
    assert all(isinstance(kw['retries'], outcomes.RetryBudget) for kw in seen) and seen[0]['retries'] is seen[1]['retries']
    step = json.loads((args.out/'steps.jsonl').read_text())
    assert (step['unverified_rate'], step['check_verifier'], step['retries_denied'], step['variation_lost_to_verifier']) == (0.5, True, 0, 1)
    # No judge was asked in this scripted run, so its failure rate is unknown, not zero.
    assert summary['training_verifier'] == dict(unverified_rate=0.5, check_verifier=True, retries_denied=0, variation_lost_groups=1,
                                                judge_undecided_rate=None, judge_asked=0, judge_undecided=0, check_judge=None)
    assert summary['early_stop'] == outcomes.VERIFIER_UNDECIDED and json.loads((args.out/'live.json').read_text())['check_verifier'] is True


def test_no_variation_is_not_blamed_on_the_verifier(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    args.num_generations, args.stop_no_variation, args.save_every, args.train_deadline = 2, 1, 99, float('inf')
    scripted_scores(monkeypatch, [(0.0, False), (0.0, False)])  # every rollout decided, and all the same
    def drive(reward, callback):
        reward([RAW, RAW], ['True', 'True'], completion_ids=[[2, 9], [2, 9]])
        callback.on_step_end(None, SimpleNamespace(global_step=1, log_history=[]), SimpleNamespace(should_training_stop=False, should_save=False), model=None)
    trl_stub(monkeypatch, drive)
    summary = dict(rollouts={})
    hf_grpo.train(model, tokenizer, [TASK], args, summary)
    assert summary['early_stop'] == outcomes.NO_VARIATION and summary['training_verifier']['unverified_rate'] == 0


def test_a_stop_mid_batch_keeps_and_marks_the_scored_rollouts(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    args.num_generations = 2
    scripted_scores(monkeypatch, [(1.0, False), TimeoutError('hard deadline')])
    def drive(reward, callback):
        with pytest.raises(TimeoutError):
            reward([RAW, RAW], ['True', 'True'], completion_ids=[[2, 9], [2, 9]])
    trl_stub(monkeypatch, drive)
    summary = dict(rollouts={})
    hf_grpo.train(model, tokenizer, [TASK], args, summary)
    saved = [json.loads(line) for line in (args.out/'training-rollouts.jsonl').read_text().splitlines()]
    assert len(saved) == 1 and saved[0]['partial_batch'] is True and summary['partial_batch_rows'] == 1
    assert summary['rollouts'] == {} and (args.out/'steps.jsonl').read_text() == ''


def test_evaluation_bounds_its_second_checks(args, monkeypatch):
    torch = pytest.importorskip('torch')
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    args.verifier_retry_seconds = 0
    seen = scripted_scores(monkeypatch, [(0.0, False)])
    summary = hf_grpo.evaluate(model, tokenizer, [TASK], args, 'probe')
    assert seen[0]['retries'].left == 0 and summary['verifier_retries_denied'] == 0


def test_rescore_keeps_the_partial_batch_mark(args, monkeypatch):
    monkeypatch.setattr(outcomes, 'evaluate_submission', lambda *a: pytest.fail('cutoff reached Lean'))
    path = args.out/'rollouts.jsonl'
    path.write_text(json.dumps(dict(task_id='true', raw=RAW, arm='baseline', genuine_why='truncated', truncated=True, partial_batch=True))+'\n')
    rescore.rescore_file(path, args.lean, {'true': TASK})
    assert json.loads(path.with_suffix('.rescored.jsonl').read_text())['partial_batch'] is True


def test_saturation_needs_a_full_window_and_credibly_few_mixed_groups():
    window = outcomes.SATURATION_WINDOW
    assert not outcomes.saturated([False]*(window-1), 1)  # too few groups yet
    assert outcomes.saturated([True]*40 + [False]*window, 1)  # early variation is outside the window
    assert outcomes.saturated([True] + [False]*(window-1), 1)  # 1 mixed in 80
    assert not outcomes.saturated([True]*2 + [False]*(window-2), 1)  # 2 in 80 is not credibly under 5%


def test_saturation_takes_whole_steps():
    """80 groups at 3 a step is 26 steps and 2 groups; the window takes 27 whole steps, so the mixed groups listed first
    in the oldest step are counted rather than cut off."""
    oldest = [True, True, False]
    assert not outcomes.saturated(oldest + [False]*78, 3)  # 2 mixed in 81


def test_the_26_nov_pair_stops_the_baseline_and_not_lambda0():
    """The calibration in outcomes.py's comment: mixed groups per step of the 60-step pair (4 groups a step)."""
    runs = dict(baseline=([2, 3, 3, 4, 1, 2, 3, 1, 4, 2, 3, 3, 1, 2, 4, 1, 3, 2, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 48),
                lambda0=([3, 4, 3, 3, 4, 4, 3, 4, 4, 2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                                1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 0], None))
    for mixed, expected in runs.values():
        groups, stop = [], None
        for step, m in enumerate(mixed, 1):
            groups.extend([True]*m + [False]*(4-m))
            if stop is None and outcomes.saturated(groups, 4):
                stop = step
        assert stop == expected


@pytest.mark.parametrize('on', [True, False])
def test_a_saturated_run_stops_only_when_the_stop_is_on(args, monkeypatch, on):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    args.num_generations, args.stop_no_variation, args.save_every, args.train_deadline, args.stop_saturated = 2, 1, 99, float('inf'), on
    # Two groups can never be credible at 0.95: P(share < 5%) after 0 of 2 is 0.41, after 1 of 2 almost 0.
    monkeypatch.setattr(outcomes, 'SATURATION_WINDOW', 2)
    monkeypatch.setattr(outcomes, 'SATURATION_CONFIDENCE', 0.3)
    scripted_scores(monkeypatch, [(0.0, False), (1.0, False)] + [(0.0, False)]*4)  # one mixed group, then two without
    stops = []
    def drive(reward, callback):
        for step in (1, 2, 3):
            reward([RAW, RAW], ['True', 'True'], completion_ids=[[2, 9], [2, 9]])
            control = callback.on_step_end(None, SimpleNamespace(global_step=step, log_history=[]), SimpleNamespace(should_training_stop=False, should_save=False), model=None)
            stops.append(control.should_training_stop)
    trl_stub(monkeypatch, drive)
    summary = dict(rollouts={})
    hf_grpo.train(model, tokenizer, [TASK], args, summary)
    assert stops == [False, False, on] and (summary.get('early_stop') == outcomes.SATURATED) == on


@pytest.mark.parametrize('scores, stop', [([(0.0, False)]*2, outcomes.NO_VARIATION),
                                           ([(0.0, False), (1.0, False)] + [(0.0, False)]*4, outcomes.SATURATED)])
def test_a_deadline_does_not_overwrite_an_earlier_stop(args, monkeypatch, scores, stop):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    args.num_generations, args.stop_no_variation, args.save_every, args.stop_saturated = 2, 1, 99, True
    args.train_deadline = float('inf')
    monkeypatch.setattr(outcomes, 'SATURATION_WINDOW', 2)
    monkeypatch.setattr(outcomes, 'SATURATION_CONFIDENCE', 0.3)
    scripted_scores(monkeypatch, scores)
    def drive(reward, callback):
        for step in range(1, len(scores)//2 + 1):
            if step == len(scores)//2:
                args.train_deadline = 0.0  # the deadline passes at the step the other stop fires
            reward([RAW, RAW], ['True', 'True'], completion_ids=[[2, 9], [2, 9]])
            callback.on_step_end(None, SimpleNamespace(global_step=step, log_history=[]), SimpleNamespace(should_training_stop=False, should_save=False), model=None)
    trl_stub(monkeypatch, drive)
    summary = dict(rollouts={})
    hf_grpo.train(model, tokenizer, [TASK], args, summary)
    assert summary['early_stop'] == stop  # the deadline has passed too, but this decides the resume


def test_a_run_that_never_varied_gets_the_no_variation_stop_not_saturation(args, monkeypatch):
    model, tokenizer = stub_model_tokenizer([2, 3, 4, 9])
    args.num_generations, args.stop_no_variation, args.save_every, args.train_deadline, args.stop_saturated = 2, 3, 99, float('inf'), True
    # At window 2 and 0.3, two groups without variation count as saturated (0.41 > 0.3): without the guard this run would
    # stop at step 2 for saturation.
    monkeypatch.setattr(outcomes, 'SATURATION_WINDOW', 2)
    monkeypatch.setattr(outcomes, 'SATURATION_CONFIDENCE', 0.3)
    assert outcomes.saturated([False, False], 1)
    scripted_scores(monkeypatch, [(0.0, False)]*6)
    stops = []
    def drive(reward, callback):
        for step in (1, 2, 3):
            reward([RAW, RAW], ['True', 'True'], completion_ids=[[2, 9], [2, 9]])
            stops.append(callback.on_step_end(None, SimpleNamespace(global_step=step, log_history=[]), SimpleNamespace(should_training_stop=False, should_save=False), model=None).should_training_stop)
    trl_stub(monkeypatch, drive)
    summary = dict(rollouts={})
    hf_grpo.train(model, tokenizer, [TASK], args, summary)
    assert stops == [False, False, True] and summary['early_stop'] == outcomes.NO_VARIATION


def test_a_probe_stopped_between_chunks_keeps_the_finished_ones(args, monkeypatch):
    """One generate call over every prompt kept nothing when its deadline struck (review round 26): the gpt-oss-120b stages
    are now sampled and saved a chunk of prompts at a time."""
    from rewardhack import vllm_probe
    args.__dict__.update(template_kwargs='{}', max_seconds=10, taskset='v1', tasks=None, model='stub', revision='stub',
                         max_model_len=10, gpu_memory_utilization=.5, samples=2, max_tokens=4, variant='neutral', part='train',
                         chunk_prompts=2, lean='none')

    class Tokenizer:
        chat_template = ''
        def apply_chat_template(self, *a, **kw):
            return 'prompt'
        def __call__(self, text):
            return dict(input_ids=[1])

    calls, seeds = [], []
    class LLM:
        def __init__(self, **kwargs):
            pass
        def get_tokenizer(self):
            return Tokenizer()
        def generate(self, prompts, params):
            calls.append(len(prompts))
            seeds.extend(p['seed'] for p in params)
            if len(calls) == 2:
                raise TimeoutError('deadline')
            return [SimpleNamespace(outputs=[SimpleNamespace(text=RAW, token_ids=[2], finish_reason='stop')] * 2) for _ in prompts]
    module = ModuleType('vllm')
    module.__dict__.update(LLM=LLM, SamplingParams=lambda **kw: kw, __version__='stub')
    monkeypatch.setitem(sys.modules, 'vllm', module)
    monkeypatch.setattr(vllm_probe, 'GpuSampler', lambda: SimpleNamespace(start=lambda: None, summary=lambda: {}))
    monkeypatch.setattr(vllm_probe, 'select_tasks', lambda *a: [dict(TASK, id=f't{i}') for i in range(3)])
    monkeypatch.setattr(vllm_probe.deadline, 'arm', lambda *a: None)
    monkeypatch.setattr(vllm_probe.deadline, 'disarm', lambda *a: None)
    with pytest.raises(TimeoutError):
        vllm_probe.run(args)
    assert calls == [2, 1]
    # Each prompt has its own seed, n apart, so no two samples share one (review round 28; args.seed is 0, samples 2).
    assert seeds == [0, 2, 4]
    rows = [json.loads(line) for line in (args.out/'rows.jsonl').read_text().splitlines()]
    assert [row['task_id'] for row in rows] == ['t0', 't0', 't1', 't1']
    summary = json.loads((args.out/'summary.json').read_text())
    assert summary['status'] == 'failed' and summary['rows_written'] == 4


@pytest.mark.parametrize('size', [0, -1])
def test_a_probe_refuses_a_chunk_size_below_one_before_anything_loads(args, size):
    """0 failed only after a multi-minute model load; a negative size saved no rows and reported success (round 28)."""
    from rewardhack import vllm_probe
    args.__dict__.update(chunk_prompts=size, out=args.out/'never')
    with pytest.raises(ValueError, match='at least 1'):
        vllm_probe.run(args)
    assert not args.out.exists()
