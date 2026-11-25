"""The whole of `hf_grpo.run` on a tiny model, and three properties the second review round (2026-11-25) found unpinned:
a non-finite loss stops the run before it damages the adapter, the prompts the trainer generates from are the ones
evaluation uses, and `run` builds the trainer before it pays for the before-evaluation. CPU only; scoring is scripted."""
import json
import math
import subprocess
import time
from types import SimpleNamespace

import pytest

from rewardhack import hf_grpo, outcomes
from rewardhack.prompts import prompt
from test_training_integration import REWARDS, TASKS, train

torch = pytest.importorskip('torch')
pytest.importorskip('trl')


def model_with_task_words():
    """A tokenizer that tells the test tasks apart: with the shared tiny one every prompt word is <unk> and all prompts are equal."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    words = ['<pad>', '<eos>', '<unk>', 'statement', '0', '1', '2', '3', 'w0'] + [f'x{i}' for i in range(8)]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tokenizer.padding_side = 'left'
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}w0"
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=len(words), n_layer=1, n_head=2, n_embd=16, bos_token_id=2, eos_token_id=1, pad_token_id=0)), tokenizer


def test_a_non_finite_loss_stops_the_run_before_the_update(tmp_path, monkeypatch):
    """The stop has to act before the optimiser step. An earlier version read the step log, which at that point still held
    the previous step: training went on for another paid batch and saved an adapter full of NaN."""
    from trl import GRPOTrainer
    real, calls = GRPOTrainer._compute_loss, [0]

    def poisoned(self, model, inputs):
        calls[0] += 1
        loss = real(self, model, inputs)
        return loss*float('nan') if calls[0] == 2 else loss
    monkeypatch.setattr(GRPOTrainer, '_compute_loss', poisoned)
    with pytest.raises(FloatingPointError, match='step 2'):
        train(tmp_path, monkeypatch, REWARDS['first of the group'], steps=4)
    out = next(tmp_path.iterdir())
    assert len((out/'training-rollouts.jsonl').read_text().splitlines()) == 8, 'training generated another batch after the bad loss'
    from safetensors.torch import load_file
    saved = load_file(str(out/'adapter-at-failure'/'adapter_model.safetensors'))
    assert all(bool(torch.isfinite(tensor).all()) for tensor in saved.values()), 'the adapter saved at the failure is not finite'
    assert any(float(tensor.abs().max()) > 0 for tensor in saved.values()), 'the one good update is missing from the saved adapter'


def test_the_trainer_generates_from_the_prompts_evaluation_uses(tmp_path, monkeypatch):
    model, tokenizer = model_with_task_words()
    expected = {task['id']: tokenizer.apply_chat_template(prompt(task['statement'], 'neutral', 'prose'), tokenize=True, add_generation_prompt=True)['input_ids']
                for task in TASKS}
    assert len({tuple(ids) for ids in expected.values()}) == len(TASKS), 'this tokenizer does not tell the tasks apart'
    seen, generate = [], model.generate

    def recording(*args, **kwargs):
        for row, mask in zip(kwargs['input_ids'].tolist(), kwargs['attention_mask'].tolist()):
            seen.append(tuple(token for token, kept in zip(row, mask) if kept))
        return generate(*args, **kwargs)
    model.generate = recording
    run = train(tmp_path, monkeypatch, REWARDS['first of the group'], steps=3, built=(model, tokenizer))
    by_task = {task: tuple(ids) for task, ids in expected.items()}
    assert len(seen) == len(run.rows) == 12
    assert seen == [by_task[row['task_id']] for row in run.rows], 'a rollout was generated from a prompt other than its task\'s evaluation prompt'


def run_args(tmp_path, **overrides):
    args = SimpleNamespace(
        mode='train', out=tmp_path/'run', lean='lean-stub', model='tiny', revision=None, quant='none', mxfp4='absent', attn_implementation='eager',
        experts_implementation=None, thinking='unsupported', reasoning_effort=None, arm='baseline', penalty=0.0, beta=0.0, prompt_variant='neutral',
        report_mode='prose', taskset='hard', adapter=None, steps=2, prompts_per_step=1, num_generations=4, learning_rate=1e-2, lr_schedule='constant',
        scale_rewards='group', max_grad_norm=1.0, router_aux_loss_coef=0.0, lora_rank=2, max_completion_length=6, eval_samples=2, eval_limit=2,
        train_limit=4, save_every=99, stop_no_variation=99, seed=3, max_seconds=600, final_eval_seconds=60,
        judge='none', judge_model=None, judge_revision=None, judge_url='http://127.0.0.1:1/v1', judge_key_env='UNUSED', judge_usd_in=0.0, judge_usd_out=0.0,
        judge_extra={}, judge_max_usd=None, dry_run=False)
    vars(args).update(overrides)
    return args


@pytest.fixture
def staged(monkeypatch):
    """`run` with the model, Lean and the GPU sampler replaced, and a log of the order things happen in."""
    events = []
    model, tokenizer = model_with_task_words()
    monkeypatch.setattr(hf_grpo, 'load_model', lambda args, summary: (model, tokenizer))
    monkeypatch.setattr(hf_grpo, 'controls', lambda lean: {'stub': True})
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'Lean (stub)\n')
    monkeypatch.setattr(hf_grpo, 'GpuSampler', lambda: SimpleNamespace(start=lambda: None, summary=lambda: dict(gpu_sampler_error='no GPU in tests', gpu_peak_mib=0, gpu_max_util=0)))
    template = outcomes.score('unused', TASKS[0], '', 'baseline', truncated=True)
    state = dict(scored=0, fail_at=None)

    def score(lean, task, raw, arm, penalty, judge_fn, **kwargs):
        state['scored'] += 1
        if state['fail_at'] is not None and state['scored'] >= state['fail_at']:
            raise RuntimeError('scripted failure in scoring')
        return dict(template, task_id=task['id'], group=task['group'], true=task['true'], tier=task['tier'], raw=raw, truncated=False, format_valid=True,
                    arm=arm, verifier_inconclusive=False, reward=float(state['scored'] % 2))
    monkeypatch.setattr(hf_grpo, 'score', score)
    prepare, evaluate = hf_grpo.prepare_training, hf_grpo.evaluate
    monkeypatch.setattr(hf_grpo, 'prepare_training', lambda *a, **kw: (events.append('trainer built'), prepare(*a, **kw))[1])
    monkeypatch.setattr(hf_grpo, 'evaluate', lambda model, tokenizer, tasks, args, stage, **kw: (events.append(stage), evaluate(model, tokenizer, tasks, args, stage, **kw))[1])
    return events, state


def test_run_builds_the_trainer_first_and_records_what_it_did(staged, tmp_path):
    events, state = staged
    args = run_args(tmp_path)
    hf_grpo.run(args)
    assert events == ['trainer built', 'before', 'after'], 'the before-evaluation was paid for before the trainer was known to build'
    summary = json.loads((args.out/'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['global_step'] == 2
    assert summary['before']['samples'] == summary['after']['samples'] == 4 and not summary['after']['incomplete']
    assert summary['trainer']['router_aux_loss_coef'] == 0.0 and summary['trainer']['lr_schedule'] == 'constant'
    assert summary['trainer']['arguments_in_effect']['max_grad_norm'] == 1.0 and summary['trainer']['arguments_in_effect']['seed'] == 3
    for name in ('before-evaluation.jsonl', 'after-evaluation.jsonl', 'training-rollouts.jsonl', 'steps.jsonl', 'training-log.json', 'step-seconds.json', 'tasks.json'):
        assert (args.out/name).exists(), name
    assert (args.out/'adapter'/'adapter_config.json').exists()
    assert math.isfinite(summary['elapsed_seconds'])


def test_a_run_that_fails_in_training_still_writes_its_summary_and_adapter(staged, tmp_path):
    events, state = staged
    args = run_args(tmp_path, steps=4)
    state['fail_at'] = 4+4+2  # the before-evaluation scores 4, the first training step 4, then the second step fails on its second rollout
    with pytest.raises(RuntimeError, match='scripted failure'):
        hf_grpo.run(args)
    summary = json.loads((args.out/'summary.json').read_text())
    assert summary['status'] == 'failed' and 'scripted failure' in summary['reason']
    assert summary['global_step'] == 1 and 'before' in summary and 'after' not in summary
    assert (args.out/'adapter-at-failure'/'adapter_config.json').exists()
    rows = [json.loads(line) for line in (args.out/'training-rollouts.jsonl').read_text().splitlines()]
    assert [bool(row.get('partial_batch')) for row in rows] == [False]*4+[True], 'the rollout scored before the failure was lost or not marked'


def test_a_run_out_of_time_shortens_training_not_the_record(staged, tmp_path, monkeypatch):
    events, state = staged
    args = run_args(tmp_path, steps=50, max_seconds=600, final_eval_seconds=600)  # no time for training beyond the step in progress
    hf_grpo.run(args)
    summary = json.loads((args.out/'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['early_stop'] == 'deadline' and summary['global_step'] == 1
    assert events == ['trainer built', 'before', 'after'] and summary['after']['samples'] == 4


def test_the_judge_failure_test_works_at_the_size_of_a_real_run():
    """`math.comb` times a float overflowed past about 1,030 readings, so one unreadable reply would have ended every long run."""
    from rewardhack.outcomes import judge_failing, unlikely_at_the_limit
    def readings(total, every):
        """`total` readings with one failure in each `every`, spread evenly as an unreliable judge's would be."""
        return [dict(judge_asked=True, judge_undecided=i % every == every-1) for i in range(total)]
    assert judge_failing(readings(5000, 5000)) is False  # one failure in five thousand
    assert judge_failing(readings(5000, 50)) is False, 'a 2% judge was stopped'
    assert judge_failing(readings(5000, 10)) is True, 'a 10% judge was not stopped'
    # Rows the judge was never asked about are no evidence either way.
    assert judge_failing([dict(judge_undecided=True)]*50) is False
    assert unlikely_at_the_limit(20, 1) is False and unlikely_at_the_limit(3, 3) is True and unlikely_at_the_limit(10, 6) is True
    assert unlikely_at_the_limit(40, 0) is False


def test_a_judge_that_starts_failing_late_is_still_stopped():
    from rewardhack.outcomes import judge_failing
    good = [dict(judge_asked=True, judge_undecided=False)]*4000
    failing = [dict(judge_asked=True, judge_undecided=i % 3 == 0) for i in range(200)]
    assert judge_failing(good) is False
    assert judge_failing(good+failing) is True, 'four thousand good readings hid a judge now failing a third of the time'


def test_non_finite_gradients_stop_the_run_before_they_reach_the_adapter(tmp_path, monkeypatch):
    from trl import GRPOTrainer
    real = GRPOTrainer._compute_loss

    def poisoned(self, model, inputs):
        loss = real(self, model, inputs)
        weight = next(p for p in model.parameters() if p.requires_grad)
        return loss+torch.sqrt((weight*0).sum())  # value 0, gradient infinity times zero: a finite loss with NaN gradients
    monkeypatch.setattr(GRPOTrainer, '_compute_loss', poisoned)
    with pytest.raises(FloatingPointError, match='non-finite gradient at step 1'):
        train(tmp_path, monkeypatch, REWARDS['first of the group'], steps=3)
    from safetensors.torch import load_file
    saved = load_file(str(next(tmp_path.iterdir())/'adapter-at-failure'/'adapter_model.safetensors'))
    assert all(bool(torch.isfinite(tensor).all()) for tensor in saved.values())


def test_an_overflowed_gradient_norm_stops_the_run(tmp_path, monkeypatch):
    """Finite gradients whose total norm overflows are clipped to zero and so pass the per-tensor check; the norm itself is the evidence."""
    from transformers import Trainer
    real = Trainer.log

    def log(self, logs, *args, **kwargs):
        if 'grad_norm' in logs:
            logs = dict(logs, grad_norm=float('inf'))
        return real(self, logs, *args, **kwargs)
    monkeypatch.setattr(Trainer, 'log', log)
    with pytest.raises(FloatingPointError, match='gradient norm is inf'):
        train(tmp_path, monkeypatch, REWARDS['first of the group'], steps=3)
