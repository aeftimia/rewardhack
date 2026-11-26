"""Owner's decisions of 26 November after the review against current RL practice (notes section 72): several prompt
groups per optimizer step, untruncated sampling, and LoRA on the gpt-oss experts as well as attention."""
from types import SimpleNamespace

import pytest

from rewardhack import hf_grpo
from test_training_integration import GENERATIONS, REWARDS, train


def test_several_prompts_make_one_step_of_one_group_per_micro_batch(tmp_path, monkeypatch):
    run = train(tmp_path, monkeypatch, REWARDS['depends on the text'], steps=2, prompts_per_step=2)
    effect = run.summary['trainer']['arguments_in_effect']
    assert (effect['per_device_train_batch_size'], effect['gradient_accumulation_steps']) == (GENERATIONS, 2)
    # Both groups of a step are sampled and scored together, as one generation batch, and the optimizer steps once.
    assert [row['step'] for row in run.rows] == [1]*2*GENERATIONS + [2]*2*GENERATIONS
    assert len(run.log) == 2


def test_sampling_is_untruncated(tmp_path, monkeypatch):
    effect = train(tmp_path, monkeypatch, REWARDS['first of the group']).summary['trainer']['arguments_in_effect']
    assert (effect['temperature'], effect['top_p'], effect['top_k']) == (1.0, 1.0, 0)


def test_gpt_oss_experts_carry_lora_and_learn(tmp_path, monkeypatch):
    from test_training_integration import tiny
    built = tiny('moe')
    run = train(tmp_path, monkeypatch, REWARDS['first of the group'], kind='moe', built=built)
    assert 'experts' in run.summary['trainer']['lora_target_modules']
    # LoRA B starts at zero; after a step with reward variation the expert B matrices have moved.
    trained = {name: p for name, p in built[0].named_parameters() if 'experts' in name and 'lora_B' in name}
    assert len(trained) == 2 and all(float(p.detach().abs().max()) > 0 for p in trained.values())
    model, _ = tiny('moe')
    from peft import get_peft_model
    peft = get_peft_model(model, hf_grpo.lora_config(model, 2))
    expert = [name for name, p in peft.named_parameters() if p.requires_grad and 'experts' in name]
    assert len(expert) == 4  # lora_A and lora_B for gate_up_proj and for down_proj
    assert run.summary['trainable_parameters'] == sum(p.numel() for p in peft.parameters() if p.requires_grad)


def test_an_unlisted_architecture_is_refused():
    with pytest.raises(KeyError):
        hf_grpo.lora_config(SimpleNamespace(config=SimpleNamespace(model_type='unlisted')), 2)


def test_a_dropped_row_leaves_the_kept_rows_gradient_as_if_never_sampled(tmp_path, monkeypatch):
    """Under 'dapo' TRL divides by the step's token count, counted before the verifier mask. A dropped long row must
    leave that count too, or it shrinks every kept row's gradient (review round 1, 26 Nov)."""
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer
    backend = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2, 'hello': 3}, unk_token='<unk>'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_layer=1, n_head=1, n_embd=8, bos_token_id=3, eos_token_id=1, pad_token_id=0))
    config = GRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to='none', num_generations=2,
                        per_device_train_batch_size=2, max_completion_length=4, beta=0, loss_type='dapo')

    class MaskedTrainer(hf_grpo.VerifierMaskMixin, GRPOTrainer):
        pass

    trainer = MaskedTrainer(model=model, args=config, reward_funcs=lambda completions, **kw: [0.]*len(completions),
                            train_dataset=Dataset.from_list([{'prompt': 'hello'}]), processing_class=tokenizer)
    trainer.model.eval()  # no accumulation rescale, so the two losses below compare directly
    lengths = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])  # the dropped row is long, the kept row short
    batch = dict(prompt_ids=torch.tensor([[3], [3]]), prompt_mask=torch.ones((2, 1), dtype=torch.long),
                 completion_ids=torch.tensor([[3, 3, 3, 1], [1, 0, 0, 0]]), completion_mask=lengths,
                 advantages=torch.tensor([1., 1.]), num_items_in_batch=lengths.sum())

    def generated(self, inputs):
        self._verifier_keep = [False, True]
        return dict(batch)

    monkeypatch.setattr(GRPOTrainer, '_generate_and_score_completions', generated)
    masked = trainer._generate_and_score_completions([])
    assert int(masked['num_items_in_batch']) == 1
    both = torch.full((2, 4), -2., requires_grad=True)
    monkeypatch.setattr(trainer, '_get_per_token_logps_and_entropies', lambda *a, **k: (both, torch.zeros_like(both), None))
    trainer._compute_loss(model, masked).backward()
    alone = torch.full((1, 4), -2., requires_grad=True)
    monkeypatch.setattr(trainer, '_get_per_token_logps_and_entropies', lambda *a, **k: (alone, torch.zeros_like(alone), None))
    trainer._compute_loss(model, {key: value if key == 'num_items_in_batch' else value[1:] for key, value in masked.items()}).backward()
    assert both.grad[0].abs().sum() == 0
    torch.testing.assert_close(both.grad[1], alone.grad[0])


def test_evaluation_samples_from_the_vllm_server_when_the_run_has_one(tmp_path, monkeypatch):
    """With a server, both evaluations go through it (owner, 26 Nov): the before-evaluation on the base model (no adapter),
    the after-evaluation on the trained adapter, from the same seeds; and the trainer's own generate is never called."""
    import torch
    from test_training_integration import TASKS, tiny
    model, tokenizer = tiny('plain')
    monkeypatch.setattr(model, 'generate', lambda **kw: pytest.fail('trainer-side generation with a server present'))
    from rewardhack import outcomes
    template = outcomes.score('unused', TASKS[0], '', 'baseline', truncated=True)  # a cut-off row needs neither Lean nor a judge
    monkeypatch.setattr(hf_grpo, 'score', lambda lean, task, raw, *a, **kw: dict(template, task_id=task['id'], group=task['group'], raw=raw))
    calls = []
    class Server:
        def sample(self, adapter_model, prompts, *, seed, record):
            calls.append((adapter_model, len(prompts), seed, record))
            return [[3, 1] for _ in prompts], None, None
    args = SimpleNamespace(out=tmp_path, lean='unused', arm='baseline', penalty=0, judge_fn=None, report_mode='prose', verifier_retry_seconds=60,
                           eval_samples=2, seed=7, prompt_variant='neutral', thinking='unsupported', reasoning_effort=None, max_completion_length=6)
    adapter = object()
    for stage, adapter_model in [('before', None), ('after', adapter)]:
        result = hf_grpo.evaluate(model, tokenizer, TASKS[:2], args, stage, rollouts=Server(), adapter_model=adapter_model)
        assert result['tasks_completed'] == 2 and len((tmp_path/f'{stage}-evaluation.jsonl').read_text().splitlines()) == 4
    assert [(call[0], call[1], call[3]) for call in calls] == [(None, 4, dict(evaluation='before')), (adapter, 4, dict(evaluation='after'))]
    assert calls[0][2] == calls[1][2] == hf_grpo.evaluation_seed(args) * 10**6


def test_a_long_server_call_near_the_deadline_still_scores_every_sample_it_paid_for(tmp_path, monkeypatch):
    """All tasks are generated in the first server call; charging that call to task 1 stopped a late after-evaluation
    after one task and dropped the rest (review round 10, 26 Nov). Only scoring still to come counts against stop_at."""
    from rewardhack import outcomes
    from test_training_integration import TASKS, tiny
    model, tokenizer = tiny('plain')
    clock = [0.0]
    monkeypatch.setattr(hf_grpo.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(hf_grpo.time, 'perf_counter', lambda: clock[0])
    template = outcomes.score('unused', TASKS[0], '', 'baseline', truncated=True)
    def score(lean, task, raw, *a, **kw):
        clock[0] += 1.0  # each answer takes a second to score
        return dict(template, task_id=task['id'], group=task['group'], raw=raw)
    monkeypatch.setattr(hf_grpo, 'score', score)
    class Server:
        def sample(self, adapter_model, prompts, *, seed, record):
            clock[0] += 300.0  # the one call for every task
            return [[3, 1] for _ in prompts], None, None
    args = SimpleNamespace(out=tmp_path, lean='unused', arm='baseline', penalty=0, judge_fn=None, report_mode='prose', verifier_retry_seconds=60,
                           eval_samples=2, seed=7, prompt_variant='neutral', thinking='unsupported', reasoning_effort=None, max_completion_length=6)
    result = hf_grpo.evaluate(model, tokenizer, TASKS[:4], args, 'after', stop_at=320.0, rollouts=Server(), adapter_model=object())
    assert result['tasks_completed'] == 4 and not result['incomplete'] and result['generation_seconds'] == 300.0
    # With too little time left for the scoring itself, it still stops between tasks.
    clock[0] = 0.0
    assert hf_grpo.evaluate(model, tokenizer, TASKS[:4], args, 'after', stop_at=303.0, rollouts=Server(), adapter_model=object())['incomplete']


def test_concurrent_scoring_keeps_item_order_and_every_finished_result_on_a_failure(monkeypatch):
    """Scoring runs on a thread pool (owner, 26 Nov). Results come back in item order whatever order they finish in, and a
    failure leaves the results that did finish, so rows already paid for are kept as a partial batch."""
    import threading
    import time
    from rewardhack import outcomes
    started = threading.Barrier(4)
    def slow_first(item):
        started.wait(timeout=10)  # all four run at once, or this test would deadlock and time out
        time.sleep(0.05 * (4 - item))
        return item * 10
    assert outcomes.score_concurrently(slow_first, [0, 1, 2, 3]) == [0, 10, 20, 30]
    def fails_on_two(item):
        if item == 2:
            raise RuntimeError('judge spend cap reached')
        return item
    with pytest.raises(outcomes.ScoringStopped) as stopped:
        outcomes.score_concurrently(fails_on_two, [0, 1, 2, 3])
    # Item 3 either finished first or was cancelled when item 2 failed: either way only finished results come back, in order.
    assert stopped.value.done in ([0, 1], [0, 1, 3]) and 'spend cap' in str(stopped.value.__cause__)
    # Calls not yet started when one fails are cancelled, not run (review round 13): one worker runs them in order.
    calls = []
    def logged(item):
        calls.append(item)
        if item > 2:
            time.sleep(0.3)  # long enough that the failure is seen while these wait in the queue
        return fails_on_two(item)
    monkeypatch.setattr(outcomes, 'SCORING_WORKERS', 1)
    with pytest.raises(outcomes.ScoringStopped) as stopped:
        outcomes.score_concurrently(logged, [0, 1, 2, 3, 4, 5])
    # Item 3 may already be running when the failure is seen; 4 and 5 never start.
    assert stopped.value.done in ([0, 1], [0, 1, 3]) and 4 not in calls and 5 not in calls


def test_the_retry_allowance_holds_under_concurrent_scoring():
    import threading
    from rewardhack import outcomes
    budget = outcomes.RetryBudget(60)
    takers = [threading.Thread(target=lambda: [budget.take(1) for _ in range(100)]) for _ in range(8)]
    for t in takers:
        t.start()
    for t in takers:
        t.join()
    assert budget.left == 0 and budget.denied == 740
