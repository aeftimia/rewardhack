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


def test_evaluation_merges_the_adapter_and_restores_the_weights(monkeypatch):
    """PEFT rebuilt the gpt-oss expert stacks on every decode step, about 8x slower (measuring pass, 26 Nov); evaluation now
    runs with the LoRA merged. It must see the adapted model, and leave the weights and adapter as they were."""
    import torch
    from peft import get_peft_model
    from test_training_integration import tiny
    model, _ = tiny('moe')
    peft = get_peft_model(model, hf_grpo.lora_config(model, 2))
    with torch.no_grad():
        for name, p in peft.named_parameters():
            if 'lora_B' in name:
                p.normal_()  # a trained adapter, so merging changes the output
    ids = torch.tensor([[3, 4, 5]])
    peft.eval()
    with torch.no_grad():
        adapted = peft(ids).logits
    weights = {name: p.detach().clone() for name, p in peft.named_parameters()}
    seen = []
    def score_samples(model, *args):
        with torch.no_grad():
            seen.append(model(ids).logits)
        return 'scored'
    monkeypatch.setattr(hf_grpo, 'score_samples', score_samples)
    assert hf_grpo.evaluate(model, None, [], None, 'after') == 'scored'
    torch.testing.assert_close(seen[0], adapted, rtol=1e-4, atol=1e-4)  # merged, the model is the adapted one
    assert not any(getattr(m, 'merged', False) for m in peft.modules())
    for name, p in peft.named_parameters():
        torch.testing.assert_close(p.detach(), weights[name], rtol=1e-5, atol=1e-6)


def test_evaluation_merges_only_the_expert_lora_so_a_small_attention_update_is_not_rounded_away(monkeypatch):
    """Merging an attention LoRA folds it into bf16 weights, where a one-step update mostly rounds away; unmerged it adds
    its branch to the output, as training and vLLM do (review round 9, 26 Nov). Only the expert ParamWrappers merge."""
    import torch
    from peft import get_peft_model
    from peft.tuners.lora.layer import ParamWrapper
    from test_training_integration import tiny
    model, _ = tiny('moe')
    peft = get_peft_model(model, hf_grpo.lora_config(model, 2))
    merged = []
    def score_samples(model, *args):
        merged.extend(type(m).__name__ for m in model.modules() if getattr(m, 'merged', False))
        return 'scored'
    monkeypatch.setattr(hf_grpo, 'score_samples', score_samples)
    hf_grpo.evaluate(model, None, [], None, 'after')
    assert merged and set(merged) == {ParamWrapper.__name__}
