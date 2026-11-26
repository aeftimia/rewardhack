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
    run = train(tmp_path, monkeypatch, REWARDS['first of the group'], kind='moe')
    assert 'experts' in run.summary['trainer']['lora_target_modules']
    from test_training_integration import tiny
    model, _ = tiny('moe')
    from peft import get_peft_model
    peft = get_peft_model(model, hf_grpo.lora_config(model, 2))
    expert = [name for name, p in peft.named_parameters() if p.requires_grad and 'experts' in name]
    assert len(expert) == 4  # lora_A and lora_B for gate_up_proj and for down_proj
    assert run.summary['trainable_parameters'] == sum(p.numel() for p in peft.parameters() if p.requires_grad)


def test_an_unlisted_architecture_is_refused():
    with pytest.raises(KeyError):
        hf_grpo.lora_config(SimpleNamespace(config=SimpleNamespace(model_type='unlisted')), 2)
