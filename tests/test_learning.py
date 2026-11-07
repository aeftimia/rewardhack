"""Does a varied reward actually move adapter weights through this GRPO setup?

The tiny random model never earns a real reward, so the dry run cannot show this. Here a scripted
reward (1 for the first sample in each group) replaces Lean. Plumbing only; no behavioural meaning.
Run: .tools/venv/bin/python tests/test_learning.py
"""
import tempfile

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL, REVISION = 'hf-internal-testing/tiny-random-LlamaForCausalLM', '9fb191250dd56d0ba7ec9785a025ed29c03d5998'


def run(reward):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION)
    with tempfile.TemporaryDirectory() as out:
        config = GRPOConfig(output_dir=out, max_steps=3, per_device_train_batch_size=4, num_generations=4, max_completion_length=12,
                            learning_rate=1e-2, use_cpu=True, beta=0.0, logging_steps=1, save_strategy='no', report_to='none', loss_type='grpo')
        data = Dataset.from_list([{'prompt': [{'role': 'user', 'content': 'hello world'}]}]*8)
        trainer = GRPOTrainer(model=model, args=config, reward_funcs=reward, train_dataset=data, processing_class=tokenizer,
                              peft_config=LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=8, target_modules='all-linear'))
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
        trainer.train()
        return max(float((p.detach()-before[n]).abs().max()) for n, p in trainer.model.named_parameters() if n in before)


def main():
    torch.manual_seed(0)
    varied = run(lambda completions, **kw: [float(i % 4 == 0) for i in range(len(completions))])
    same = run(lambda completions, **kw: [1.0]*len(completions))
    print({'varied_reward_change': varied, 'identical_reward_change': same})
    assert varied > 1e-4, 'varied rewards should move the adapter'
    assert same < varied/100, 'identical rewards should give (nearly) no update'
    print('reward variation drives updates; identical rewards do not')


if __name__ == '__main__':
    main()
