#!/usr/bin/env python3
"""GPU path (TRL GRPO): `controls`, `probe` or `train` on a rented GPU, recording cost-relevant timings.

`--prompt-variant hint` is a labelled mechanism check, never pooled with the main arms.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from rewardhack import judge, reporting, retry_options, vllm_rollouts
from rewardhack.hf_grpo import run
from rewardhack.outcomes import ARMS, SATURATION_FLOOR, SATURATION_WINDOW
from rewardhack.prompts import VARIANTS
from rewardhack.tasks import TASKSETS
from rewardhack.training_sampling import MODES as TRAINING_SAMPLERS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['controls', 'probe', 'train'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run-config', type=Path)
    ap.add_argument('--lean', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--revision', default='b968826d9c46dd6066d109eabc6255188de91218')
    ap.add_argument('--quant', choices=['none', '8bit', '4bit'], default='none')
    ap.add_argument('--attn-implementation', choices=['sdpa', 'eager'], default='sdpa',
                    help='attention path to load the model with; gpt-oss supports only eager, which holds batch x heads x length^2 per layer')
    ap.add_argument('--mxfp4', choices=['absent', 'dequantize'], default='absent',
                    help='gpt-oss ships 4-bit expert weights: dequantize expands them to bf16, which training needs; absent is for every other model')
    ap.add_argument('--experts-implementation', help='mixture-of-experts layer implementation; unset keeps the loaded default, which the summary records')
    ap.add_argument('--thinking', choices=['on', 'off', 'unsupported'], default='off')
    ap.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'], help='gpt-oss: passed to the chat template instead of --thinking')
    ap.add_argument('--arm', choices=ARMS, default='baseline')
    ap.add_argument('--penalty', type=float, default=0.0, help='λ of the report_correct arm: the share of the cheat payout removed')
    ap.add_argument('--beta', type=float, default=0.0, help='KL coefficient against the reference policy; above 0 TRL also logs kl')
    ap.add_argument('--prompt-variant', choices=sorted(VARIANTS), default='neutral')
    ap.add_argument('--taskset', choices=TASKSETS, default='v1')
    start = ap.add_mutually_exclusive_group()
    start.add_argument('--adapter', type=Path, help='start from these trained weights with a fresh optimizer and step count')
    start.add_argument('--resume-from', type=Path,
                       help="continue a train run from one of its checkpoints (<out>/trainer/checkpoint-N): weights, optimizer, step count, "
                            "random state and task order; the run's settings must match, and --steps is the new total")
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--prompts-per-step', type=int, default=1)
    ap.add_argument('--num-generations', type=int, default=8)
    ap.add_argument('--backward-batch-size', type=int,
                    help='completions per backward microbatch; unset uses num-generations; must divide the effective update batch')
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--lr-schedule', choices=['constant', 'linear'], default='constant', help='linear decays the rate to zero over --steps')
    ap.add_argument('--scale-rewards', choices=['group', 'batch', 'none'], default='none',
                    help="how GRPO scales advantages: none (owner, 26 Nov) keeps the size of a reward gap, such as the penalty; "
                         "group divides by each prompt group's reward spread (TRL's default, used by every run before 26 Nov), which hides it")
    ap.add_argument('--max-grad-norm', type=float, default=1.0, help='gradient clipping threshold')
    ap.add_argument('--router-aux-loss-coef', type=float, default=0.0,
                    help="weight of a mixture-of-experts model's router load-balancing loss; 0 trains on the reward alone")
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--max-completion-length', type=int, default=512)
    ap.add_argument('--eval-samples', type=int, default=4)
    ap.add_argument('--eval-limit', type=int)
    ap.add_argument('--train-limit', type=int)
    ap.add_argument('--save-every', type=int, default=16)
    ap.add_argument('--stop-no-variation', type=int, default=24)
    ap.add_argument('--stop-saturated', action=argparse.BooleanOptionalAction, default=True,
                    help=f'stop once reward variation is credibly below {SATURATION_FLOOR:.0%}% of the latest {SATURATION_WINDOW} prompt groups (outcomes.saturated)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--training-sampler', choices=TRAINING_SAMPLERS, default='uniform',
                    help='family-truth gives equal mass to each training family/truth stratum in short blocks')
    ap.add_argument('--max-seconds', type=int, default=3600)
    ap.add_argument('--final-eval-seconds', type=int, default=600, help='reserved after training for the after-evaluation')
    reporting.add_arguments(ap)
    vllm_rollouts.add_arguments(ap)
    retry_options.add_arguments(ap)
    judge.add_arguments(ap)
    args = ap.parse_args()
    if args.run_config:
        if args.mode != 'train':
            ap.error('--run-config requires train mode')
        from rewardhack.run_registry import training_arguments
        config = json.loads(args.run_config.read_text())
        args, flags = training_arguments(ap, sys.argv[1:], config)
        if 'historical_commit' in config:
            historical = Path('.tools/experiment-source').resolve()
            arguments = list(sys.argv[1:])
            position = arguments.index('--run-config')
            del arguments[position:position + 2]
            env = {**os.environ, 'PYTHONPATH': str(historical / 'src')}
            os.execve(sys.executable, [sys.executable, str(historical / 'entrypoints/run.py'),
                                      *arguments, *flags, '--resume-from', str(args.resume_from)], env)
    if args.resume_from and args.mode != 'train':
        ap.error(f'--resume-from continues training; {args.mode} would ignore it')
    judge.check_reward_args(ap, args)
    vllm_rollouts.check_args(args)
    run(args)


if __name__ == '__main__':
    main()
