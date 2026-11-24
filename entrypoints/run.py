#!/usr/bin/env python3
"""GPU path (TRL GRPO): `controls`, `probe` or `train` on a rented GPU, recording cost-relevant timings.

`--prompt-variant hint` is a labelled mechanism check, never pooled with the main arms.
"""
import argparse
from pathlib import Path

from rewardhack import judge, retry_options
from rewardhack.hf_grpo import run
from rewardhack.outcomes import ARMS
from rewardhack.prompts import VARIANTS
from rewardhack.tasks import TASKSETS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['controls', 'probe', 'train'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lean', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--revision', default='b968826d9c46dd6066d109eabc6255188de91218')
    ap.add_argument('--quant', choices=['none', '8bit', '4bit'], default='none')
    ap.add_argument('--thinking', choices=['on', 'off', 'unsupported'], default='off')
    ap.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'], help='gpt-oss: passed to the chat template instead of --thinking')
    ap.add_argument('--arm', choices=ARMS, default='baseline')
    ap.add_argument('--penalty', type=float, default=0.0, help='λ of the report_correct arm: the share of the cheat payout removed')
    ap.add_argument('--beta', type=float, default=0.0, help='KL coefficient against the reference policy; above 0 TRL also logs kl')
    ap.add_argument('--prompt-variant', choices=sorted(VARIANTS), default='neutral')
    ap.add_argument('--taskset', choices=TASKSETS, default='v1')
    ap.add_argument('--adapter', type=Path)
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--prompts-per-step', type=int, default=1)
    ap.add_argument('--num-generations', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--max-completion-length', type=int, default=512)
    ap.add_argument('--eval-samples', type=int, default=4)
    ap.add_argument('--eval-limit', type=int)
    ap.add_argument('--train-limit', type=int)
    ap.add_argument('--save-every', type=int, default=16)
    ap.add_argument('--stop-no-variation', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-seconds', type=int, default=3600)
    ap.add_argument('--final-eval-seconds', type=int, default=600, help='reserved after training for the after-evaluation')
    retry_options.add_arguments(ap)
    judge.add_arguments(ap)
    args = ap.parse_args()
    judge.check_reward_args(ap, args)
    run(args)


if __name__ == '__main__':
    main()
