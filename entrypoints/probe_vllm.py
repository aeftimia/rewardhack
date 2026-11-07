#!/usr/bin/env python3
"""Exploration probe of a large model with vLLM. `--lean none` keeps model-written code off the host; rescore later."""
import argparse
from pathlib import Path

from rewardhack.prompts import VARIANTS
from rewardhack.tasks import TASKSETS
from rewardhack.vllm_probe import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lean', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='neutral')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-seconds', type=int, default=1800)
    ap.add_argument('--taskset', choices=TASKSETS, default='v1')
    ap.add_argument('--tasks', choices=['all', 'true', 'false'], default='all')
    ap.add_argument('--template-kwargs', default='{}', help='JSON passed to the chat template, e.g. {"enable_thinking": true} or {"reasoning_effort": "high"}')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
