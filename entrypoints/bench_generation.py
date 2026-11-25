#!/usr/bin/env python3
"""Time generation, the largest part of a training step: `hf` for the training sampler, `server` for a vLLM server."""
import argparse
from pathlib import Path

from rewardhack.gen_bench import run_hf, run_server
from rewardhack.tasks import TASKSETS


def numbers(text):
    return [int(part) for part in text.split(',') if part]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['hf', 'server'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--taskset', choices=TASKSETS, default='hard')
    ap.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'])
    ap.add_argument('--thinking', choices=['on', 'off', 'unsupported'], default='unsupported')
    # hf
    ap.add_argument('--revision')
    ap.add_argument('--mxfp4', choices=['absent', 'dequantize'], default='absent')
    ap.add_argument('--attn-implementation', choices=['sdpa', 'eager'], default='sdpa')
    ap.add_argument('--adapters', type=Path, help='hf: folder to save the untrained adapters in')
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--batches', type=numbers, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--lora-batches', type=numbers, default=[4, 8, 32])
    ap.add_argument('--focus-batch', type=int, default=8, help='batch size for the comparisons made at one size')
    ap.add_argument('--experts', type=lambda text: [part for part in text.split(',') if part], default=[],
                    help='expert-layer implementations to compare with the default, e.g. eager')
    ap.add_argument('--experts-implementation', help='expert-layer implementation for the whole run instead of the loaded default')
    ap.add_argument('--new-tokens', type=int, default=96)
    ap.add_argument('--long-tokens', type=int, default=384)
    ap.add_argument('--profile-tokens', type=int, default=12)
    # server
    ap.add_argument('--url', default='http://127.0.0.1:8001/v1')
    ap.add_argument('--adapter', action='append', default=[], help='server: name=path of an adapter to load, repeatable')
    ap.add_argument('--concurrency', type=numbers, default=[4, 8, 16, 32])
    ap.add_argument('--base-concurrency', type=numbers, default=[4, 32])
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--template-kwargs', default='{}')
    ap.add_argument('--request-seconds', type=int, default=300)
    args = ap.parse_args()
    if args.mode == 'hf':
        if args.adapters is None:
            ap.error('hf needs --adapters')
        run_hf(args)
    else:
        run_server(args)


if __name__ == '__main__':
    main()
