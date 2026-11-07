#!/usr/bin/env python3
"""Free local measurements on Apple silicon with a 4-bit MLX model: `probe` or `rl`."""
import argparse
from pathlib import Path

from rewardhack.mlx_rl import run
from rewardhack.prompts import VARIANTS
from rewardhack.tasks import TASKSETS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['probe', 'rl'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lean', required=True)
    ap.add_argument('--model', default='mlx-community/Qwen3-8B-4bit')
    ap.add_argument('--revision', default='545dc4251c05440727734bcd94334791f6ab0192')
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='neutral')
    ap.add_argument('--taskset', choices=TASKSETS, default='v1')
    ap.add_argument('--tasks', choices=['all', 'true', 'false'], default='all')
    ap.add_argument('--thinking', action='store_true')
    ap.add_argument('--arm', choices=['baseline', 'equal_reward'], default='baseline')
    ap.add_argument('--samples', type=int, default=4)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--max-seconds', type=int, default=3600)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--lora-rank', type=int, default=8)
    ap.add_argument('--lora-layers', type=int, default=8)
    ap.add_argument('--lora-scale', type=float, default=20.0)
    ap.add_argument('--adapter', type=Path, help='adapter.safetensors from an earlier rl run to branch from')
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--monitor-every', type=int, default=10)
    ap.add_argument('--monitor-samples', type=int, default=2)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
