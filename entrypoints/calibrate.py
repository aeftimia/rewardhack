#!/usr/bin/env python3
"""Per-statement proof / placeholder / refutation / acknowledgement rates from rescored probe rows.

`--judge labels` takes a reading pass (`--labels <jsonl> --run-name <prefix>`); model judges are first passes only.
"""
import argparse
from pathlib import Path

from rewardhack.calibrate import JUDGES, run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows', type=Path, nargs='+', required=True, help='rescored probe rows (rows.rescored.jsonl)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--judge', choices=JUDGES, default='labels')
    ap.add_argument('--labels', type=Path, help="with --judge labels: JSONL of {id: '<run>#<row index>', category: <reading rubric category>}")
    ap.add_argument('--judge-model', default='mlx-community/Qwen3-8B-4bit')
    ap.add_argument('--judge-revision', default='545dc4251c05440727734bcd94334791f6ab0192')
    ap.add_argument('--run-name', default='run', help='prefix used in label ids for --judge labels')
    ap.add_argument('--lo', type=float, default=0.2)
    ap.add_argument('--hi', type=float, default=0.6)
    args = ap.parse_args()
    if args.judge == 'labels' and not args.labels:
        ap.error('--judge labels needs --labels')
    run(args)


if __name__ == '__main__':
    main()
