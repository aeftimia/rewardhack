#!/usr/bin/env python3
"""Spending ledger status and the pre-launch budget guard. Exit 1 when a launch would breach a cap."""
import argparse
import sys
from pathlib import Path

from rewardhack.budget import MODEL_8B_CAP, TOTAL_CAP, check, read_ledger, spent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ledger', type=Path, default=Path(__file__).resolve().parent.parent/'spending'/'ledger.csv')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    guard = sub.add_parser('check')
    guard.add_argument('--provider', required=True, choices=['aws', 'runpod', 'other'])
    guard.add_argument('--model', required=True)
    guard.add_argument('--hourly', type=float, required=True)
    guard.add_argument('--max-hours', type=float, required=True)
    guard.add_argument('--other-usd', type=float, default=0.0)
    args = ap.parse_args()
    rows = read_ledger(args.ledger)
    if args.cmd == 'status':
        print(f'total ${spent(rows):.2f} of ${TOTAL_CAP}')
        print(f"8B ${spent(rows, model='8B'):.2f} of ${MODEL_8B_CAP}")
        for key in ('provider', 'model'):
            for value in sorted({r[key] for r in rows}):
                print(f'  {key}={value}: ${spent(rows, **{key: value}):.2f}')
        return 0
    worst, problems = check(rows, args.provider, args.model, args.hourly, args.max_hours, args.other_usd)
    print(f'worst case ${worst:.2f}; remaining after ${TOTAL_CAP-spent(rows)-worst:.2f}')
    for problem in problems:
        print('REFUSED:', problem)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
