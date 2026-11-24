#!/usr/bin/env python3
"""Spending ledger status and the pre-launch budget guard. Exit 1 when a launch would breach a cap."""
import argparse
import sys
from pathlib import Path

from rewardhack.budget import LINES, TOTAL_CAP, check, grant_spent, read_ledger, spent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ledger', type=Path, default=Path(__file__).resolve().parent.parent/'spending'/'ledger.csv')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    guard = sub.add_parser('check')
    guard.add_argument('--line', required=True, choices=sorted(LINES), help='grant budget line this launch is charged to')
    guard.add_argument('--hourly', type=float, required=True)
    guard.add_argument('--max-hours', type=float, required=True)
    guard.add_argument('--other-usd', type=float, default=0.0)
    guard.add_argument('--launch-cap', type=float, help='ceiling approved for this one launch')
    args = ap.parse_args()
    try:
        rows = read_ledger(args.ledger)
        if args.cmd == 'status':
            print(f'grant ${grant_spent(rows):.2f} of ${TOTAL_CAP:.2f}')
            for line, cap in LINES.items():
                print(f'  {line}: ${spent(rows, line=line):.2f} of ${cap:.2f}')
            outside = spent(rows)-grant_spent(rows)  # a real negative is a refund booked to no line: show it
            print(f"outside the grant (pilot and other funds): ${0.0 if abs(outside) < 0.005 else outside:.2f}")
            return 0
        worst, problems = check(rows, args.line, args.hourly, args.max_hours, args.other_usd, args.launch_cap)
        print(f'worst case ${worst:.2f}; {args.line} remaining after ${LINES[args.line]-spent(rows, line=args.line)-worst:.2f}')
        for problem in problems:
            print('REFUSED:', problem)
        return 1 if problems else 0
    except ValueError as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
