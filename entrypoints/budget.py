#!/usr/bin/env python3
"""Read budget status/checks, or atomically claim a pilot launch's reserved allowance."""
import argparse
import os
import sys
from pathlib import Path

from rewardhack.budget import LINES, RESERVATION_ENV, TOTAL_CAP, check, claim, grant_spent, number_text, read_ledger, spent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ledger', type=Path, default=Path(__file__).resolve().parent.parent/'spending'/'ledger.csv')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    for name in ('check', 'claim'):
        guard = sub.add_parser(name, help='read-only grant check' if name == 'check' else 'consume pilot allowance before one launch')
        guard.add_argument('--line', required=True, choices=sorted(LINES), help='grant budget line this launch is charged to')
        # Prices are read as exact decimals; their range is checked by the budget, which refuses with the reason.
        guard.add_argument('--hourly', type=number_text, required=True)
        guard.add_argument('--max-hours', type=number_text, required=True)
        guard.add_argument('--other-usd', type=number_text, default='0')
        guard.add_argument('--launch-cap', type=number_text, help='ceiling approved for this one launch')
    args = ap.parse_args()
    try:
        if args.cmd == 'claim':
            result = claim(args.ledger, os.environ.get(RESERVATION_ENV), args.line,
                           args.hourly, args.max_hours, args.other_usd, args.launch_cap)
            print(f"worst case ${result['worst_usd']:.2f}; stage allowance remaining ${result['remaining_usd']:.2f}")
            problems = result['problems']
        else:
            rows = read_ledger(args.ledger)
            if args.cmd == 'status':
                print(f'grant charged/reserved ${grant_spent(rows):.2f} of ${TOTAL_CAP:.2f}')
                for line, cap in LINES.items():
                    print(f'  {line}: charged/reserved ${spent(rows, line=line):.2f} of ${cap:.2f}')
                outside = spent(rows)-grant_spent(rows)
                print(f"outside the grant (pilot and other funds): ${0.0 if abs(outside) < 0.005 else outside:.2f}")
                return 0
            worst, problems = check(rows, args.line, args.hourly, args.max_hours, args.other_usd, args.launch_cap)
            print(f'worst case ${worst:.2f}; {args.line} remaining after ${LINES[args.line]-spent(rows, line=args.line)-worst:.2f}')
            if os.environ.get(RESERVATION_ENV):
                problems.append('pilot launches require an explicit claim; check is read-only and cannot consume the stage reservation')
        for problem in problems:
            print('REFUSED:', problem)
        return 1 if problems else 0
    except (ValueError, OSError) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
