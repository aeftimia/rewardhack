#!/usr/bin/env python3
"""Plan a bounded compute pilot; --execute explicitly opts into its staged commands."""
import argparse
import json
from pathlib import Path

from rewardhack.pilot_runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--spec', type=Path)
    source.add_argument('--example', type=Path, metavar='DIR', help='create an offline synthetic example in a new directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='execute frozen commands; otherwise only print the plan')
    args = parser.parse_args()
    try:
        if args.example is not None:
            from rewardhack.pilot_simulation import example_spec
            args.spec = example_spec(args.example)
        result = run(args.spec, args.out, execute=args.execute)
    except (ValueError, OSError) as error:
        parser.exit(1, f'REFUSED: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    if not args.execute:
        return 0
    return int(result['reason'] in ('stage_failure', 'measurement_failure'))


if __name__ == '__main__':
    raise SystemExit(main())
