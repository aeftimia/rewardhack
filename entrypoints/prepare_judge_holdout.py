#!/usr/bin/env python3
"""Freeze a judge protocol, then prepare a blind sheet from fresh reports. No paid calls."""
import argparse
import json
import subprocess
from pathlib import Path

from rewardhack.judge import add_arguments
from rewardhack.judge_holdout import freeze, judge_config, prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('freeze', 'prepare'))
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--read', type=Path, default=Path(__file__).resolve().parent.parent/'results'/'report-reading')
    ap.add_argument('--lean')
    ap.add_argument('--rows', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--seed', type=int, default=20261124)
    ap.add_argument('--descriptive', action='store_true',
                    help='prepare: a small human audit (at most 24 reports, read with validate_judge.py --audit), not an acceptance holdout')
    add_arguments(ap)
    args = ap.parse_args()
    if args.command == 'freeze':
        if not args.lean or args.judge == 'none' or args.judge == 'triage':
            ap.error('freeze requires --lean and an actual judge configuration')
        version = subprocess.check_output([args.lean, '--version'], text=True).strip()
        try:
            protocol = freeze(args.read, judge_config(args), version)
        except ValueError as error:
            ap.error(str(error))
        args.protocol.parent.mkdir(parents=True, exist_ok=True)
        with args.protocol.open('x') as sink:
            sink.write(json.dumps(protocol, indent=2)+'\n')
        print(f'Frozen protocol: {args.protocol}; all existing read reports are development data')
    else:
        if not args.rows or not args.out:
            ap.error('prepare requires --rows and --out')
        rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
        try:
            prepare(rows, json.loads(args.protocol.read_text()), args.out, args.seed, descriptive=args.descriptive)
        except ValueError as error:
            ap.error(str(error))
        print(f'Blind review sheet: {args.out}/sheet.md')


if __name__ == '__main__':
    main()
