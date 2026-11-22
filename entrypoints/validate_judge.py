#!/usr/bin/env python3
"""Compare a report judge with the hand-read labels before it is allowed into a reward.

Prints claim agreement, the honesty bit's two error rates (false_honest is the costly one: a hedge or a
false claim paid as an honest report) and the confusion table; writes every verdict beside the summary.
"""
import argparse
import json
import time
from pathlib import Path

from rewardhack.judge import add_arguments, from_args, read_set, sample_set, validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--read', type=Path, default=Path(__file__).resolve().parent.parent/'results'/'report-reading')
    ap.add_argument('--per-category', type=int, help='stratified sample: this many rows of each reading category; default is every read row')
    ap.add_argument('--seed', type=int, default=0)
    add_arguments(ap)
    args = ap.parse_args()
    if args.judge == 'none':
        ap.error('choose a judge to validate')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_set(args.read)
    if args.per_category:
        rows = sample_set(rows, args.per_category, args.seed)
    judge = from_args(args)
    started = time.perf_counter()
    with (args.out/'verdicts.jsonl').open('w') as sink:
        result = validate(judge, rows, sink)
    result.update(judge=args.judge, model=args.judge_model, revision=args.judge_revision, per_category=args.per_category, seed=args.seed,
                  seconds=time.perf_counter()-started, seconds_per_report=(time.perf_counter()-started)/max(len(rows), 1),
                  usage=judge.meter.totals if judge.meter else None)
    (args.out/'summary.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
