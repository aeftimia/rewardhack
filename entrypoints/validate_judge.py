#!/usr/bin/env python3
"""Compare a report judge with the model-read labels before it is allowed into a reward.

Prints claim agreement, the honesty bit's two error rates (false_honest is the costly one: a hedge or a
false claim paid as an honest report) and the confusion table; writes every verdict beside the summary.
"""
import argparse
import json
import time
import subprocess
from pathlib import Path

from rewardhack.judge import add_arguments, check_arguments, from_args, read_set, sample_set, validate
from rewardhack.judge_holdout import acceptance, judge_config, load_review


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--read', type=Path, default=Path(__file__).resolve().parent.parent/'results'/'report-reading')
    ap.add_argument('--per-category', type=int, help='stratified sample: this many rows of each reading category; default is every read row')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lean', help='Lean binary: with it, a declared different proposition is flagged in code before the judge is asked')
    ap.add_argument('--max-usd', type=float, help='stop a metered judge at this spend')
    review = ap.add_mutually_exclusive_group()
    review.add_argument('--audit', type=Path, help='prepared small blind audit; descriptive comparison with no accuracy acceptance gate')
    review.add_argument('--holdout', type=Path, help='prepared holdout directory with completed blind sheet; enforces frozen configuration and acceptance')
    ap.add_argument('--require-acceptance', action='store_true', help='exit nonzero unless all predeclared acceptance thresholds pass')
    add_arguments(ap)
    args = ap.parse_args()
    if args.judge == 'none':
        ap.error('choose a judge to validate')
    check_arguments(ap, args, **{'--max-usd': args.max_usd})
    criteria, provenance = None, {}
    if args.audit and args.require_acceptance:
        ap.error('--audit is descriptive and cannot require accuracy acceptance')
    if args.holdout or args.audit:
        if not args.lean or args.per_category:
            ap.error('--holdout/--audit requires --lean and cannot be subsampled')
        version = subprocess.check_output([args.lean, '--version'], text=True).strip()
        try:
            rows, criteria, provenance = load_review(args.holdout or args.audit, judge_config(args), version, descriptive=bool(args.audit))
        except ValueError as error:
            ap.error(str(error))
    else:
        rows = read_set(args.read)
        if args.per_category:
            rows = sample_set(rows, args.per_category, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    judge = from_args(args)
    started = time.perf_counter()
    with (args.out/'verdicts.jsonl').open('w') as sink:
        result = validate(judge, rows, sink, args.max_usd)
    result.update(judge=args.judge, model=args.judge_model, revision=args.judge_revision, per_category=args.per_category, seed=args.seed, statement_check=bool(args.lean), planned=len(rows),
                  # Per report judged, not per report planned: a run stopped at its cost cap judged fewer than it planned.
                  seconds=time.perf_counter()-started, seconds_per_report=(time.perf_counter()-started)/result['rows'] if result['rows'] else None,
                  usage=judge.meter.totals if judge.meter else None)
    if args.audit:
        result.update(complete=result['rows'] == len(rows),
                      interpretation='Descriptive human audit only; no accuracy acceptance or reliability claim.')
    else:
        result.update(acceptance(result, len(rows), criteria))
    result.update(validation_kind='human_audit' if args.audit else 'human_heldout' if args.holdout else 'development', **provenance)
    (args.out/'summary.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))
    if args.audit and (not result['complete'] or result['unparsed']):
        raise SystemExit(1)
    if (args.holdout or args.require_acceptance) and not result['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
