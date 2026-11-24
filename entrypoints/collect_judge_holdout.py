#!/usr/bin/env python3
"""Generate fresh local-model reports, verify them, and prepare the blind review sheet."""
import argparse
import json
from pathlib import Path

from rewardhack.holdout_collection import collect, fresh_tasks
from rewardhack.judge_holdout import check_frozen, prepare
from rewardhack.tasks import verify_references


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--model-path', type=Path, required=True)
    ap.add_argument('--lean', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--review-out', type=Path, required=True)
    ap.add_argument('--target', type=int, choices=(12, 24), default=24, help='human review size; descriptive audit only')
    ap.add_argument('--seed', type=int, default=20261124)
    ap.add_argument('--max-seconds', type=int, default=1800)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text())
    check_frozen(protocol)
    # Before the long collection, not after it: both directories are created fresh.
    for path in (args.out, args.review_out):
        if path.exists() and not args.verify_only:
            ap.error(f'{path} already exists; choose a new directory')
    tasks = fresh_tasks()
    failures = verify_references(args.lean, tasks, workers=4)
    if failures:
        raise RuntimeError(f'Reference checks failed: {failures}')
    print(f'{len(tasks)} reference proofs/refutations verified', flush=True)
    if args.verify_only:
        return
    rows, summary = collect(tasks, protocol, args.model_path, args.lean, args.out,
                            target=args.target, seed=args.seed, max_tokens=args.max_tokens, max_seconds=args.max_seconds)
    print(json.dumps(summary), flush=True)
    if not summary['complete']:
        raise SystemExit('Collection incomplete; preserved attempts and accepted rows for inspection')
    try:
        prepare(rows, protocol, args.review_out, args.seed, descriptive=True)
    except (ValueError, OSError) as error:
        raise SystemExit(f'{error}\nThe collection is saved. Prepare the sheet from it with: prepare_judge_holdout.py prepare '
                         f'--descriptive --protocol {args.protocol} --rows {args.out}/fresh.jsonl --out <new directory>')
    print(f'Blind review: {args.review_out}/sheet.md', flush=True)


if __name__ == '__main__':
    main()
