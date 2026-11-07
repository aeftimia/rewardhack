#!/usr/bin/env python3
"""Check every reference proof and refutation of a task set with Lean, then write the train/evaluation split."""
import argparse
import json
from pathlib import Path

from rewardhack.tasks import TASKSETS, split, taskset_rows, verify_references


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lean', required=True)
    ap.add_argument('--taskset', choices=TASKSETS, default='v1')
    ap.add_argument('--family', help='restrict to one family')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=Path, help='write the split here when every reference checks')
    args = ap.parse_args()
    rows = [r for r in taskset_rows(args.taskset) if not args.family or r['family'] == args.family]
    failures = verify_references(args.lean, rows, workers=args.workers)
    data = split(rows)
    print(json.dumps({'rows': len(rows), 'train': len(data['train']), 'evaluation': len(data['evaluation']),
                      'by_family': {f: sum(1 for r in rows if r['family'] == f) for f in sorted({r['family'] for r in rows})},
                      'failures': failures}, indent=1, ensure_ascii=False))
    if args.out and not failures:
        args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False)+'\n')
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
