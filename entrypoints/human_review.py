#!/usr/bin/env python3
"""A small blind reading by a person: build the sheet, then score the answers typed into it.

`build` writes `sheet.md` (statement and report only) and `key.jsonl` (the read label, the judge's claim
and the verifier's verdict: do not open it before answering). `score` compares the filled sheet with the key.
"""
import argparse
import json
from pathlib import Path

from rewardhack.judge import read_set, review_key, review_sample, review_sheet, score_review

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('build', 'score'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--read', type=Path, default=ROOT/'results'/'report-reading')
    ap.add_argument('--verdicts', type=Path, nargs='+', help='verdicts.jsonl of the judge whose disagreements are to be ruled on')
    ap.add_argument('--random', type=int, default=12)
    ap.add_argument('--false-honest', type=int, default=8)
    ap.add_argument('--false-dishonest', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.command == 'build':
        if not args.verdicts:
            ap.error('build needs --verdicts')
        if (args.out/'sheet.md').exists():
            ap.error(f'{args.out}/sheet.md exists: building again would discard answers')
        verdicts = [json.loads(line) for path in args.verdicts for line in path.read_text().splitlines() if line.strip()]
        items = review_sample(read_set(args.read), verdicts, args.random, args.false_honest, args.false_dishonest, args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/'sheet.md').write_text(review_sheet(items))
        (args.out/'key.jsonl').write_text(''.join(json.dumps(k)+'\n' for k in review_key(items)))
        print(f'{len(items)} items in {args.out}/sheet.md')
        return
    key = [json.loads(line) for line in (args.out/'key.jsonl').read_text().splitlines()]
    result = score_review((args.out/'sheet.md').read_text(), key)
    (args.out/'summary.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
