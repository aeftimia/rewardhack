#!/usr/bin/env python3
"""Automatically check formal fields in a fresh report collection; do not interpret prose."""
import argparse
import json
from pathlib import Path

from rewardhack.formal_audit import audit_file
from rewardhack.holdout_collection import fresh_tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lean', required=True)
    args = ap.parse_args()
    try:
        print(json.dumps(audit_file(args.lean, args.rows, args.out, {t['id']: t for t in fresh_tasks()}), indent=2))
    except ValueError as error:
        ap.error(str(error))


if __name__ == '__main__':
    main()
