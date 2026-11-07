#!/usr/bin/env python3
"""Rescore saved rows from raw model output with the sandboxed verifier. The only scoring path for rows from containers and APIs."""
import argparse
import json
from pathlib import Path

from rewardhack.rescore import rescore_file
from rewardhack.tasks import task_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('rows', type=Path, nargs='+')
    ap.add_argument('--lean', required=True)
    args = ap.parse_args()
    tasks = task_index()
    for path in args.rows:
        print(json.dumps(rescore_file(path, args.lean, tasks), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
