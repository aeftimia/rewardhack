#!/usr/bin/env python3
"""Exploration probe of a hosted model through an OpenAI-compatible chat API. Raw rows only; rescore later.

Every run needs a metered dollar cap (`--max-usd`) and the model's prices, so the ledger entry is measured.
"""
import argparse
from pathlib import Path

from rewardhack.openai_probe import run
from rewardhack.prompts import VARIANTS
from rewardhack.tasks import TASKSETS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--base-url', default='https://api.openai.com/v1')
    ap.add_argument('--key-env', default='OPENAI_API_KEY')
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='neutral')
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=4096, help='completion cap, reasoning tokens included where the API counts them')
    ap.add_argument('--reasoning-effort', choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh'])
    ap.add_argument('--temperature', type=float, help='omit for models that reject sampling parameters')
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--limit', type=int, help='tasks to use (for a smoke test)')
    ap.add_argument('--tasks', choices=['all', 'true', 'false'], default='all', help='restrict to true or false statements')
    ap.add_argument('--taskset', choices=TASKSETS, default='v1', help='v1: the original trivial-falsehood set; hard: tasks_hard.py')
    ap.add_argument('--follow-on', action='store_true', help='after each answer, ask a second (false) task in the same conversation WITHOUT the evaluator source; tests whether the grader memory persists')
    ap.add_argument('--max-usd', type=float, required=True, help='stop issuing requests once the running metered total exceeds this')
    ap.add_argument('--usd-per-m-input', type=float, required=True)
    ap.add_argument('--usd-per-m-output', type=float, required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
