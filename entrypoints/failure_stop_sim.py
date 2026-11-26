#!/usr/bin/env python3
"""Simulate the run-time failure stop (`outcomes.judge_failing`, `verifier_failing`) at several true failure rates.

A run of `--steps` batches of `--batch` readings, each failing independently at the true rate, is checked after every
batch over the whole run and over the latest FAILURE_WINDOW readings, as training does. Prints, per true rate, the share
of runs stopped and the median failures and readings at the stop. Numbers for the defaults: docs/notes-2026-11-21.md (27 Nov).
"""
import argparse

import numpy as np

from rewardhack.outcomes import FAILURE_WINDOW, credibly_failing


def stops(rate, runs, steps, batch, seed):
    """Per run: whether it stopped, and the failures and readings counted over the whole run at its first stop."""
    rng = np.random.default_rng(seed)
    flags = rng.random((runs, steps*batch)) < rate
    total = np.concatenate([np.zeros((runs, 1), int), flags.cumsum(axis=1)], axis=1)
    ends = np.arange(1, steps+1)*batch
    starts = np.maximum(ends-FAILURE_WINDOW, 0)
    whole = credibly_failing(total[:, ends], np.broadcast_to(ends, (runs, steps)))
    window = credibly_failing(total[:, ends]-total[:, starts], np.broadcast_to(ends-starts, (runs, steps)))
    stop = whole | window
    stopped, first = stop.any(axis=1), stop.argmax(axis=1)
    return stopped, total[np.arange(runs), ends[first]][stopped], ends[first][stopped]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rates', type=float, nargs='+', default=[0.02, 0.05, 0.10, 0.20, 0.50])
    ap.add_argument('--runs', type=int, default=20_000)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    for rate in args.rates:
        stopped, failures, readings = stops(rate, args.runs, args.steps, args.batch, args.seed)
        at = f'median {np.median(failures):g} failures in {np.median(readings):g} readings' if stopped.any() else 'never stopped'
        print(f'true rate {rate:.2f}: {stopped.mean():.4f} of {args.runs} runs stopped; {at}')


if __name__ == '__main__':
    main()
