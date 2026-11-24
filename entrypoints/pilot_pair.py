#!/usr/bin/env python3
"""Write one paired seed/checkpoint observation; a stage combines its planned seed files."""
import argparse
import json
from pathlib import Path

from rewardhack.pilot import load_spec, read_json
from rewardhack.pilot_evidence import paired_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'protocol', 'seed', 'baseline', 'intervention', 'out'):
        parser.add_argument('--'+name, required=True)
    for name in ('checkpoint', 'baseline-compute', 'intervention-compute'):
        parser.add_argument('--'+name, required=True, type=float)
    args = parser.parse_args()
    row = paired_observation(load_spec(args.spec), read_json(args.protocol), args.seed, args.checkpoint,
                             args.baseline, args.intervention, args.baseline_compute, args.intervention_compute)
    # One observation per file; a stage producer combines its planned seed files.
    with Path(args.out).open('x') as handle:
        handle.write(json.dumps(row)+'\n')


if __name__ == '__main__':
    main()
