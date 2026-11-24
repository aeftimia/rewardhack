#!/usr/bin/env python3
"""Write one free synthetic checkpoint for testing the bounded pilot runner."""
import argparse
from rewardhack.pilot_simulation import SCENARIOS, write_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--checkpoint', type=float, required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, default='delayed')
    args = parser.parse_args()
    write_stage(args.spec, args.checkpoint, args.scenario)


if __name__ == '__main__':
    main()
