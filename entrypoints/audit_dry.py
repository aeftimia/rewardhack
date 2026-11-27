#!/usr/bin/env python3
"""Report DRY candidates without modifying sources or invoking a judge."""
import argparse
import json
from pathlib import Path

from rewardhack.dry_audit import MIN_CLONE_NODES, audit, read_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--revision', help='audit a historical commit instead of the working tree')
    parser.add_argument('--include-tests', action='store_true', help='include independent test expectations as candidates')
    parser.add_argument('--min-clone-nodes', type=int, default=MIN_CLONE_NODES)
    parser.add_argument('--json', action='store_true', help='emit findings and coverage as JSON')
    parser.add_argument('--check', action='store_true', help='exit 1 when candidates exist (default: report only)')
    args = parser.parse_args()
    sources, excluded = read_repository(args.root, revision=args.revision, include_tests=args.include_tests)
    report = audit(sources, min_nodes=args.min_clone_nodes)
    report['excluded_files'] = excluded
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['files']} files; {report['constants']} scalar occurrences; "
              f"{report['functions']} functions above size threshold; {len(report['findings'])} candidate groups")
        print(f"{len(excluded)} excluded/deleted files; {len(report['coverage_limits'])} coverage limitations")
        for item in report['coverage_limits']:
            print(f"  coverage: {item}")
        for item in report['findings']:
            print(f"\n{item['rule']}: {item['reason']}")
            for occurrence in item['occurrences']:
                print(f"  {occurrence['path']}:{occurrence['line']}: {occurrence['role']}")
                if item['rule'] != 'function-clone':
                    print(f"    {occurrence['text']}")
        print('\nCandidates require review; structural similarity does not establish shared policy or safe extraction.')
    return int(args.check and bool(report['findings']))


if __name__ == '__main__':
    raise SystemExit(main())
