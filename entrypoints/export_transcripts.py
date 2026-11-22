#!/usr/bin/env python3
"""Export every scored transcript under results/ as one self-contained JSONL with prompts, verdicts and reading labels."""
import argparse
import json
from pathlib import Path

from rewardhack.export_transcripts import export


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--out', type=Path, required=True, help='JSONL to write; a .manifest.json lands beside it')
    args = ap.parse_args()
    manifest = export(args.results, args.out)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'cells'}, indent=1))
    for cell in manifest['cells']:
        print(f"{cell['n']:5d}  {cell['run']}  {cell['provenance']}  {cell['cheat']}/{cell['disclosure']}")


if __name__ == '__main__':
    main()
