"""Replay saved Qwen termination evidence; no model generation or paid API calls.

The older AWQ probe is a sizing reference, not a prediction on the hard bf16 run.
Shorter caps count only whole recorded completions that fit: prefixes of thinking
text must never become submissions. Longer caps cannot uncensor saved answers.
"""
import argparse
import json
from pathlib import Path

from rewardhack import jsonl
from rewardhack.grader import parse_completion
from rewardhack.hf_grpo import check_completion_limit
from rewardhack.outcomes import token_limit_advice


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--measurement', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--caps', type=int, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = list(jsonl.read(args.measurement/'before-evaluation.jsonl'))
    reference = list(jsonl.read(args.reference/'rows.jsonl'))
    settings = json.loads((args.reference/'summary.json').read_text())
    if not rows or not reference:
        raise ValueError('both evidence sets must contain rows')
    if any(cap <= 0 or cap > settings['max_tokens'] for cap in args.caps):
        raise ValueError('caps must be positive and no larger than the recorded reference cap')
    advice = token_limit_advice(rows)
    try:
        check_completion_limit(advice)
    except ValueError as error:
        guard = dict(refused=True, reason=str(error))
    else:
        guard = dict(refused=False)
    comparisons = []
    for cap in args.caps:
        fits = [r for r in reference if not r['truncated'] and r['tokens'] <= cap]
        comparisons.append(dict(cap=cap, samples=len(reference), complete=len(fits),
            cut_off=len(reference)-len(fits),
            parseable_complete=sum(parse_completion(r['raw'])['report'] is not None for r in fits),
            sampled_tokens=sum(min(cap, r['tokens']) for r in reference),
            advice=token_limit_advice([dict(truncated=r['truncated'] or r['tokens'] > cap) for r in reference])))
    report = dict(measurement=str(args.measurement), reference=str(args.reference),
        interpretation='Offline whole-completion replay, not fresh generations; AWQ reference uses different tasks.',
        measurement_samples=len(rows), measurement_advice=advice, guard=guard,
        truncated_before_thinking_closed=sum(r['truncated'] and '</think>' not in r['raw'] for r in rows),
        measurement_parseable=sum(r['format_valid'] for r in rows),
        reference_generation_seconds=settings['generation_seconds'], caps=comparisons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
