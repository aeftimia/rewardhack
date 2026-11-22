#!/usr/bin/env bash
# CPU end-to-end check with a tiny random model: baseline training, an equal-reward branch, and the report_correct arm.
# Tests plumbing only. The model emits noise, so no behavioural conclusion is possible.
set -euo pipefail
cd "$(dirname "$0")/.."
LEAN="$PWD/.tools/lean-4.19.0-darwin_aarch64/bin/lean"
PY=.tools/venv/bin/python
OUT=.tools/dry
rm -rf "$OUT"
COMMON=(--lean "$LEAN" --model hf-internal-testing/tiny-random-LlamaForCausalLM
        --revision 9fb191250dd56d0ba7ec9785a025ed29c03d5998 --thinking unsupported
        --max-completion-length 24 --eval-limit 2 --eval-samples 2 --train-limit 4
        --num-generations 2 --save-every 2 --stop-no-variation 99 --max-seconds 500 --final-eval-seconds 60)
"$PY" entrypoints/run.py train --out "$OUT/baseline" "${COMMON[@]}" --steps 3
"$PY" entrypoints/run.py train --out "$OUT/branch" "${COMMON[@]}" --steps 2 --arm equal_reward --adapter "$OUT/baseline/adapter"
# The funded arm: report correctness with a cheat penalty. `triage` is the keyword reader, for plumbing only.
"$PY" entrypoints/run.py train --out "$OUT/report" "${COMMON[@]}" --steps 2 --arm report_correct --penalty 0.5 --judge triage
"$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
for run in ('baseline', 'branch', 'report'):
    s = json.loads((out/run/'summary.json').read_text())
    assert s['status'] == 'completed', s
    assert s['global_step'] >= 2 and s['rollouts']['n'] > 0, s
    assert (out/run/'adapter'/'adapter_model.safetensors').exists()
    assert (out/run/'before-evaluation.jsonl').exists() and (out/run/'after-evaluation.jsonl').exists()
    print(run, {k: s.get(k) for k in ('global_step', 'trainable_parameters', 'adapter_max_abs_change', 'mean_seconds', 'rollouts')})
assert 'initial_adapter_sha256' in json.loads((out/'branch'/'summary.json').read_text())
steps = [json.loads(line) for line in (out/'report'/'steps.jsonl').read_text().splitlines()]
rows = [json.loads(line) for line in (out/'report'/'training-rollouts.jsonl').read_text().splitlines()]
assert len(steps) >= 2 and all(s['groups'] == s['mixed_groups'] + s['zero_advantage_groups'] for s in steps), steps
assert all(r['arm'] == 'report_correct' and r['penalty'] == 0.5 and r['claim'] is not None for r in rows)
print('dry run passed')
PY
