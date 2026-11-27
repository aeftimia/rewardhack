#!/usr/bin/env bash
# Prompt-permission grid on the hard task set with a follow-on turn, through the OpenAI API.
# Stage A: gpt-5.4-nano (high reasoning, all tasks) and gpt-5.5 (no reasoning, false tasks) x {neutral, nocheat, maycheat}.
# Every run has its own metered cap, at the model's price in model_io.PRICES; the stage stops when the running total passes STAGE_CAP.
# Usage: bash ops/api_grid.sh <stage>   (A)
set -euo pipefail
cd "$(dirname "$0")/.." || { echo "cannot enter the repository root from $0" >&2; exit 1; }
: "${OPENAI_API_KEY:?export OPENAI_API_KEY before running the API grid}"
STAGE=${1:-A}; OUT=results/api-grid-$STAGE; mkdir -p "$OUT"
STAGE_CAP=${STAGE_CAP:-8.0}
total=0
run() { name=$1; shift
  if [[ -f "$OUT/$name/summary.json" ]]; then echo "skip $name (done)"; return; fi
  .tools/venv/bin/python entrypoints/probe_api.py --out "$OUT/$name" --taskset hard --follow-on "$@" > "$OUT/$name.log" 2>&1 \
    || { echo "$name failed; see $OUT/$name.log and reconcile its spend before rerunning" >&2; exit 1; }
  # A run that called the API and crashed before writing its summary has spent an unknown amount. That is not $0 toward STAGE_CAP: the stage stops.
  [[ -f "$OUT/$name/summary.json" ]] || { echo "$name wrote no summary.json (see $OUT/$name.log), so what it spent is unknown and the stage stops here. Read the spend from the API usage page before rerunning."; exit 1; }
  u=$(python3 -c "import json; print(float(json.load(open('$OUT/$name/summary.json'))['usd']))") || { echo "$name: no readable usd in $OUT/$name/summary.json, so what it spent is unknown and the stage stops here."; exit 1; }
  total=$(python3 -c "print(round($total+$u,4))"); echo "$name usd=$u total=$total"
  python3 -c "import sys; sys.exit(0 if $total <= $STAGE_CAP else 1)" || { echo "STAGE CAP $STAGE_CAP reached"; exit 0; }; }
case $STAGE in
A)
  for v in neutral nocheat maycheat; do
    run nano-high-$v --tasks all --variant $v --model gpt-5.4-nano --reasoning-effort high --max-tokens 12000 --samples 1 --max-usd 2.0
  done
  for v in neutral nocheat maycheat; do
    run gpt55-none-$v --tasks false --variant $v --model gpt-5.5 --reasoning-effort none --max-tokens 1024 --samples 1 --max-usd 1.2
  done;;
B)
  for v in neutral nocheat maycheat; do
    run mini-high-$v --tasks false --variant $v --model gpt-5.4-mini --reasoning-effort high --max-tokens 12000 --samples 1 --max-usd 5.0
  done;;
C)
  run gpt55-high-neutral --tasks false --variant neutral --model gpt-5.5 --reasoning-effort high --max-tokens 12000 --samples 1 --max-usd 12.0;;
*) echo "unknown stage: $STAGE; expected A, B, or C" >&2; exit 2;;
esac
echo "STAGE $STAGE DONE total=$total"
