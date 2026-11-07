#!/usr/bin/env bash
# Prompt-permission grid on the hard task set with a follow-on turn, through the OpenAI API.
# Stage A: gpt-5.4-nano (high reasoning, all tasks) and gpt-5.5 (no reasoning, false tasks) x {neutral, nocheat, maycheat}.
# Every run has its own metered cap; the stage stops when the running total passes STAGE_CAP.
# Usage: bash ops/api_grid.sh <stage>   (A)
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/.zshrc 2>/dev/null
STAGE=${1:-A}; OUT=results/api-grid-$STAGE; mkdir -p "$OUT"
STAGE_CAP=${STAGE_CAP:-8.0}
total=0
run() { name=$1; shift
  if [[ -f "$OUT/$name/summary.json" ]]; then echo "skip $name (done)"; return; fi
  .tools/venv/bin/python entrypoints/probe_api.py --out "$OUT/$name" --taskset hard --follow-on "$@" > "$OUT/$name.log" 2>&1
  u=$(python3 -c "import json; print(json.load(open('$OUT/$name/summary.json'))['usd'])" 2>/dev/null || echo 0)
  total=$(python3 -c "print(round($total+$u,4))"); echo "$name usd=$u total=$total"
  python3 -c "import sys; sys.exit(0 if $total <= $STAGE_CAP else 1)" || { echo "STAGE CAP $STAGE_CAP reached"; exit 0; }; }
case $STAGE in
A)
  for v in neutral nocheat maycheat; do
    run nano-high-$v --tasks all --variant $v --model gpt-5.4-nano --reasoning-effort high --max-tokens 12000 --samples 1 --max-usd 2.0 --usd-per-m-input 0.20 --usd-per-m-output 1.25
  done
  for v in neutral nocheat maycheat; do
    run gpt55-none-$v --tasks false --variant $v --model gpt-5.5 --reasoning-effort none --max-tokens 1024 --samples 1 --max-usd 1.2 --usd-per-m-input 5.00 --usd-per-m-output 30.00
  done;;
B)
  for v in neutral nocheat maycheat; do
    run mini-high-$v --tasks false --variant $v --model gpt-5.4-mini --reasoning-effort high --max-tokens 12000 --samples 1 --max-usd 5.0 --usd-per-m-input 0.75 --usd-per-m-output 4.50
  done;;
C)
  run gpt55-high-neutral --tasks false --variant neutral --model gpt-5.5 --reasoning-effort high --max-tokens 12000 --samples 1 --max-usd 12.0 --usd-per-m-input 5.00 --usd-per-m-output 30.00;;
esac
echo "STAGE $STAGE DONE total=$total"
