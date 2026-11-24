#!/usr/bin/env bash
# Shared remote runner: bounds process runtime and atomically records the real exit code.
# This stops the job, NOT RunPod billing. The operator must still terminate the pod.
set -uo pipefail
OUT=${REMOTE_JOB_OUT:-$HOME/out}
mkdir -p "$OUT"
finish() {
  code=$?
  rm -f "$HOME/.judge_key"  # never outlives the job, whichever remote script ran and however it ended
  printf '%s\n' "$code" > "$OUT/EXIT_CODE.tmp"
  mv "$OUT/EXIT_CODE.tmp" "$OUT/EXIT_CODE"
  touch "$OUT/DONE"
}
trap finish EXIT
[[ $# == 2 && $2 =~ ^[1-9][0-9]*$ ]] || exit 2
command -v timeout >/dev/null || { echo 'GNU timeout is required'; exit 2; }
timeout --signal=TERM --kill-after=30 "$2" bash "$1"
