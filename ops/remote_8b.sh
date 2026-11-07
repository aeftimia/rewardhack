#!/usr/bin/env bash
# Approved 8B engineering run. Each step has its own time cap; together they end before the 90-minute
# OS shutdown so results can be copied off. A failed step is recorded and does not block the rest.
cd ~/pilot-repo
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1
mkdir -p ~/out
{
  date -u +%FT%TZ
  bash ops/remote_setup.sh || { echo SETUP_FAILED; touch ~/out/DONE; exit 1; }
  date -u +%FT%TZ
} > ~/out/setup.log 2>&1
BASE=(--lean ~/lean-safe --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218)
step() { name=$1; shift; echo "$(date -u +%FT%TZ) start $name" >> ~/out/steps.log
  ~/venv/bin/python entrypoints/run.py "$@" --out ~/out/"$name" "${BASE[@]}" > ~/out/"$name".log 2>&1
  echo "$(date -u +%FT%TZ) end $name exit=$?" >> ~/out/steps.log; }
# Exploration: how often does the untrained model even try the exploit?
step probe-nothink probe --thinking off --max-completion-length 512 --eval-samples 4 --max-seconds 600
step probe-think probe --thinking on --max-completion-length 1536 --eval-samples 2 --max-seconds 900
# Training throughput and whether anything moves, on the neutral prompt.
step train-neutral train --thinking off --max-completion-length 512 --steps 32 --num-generations 8 --eval-samples 2 --stop-no-variation 16 --max-seconds 1200 --final-eval-seconds 300
# Mechanism check, labelled and never pooled: can GRPO amplify the exploit once it is being sampled?
step train-hint-mechanism train --prompt-variant hint --thinking off --max-completion-length 256 --steps 16 --num-generations 8 --eval-samples 2 --stop-no-variation 16 --max-seconds 900 --final-eval-seconds 240
touch ~/out/DONE
