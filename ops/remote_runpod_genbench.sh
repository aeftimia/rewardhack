#!/usr/bin/env bash
# deadline: 65
# vLLM generation benchmark for gpt-oss-20b on one pod (docs/notes-2026-11-21.md, sections 54 to 56). About 13 minutes if it goes as estimated; its inner limits add up to
# about 61 (a 10 minute install limit, a 15 minute readiness wait, a 35 minute benchmark limit), so drive it with a 65 minute deadline.
# No Lean, no judge, no key, no training. A vLLM server holds the base model in 30% of the card, the share left beside a
# trainer that peaks at 62 GiB. Questions: does it load the trainer's LoRA adapter, can it replace it in place, and how
# fast does it generate at 4 to 32 concurrent answers.
# The adapter is the one the trainer saved, copied by hand to payload/adapter before driving (weights are not in git):
#   mkdir -p payload && cp -r results/runpod-smoke-20261125-gen4/train-lr1e-4/adapter-step8 payload/adapter
# Earlier versions of this script timed the training sampler instead: commit 2ae2b6c (default expert path, batches 1 to 32,
# results/runpod-genbench-20261125) and e1d5218 (per-expert loop, batches 8 to 64, results/runpod-genbench-20261125-b).
# Both times the vLLM stage measured nothing: first port 8001 belonged to the image's nginx, which answered the readiness
# check; then the server spent 268 s fetching weights itself and came up after the 3 minute wait had given up.
set -euo pipefail
cd ~/pilot-repo
# VLLM_USE_FLASHINFER_SAMPLER=0: confirmed honoured by 0.34.0 ("sampling disabled via" in the 25 Nov serve log).
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export PATH="/workspace/vllm-venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
mkdir -p ~/out /workspace/hf
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; exit 1; }
MODEL=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee
PORT=8011
ADAPTER=~/pilot-repo/payload/adapter
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
source ops/pod_lib.sh
pod_self_test 40 5 "$PORT"  # its results are timings and logs
[[ -s $ADAPTER/adapter_model.safetensors && -s $ADAPTER/adapter_config.json ]] || fail 'NO_ADAPTER: copy the trained adapter to payload/adapter before driving'

log start install
# Installation took 108 s on 25 Nov; retain ample room for network variability.
# shellcheck disable=SC2015  # the failure branch is meant to run when either step fails
# cap: 2x measured 108 s
python -m venv /workspace/vllm-venv && timeout 600 /workspace/vllm-venv/bin/pip install -q -r requirements-vllm.txt > ~/out/install-vllm.log 2>&1 || fail VLLM_INSTALL_FAILED
/workspace/vllm-venv/bin/pip freeze > ~/out/pip-freeze-vllm.txt
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt

log 'start vllm serve at 0.30 of the card'
/workspace/vllm-venv/bin/vllm serve "$MODEL" --revision "$REV" --port "$PORT" --host 127.0.0.1 --max-model-len 4096 \
  --gpu-memory-utilization 0.30 --enable-lora --max-lora-rank 16 --max-loras 2 > ~/out/vllm-serve.log 2>&1 &
SERVER=$!
trap 'if kill -0 "$SERVER"; then stop_server "$SERVER"; fi' EXIT
# 15 minutes of elapsed time, 2.3x the 6.5 minutes (390 s) it took the one time it was timed: the server fetches the 13 GB of weights itself (268 s last time) and then compiles and warms up (about 90 s).
# cap: 2x measured 390 s
SERVER_WAIT=900
wait_for_server "$SERVER" "$PORT" "$MODEL" "$SERVER_WAIT" || fail 'VLLM_NOT_READY'
refuse_ignored_settings ~/out/vllm-serve.log
log "vllm ready; card memory in use $(gpu_used)"
# The client is standard library plus this package's task list, so the server's own Python runs it from the source tree.
STATUS=0
# The client has never been timed, so its basis is an estimate: about 5 minutes (300 s), which is this script's 13 minute estimate
# less the measured install (1.2) and server start (6.5). Three times that is 900 s. The limit is 2100 s so that it also sits above
# the client's own worst case and fires only on a hang: five measurements bounded by its 300 s per request, plus two adapter
# loads of up to 120 s each, 1740 s in all.
# cap: 3x estimated 300 s
PYTHONPATH=src timeout 2100 /workspace/vllm-venv/bin/python entrypoints/bench_generation.py server --out ~/out/vllm --model "$MODEL" --url "http://127.0.0.1:$PORT/v1" \
  --adapter trained="$ADAPTER" --concurrency 4,16,32 --base-concurrency 4,16 --template-kwargs '{"reasoning_effort": "low"}' > ~/out/vllm.log 2>&1 || STATUS=$?
log "vllm bench exit $STATUS; card memory in use $(gpu_used)"
exit "$STATUS"
