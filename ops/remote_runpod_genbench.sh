#!/usr/bin/env bash
# vLLM generation benchmark for gpt-oss-20b on one pod (docs/notes-2026-11-21.md, sections 54 to 56). About 13 minutes.
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
[[ -s $ADAPTER/adapter_model.safetensors && -s $ADAPTER/adapter_config.json ]] || fail 'NO_ADAPTER: copy the trained adapter to payload/adapter before driving'

log start install
python -m venv /workspace/vllm-venv && /workspace/vllm-venv/bin/pip install -q vllm ninja > ~/out/install-vllm.log 2>&1 || fail VLLM_INSTALL_FAILED
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt

log 'start vllm serve at 0.30 of the card'
/workspace/vllm-venv/bin/vllm serve "$MODEL" --revision "$REV" --port "$PORT" --host 127.0.0.1 --max-model-len 4096 \
  --gpu-memory-utilization 0.30 --enable-lora --max-lora-rank 16 --max-loras 2 > ~/out/vllm-serve.log 2>&1 &
SERVER=$!
READY=0
# Up to 10 minutes: the server fetches the 13 GB of weights itself (268 s last time) and then compiles and warms up (about 90 s).
for _ in $(seq 120); do
  # The answer has to name the model: another service on the port would also answer.
  if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then READY=1; break; fi
  kill -0 "$SERVER" || break
  sleep 5
done
if [[ $READY != 1 ]]; then
  # SIGKILL: the server ignored SIGTERM while it was still fetching weights, and waiting on it hung the last run.
  if kill -0 "$SERVER"; then kill -9 "$SERVER"; fi
  fail 'VLLM_NOT_READY'
fi
log "vllm ready; card memory in use $(gpu_used)"
# The client is standard library plus this package's task list, so the server's own Python runs it from the source tree.
STATUS=0
PYTHONPATH=src timeout 270 /workspace/vllm-venv/bin/python entrypoints/bench_generation.py server --out ~/out/vllm --model "$MODEL" --url "http://127.0.0.1:$PORT/v1" \
  --adapter trained="$ADAPTER" --concurrency 4,16,32 --base-concurrency 4,16 --template-kwargs '{"reasoning_effort": "low"}' > ~/out/vllm.log 2>&1 || STATUS=$?
log "vllm bench exit $STATUS; card memory in use $(gpu_used)"
kill -9 "$SERVER"
exit "$STATUS"
