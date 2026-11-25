#!/usr/bin/env bash
# Generation cost benchmark for gpt-oss-20b on one pod (docs/notes-2026-11-21.md, sections 54 and 55). About 15 minutes.
# No Lean, no judge, no key, no training: it times generation two ways and nothing else.
#   hf-eager  the training sampler as the trainer loads it (weights expanded to bf16), with the expert layer's plain per-expert loop:
#             time per decoding step at batches 8 to 64, with a profile
#   vllm      a vLLM server holding the base model in a fixed share of the card: does it load the trainer's adapter, and how fast is it
# First run, 2026-11-25 (this script at commit 2ae2b6c, results/runpod-genbench-20261125): the default expert path was timed at batches 1 to 32. The default expert path copies each chosen expert's weights for every token while decoding,
# so a step costs about 8 ms per sequence however large the batch. The vllm stage measured nothing: port 8001 belongs to the
# pod image's nginx, which answered the readiness check. A 4-bit stage (`kernels` package) was dropped: transformers will only
# load those kernels with trust_remote_code, which is the owner's decision.
# The two are separate measurements, so each is run under `if`: a failed one is logged by name, the others still run,
# and the script exits 1 at the end. Setup failures stop it at once.
set -euo pipefail
cd ~/pilot-repo
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
mkdir -p ~/out /workspace/hf /workspace/adapters
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; exit 1; }
FAILED=0
MODEL=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee
PY=/workspace/train-venv/bin/python
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
stop_server() { kill "$SERVER"; if wait "$SERVER"; then log 'vllm serve exit 0'; else log "vllm serve exit $?"; fi; }

# vLLM installs in the background from the start: it is the slowest install and needs neither the model nor the GPU.
log start install
( python -m venv /workspace/vllm-venv && /workspace/vllm-venv/bin/pip install -q vllm ninja > ~/out/install-vllm.log 2>&1 ) &
VLLM_INSTALL=$!
python -m venv /workspace/train-venv && /workspace/train-venv/bin/pip install -q torch -r requirements.txt 'huggingface_hub[cli]' > ~/out/install-train.log 2>&1 || fail TRAIN_INSTALL_FAILED
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt
log "start download $MODEL"
/workspace/train-venv/bin/hf download "$MODEL" --revision "$REV" --exclude 'original/*' --exclude 'metal/*' > ~/out/download.log 2>&1 || fail DOWNLOAD_FAILED

BENCH=(entrypoints/bench_generation.py hf --model "$MODEL" --revision "$REV" --reasoning-effort low --attn-implementation eager --adapters /workspace/adapters)
log start hf-eager
if timeout 200 "$PY" "${BENCH[@]}" --out ~/out/hf-eager --experts-implementation eager --batches 8,16,32,64 --lora-batches 32 --focus-batch 32 \
     --long-tokens 192 --profile-tokens 12 > ~/out/hf-eager.log 2>&1; then log 'end hf-eager passed'
else log "hf-eager FAILED exit=$?"; FAILED=1; fi

log start vllm
wait "$VLLM_INSTALL" || fail VLLM_INSTALL_FAILED
[[ -s /workspace/adapters/all-linear/adapter_model.safetensors ]] || fail 'NO_ADAPTER: the hf stage did not save one'
export PATH="/workspace/vllm-venv/bin:$PATH"
READY=0
PORT=8011
# The trainer peaks at 62 GiB of this 96 GB card, so vLLM has to live in about 30% of it to share the card. The larger share is
# tried only if the smaller one does not come up, and which one worked is the first finding.
for FRACTION in 0.30 0.60; do
  log "start vllm serve at $FRACTION of the card"
  /workspace/vllm-venv/bin/vllm serve "$MODEL" --revision "$REV" --port "$PORT" --host 127.0.0.1 --max-model-len 4096 \
    --gpu-memory-utilization "$FRACTION" --enable-lora --max-lora-rank 16 --max-loras 2 > ~/out/vllm-serve-"$FRACTION".log 2>&1 &
  SERVER=$!
  for _ in $(seq 36); do
    # The answer has to name the model: another service on the port (the image's nginx on 8001) would also answer.
    if curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then READY=1; break; fi
    kill -0 "$SERVER" || break
    sleep 5
  done
  if [[ $READY == 1 ]]; then log "vllm ready at $FRACTION; card memory in use $(gpu_used)"; break; fi
  log "vllm did not come up at $FRACTION"
  if kill -0 "$SERVER"; then stop_server; fi
done
[[ $READY == 1 ]] || fail 'vllm FAILED: the server did not start at either share'
if timeout 240 "$PY" entrypoints/bench_generation.py server --out ~/out/vllm --model "$MODEL" --url "http://127.0.0.1:$PORT/v1" \
     --adapter trained=/workspace/adapters/all-linear --concurrency 4,16,32 --base-concurrency 16 --template-kwargs '{"reasoning_effort": "low"}' > ~/out/vllm.log 2>&1; then log "end vllm passed; card memory in use $(gpu_used)"
else log "vllm bench FAILED exit=$?"; FAILED=1; fi
stop_server
exit "$FAILED"
