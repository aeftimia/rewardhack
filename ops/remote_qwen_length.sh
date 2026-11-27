#!/usr/bin/env bash
# deadline: 140
# Inference only: same pinned base model and evaluation draws as the 8k run.
set -euo pipefail
cd ~/pilot-repo
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
mkdir -p ~/out /workspace/hf
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; exit 1; }
source ops/pod_lib.sh
SERVER_PID=
cleanup() { if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID"; then stop_server "$SERVER_PID"; fi; }
trap cleanup EXIT
pod_self_test 1 95 2 "$ROLLOUT_PORT"
log "creating uv bootstrap environment"
python -m venv /workspace/uv-venv
# cap: 2x estimated 60 s
timeout 120 /workspace/uv-venv/bin/pip install uv==0.13.8 > ~/out/install-uv.log 2>&1
UV=/workspace/uv-venv/bin/uv
export UV_CACHE_DIR=/workspace/uv-cache
grep '^numpy==' requirements-vllm.txt > /workspace/probe-override.txt
# cap: 2x measured 600 s (previous install was still running at its timeout)
INSTALL_WAIT=1200 make_env probe -r requirements-vllm.txt -e . --override /workspace/probe-override.txt > ~/out/install-probe.log 2>&1
HF=$(env_bin probe hf)
log download
# cap: 2x measured 232 s (latest Qwen download, 27 Nov)
DOWNLOAD_WAIT=480 fetch_model "$QWEN3_32B" "$QWEN3_32B_REV"
LIMIT=${SMOKE_TOKEN_LIMIT:?}
[[ $LIMIT == 16384 ]] || fail EXPECTED_16384_TOKEN_DIAGNOSTIC
log server
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  in_env probe vllm serve "$QWEN3_32B" --revision "$QWEN3_32B_REV" \
    --served-model-name "$QWEN3_32B@$QWEN3_32B_REV" --host 127.0.0.1 --port "$ROLLOUT_PORT" \
    --max-model-len $((LIMIT + PROMPT_ROOM_TOKENS)) --gpu-memory-utilization "${MODEL_ROLLOUT_FRACTION[1]}" \
    --max-num-seqs 16 --max-cudagraph-capture-size 32 --enable-lora --max-lora-rank 16 --max-loras 2 \
    --enable-sleep-mode --enable-scale-out --logprobs-mode processed_logprobs > ~/out/rollout-server.log 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_PID" "$ROLLOUT_PORT" "$QWEN3_32B@$QWEN3_32B_REV" "$SERVER_WAIT"
refuse_ignored_settings ~/out/rollout-server.log
log inference
# cap: 1.5x estimated 3600 s (24x16k generation, allowing longer-context slowdown)
in_env probe timeout 5400 python entrypoints/qwen_length_measure.py --out ~/out/inference \
  --model "$QWEN3_32B" --revision "$QWEN3_32B_REV" --rollout-url "http://127.0.0.1:$ROLLOUT_PORT" \
  --max-completion-length "$LIMIT" --seed "$PROBE_SEED" > ~/out/inference.log 2>&1
log completed
touch ~/out/DONE
