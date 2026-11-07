#!/usr/bin/env bash
# Retry of the four gpt-oss runs on the same pod. First attempt failed because vLLM chose the FlashInfer
# attention backend for gpt-oss (attention sinks) and its JIT build needs ninja, and because the
# gpt-oss-120b download had a truncated shard. Fixes: ninja installed, Triton attention backend forced,
# download re-verified. Inference only; rows scored on the laptop.
cd ~/pilot-repo
# ninja lives in the venv and nvcc under /usr/local/cuda; neither was on PATH, which is why the JIT build failed.
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN
export PATH="/workspace/vllm-venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
mkdir -p ~/out
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
log start retry2
pkill -f "[E]ngineCore" ; pkill -f "[p]robe_vllm"; sleep 3   # leftover engine processes still held GPU memory
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> ~/out/steps.log
which ninja nvcc >> ~/out/steps.log 2>&1
step() { name=$1 model=$2 rev=$3; shift 3; log "start $name (retry)"
  /workspace/vllm-venv/bin/python entrypoints/probe_vllm.py --out ~/out/"$name" --lean none --model "$model" --revision "$rev" --samples 2 "$@" > ~/out/"$name".log 2>&1
  log "end $name exit=$?"; }
M=openai/gpt-oss-20b; R=6cee5e81ee83917806bbde320786a8fb61efebee
step oss20-low  "$M" "$R" --max-tokens 2048 --max-model-len 3072 --max-seconds 600 --template-kwargs '{"reasoning_effort": "low"}'
step oss20-high "$M" "$R" --max-tokens 8192 --max-model-len 9216 --max-seconds 900 --template-kwargs '{"reasoning_effort": "high"}'
M=openai/gpt-oss-120b; R=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
log "start re-verify download $M"
/workspace/vllm-venv/bin/hf download "$M" --revision "$R" --exclude 'original/*' --exclude 'metal/*' > ~/out/download-gpt-oss-120b-retry.log 2>&1; log "end re-verify exit=$?"
step oss120-low  "$M" "$R" --max-tokens 2048 --max-model-len 3072 --max-seconds 600 --template-kwargs '{"reasoning_effort": "low"}'
step oss120-high "$M" "$R" --max-tokens 8192 --max-model-len 9216 --max-seconds 900 --template-kwargs '{"reasoning_effort": "high"}'
touch ~/out/DONE
