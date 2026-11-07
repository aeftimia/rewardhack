#!/usr/bin/env bash
# 70B exploration probe on a RunPod pod (one 48-80 GB card). Inference only: no Lean, no training, no
# model-written code executed here; rows are scored afterwards on the laptop with entrypoints/rescore.py.
# The AWQ-INT4 repo is ungated, so no Hugging Face token is needed.
cd ~/pilot-repo
# FlashInfer's sampler needs a JIT build (ninja + nvcc) that this container lacks; use the PyTorch sampler.
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p ~/out /workspace/hf
rm -f ~/out/DONE
MODEL=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
REV=2123003760781134cfc31124aa6560a45b491fdf
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
log start vllm-install
python -m venv /workspace/vllm-venv && /workspace/vllm-venv/bin/pip install -q vllm -e . > ~/out/vllm-install.log 2>&1 \
  || { log VLLM_INSTALL_FAILED; touch ~/out/DONE; exit 1; }
/workspace/vllm-venv/bin/python -c 'import vllm; print(vllm.__version__)' >> ~/out/steps.log
log start download
/workspace/vllm-venv/bin/hf download "$MODEL" --revision "$REV" > ~/out/download.log 2>&1 || { log DOWNLOAD_FAILED; touch ~/out/DONE; exit 1; }
log end download
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt
step() { name=$1; shift; log "start $name"
  /workspace/vllm-venv/bin/python entrypoints/probe_vllm.py --out ~/out/"$name" --lean none --model "$MODEL" --revision "$REV" "$@" > ~/out/"$name".log 2>&1
  log "end $name exit=$?"; }
# Same settings as the local 8B neutral probe: 60 tasks x 4 samples, 512-token cap.
step probe-short --samples 4 --max-tokens 512 --max-model-len 2560 --max-seconds 1500
# Room to reason: 60 tasks x 2 samples, 2048-token cap.
step probe-long --samples 2 --max-tokens 2048 --max-model-len 4096 --max-seconds 1800
touch ~/out/DONE
