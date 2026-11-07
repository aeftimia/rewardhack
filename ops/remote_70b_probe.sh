#!/usr/bin/env bash
# 70B exploration probe on one g6e.xlarge (L40S 48 GB). Inference only, no training, no Hugging Face token:
# the AWQ-INT4 repo is ungated. Two passes on the neutral prompt: a short-answer pass matching the local
# 8B probes, and a longer pass to see whether the larger model reasons before answering.
cd ~/pilot-repo
export PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1
mkdir -p ~/out
MODEL=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
REV=2123003760781134cfc31124aa6560a45b491fdf
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
log start setup
bash ops/remote_setup.sh > ~/out/setup.log 2>&1 || { log SETUP_FAILED; touch ~/out/DONE; exit 1; }
log start vllm-install
uv venv ~/vllm-venv --python 3.12 -q && uv pip install -q --python ~/vllm-venv/bin/python vllm -e . > ~/out/vllm-install.log 2>&1 \
  || { log VLLM_INSTALL_FAILED; touch ~/out/DONE; exit 1; }
~/vllm-venv/bin/python -c 'import vllm; print(vllm.__version__)' >> ~/out/steps.log
log start download
~/vllm-venv/bin/hf download "$MODEL" --revision "$REV" > ~/out/download.log 2>&1 || { log DOWNLOAD_FAILED; touch ~/out/DONE; exit 1; }
log end download
step() { name=$1; shift; log "start $name"
  ~/vllm-venv/bin/python entrypoints/probe_vllm.py --out ~/out/"$name" --lean ~/lean-safe --model "$MODEL" --revision "$REV" "$@" > ~/out/"$name".log 2>&1
  log "end $name exit=$?"; }
# Same settings as the local 8B neutral probe: 60 tasks x 4 samples, 512-token cap.
step probe-short --samples 4 --max-tokens 512 --max-model-len 2560 --max-seconds 1500
# Room to reason: 60 tasks x 2 samples, 2048-token cap.
step probe-long --samples 2 --max-tokens 2048 --max-model-len 4096 --max-seconds 1800
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt
touch ~/out/DONE
