#!/usr/bin/env bash
# Open-weight reasoning rung on one RunPod A100 80 GB: Qwen3-32B (AWQ) with thinking off/on, gpt-oss-20b and
# gpt-oss-120b at reasoning low/high. Inference only, no Lean, no model-written code executed here; rows are
# scored on the laptop with entrypoints/rescore.py. All repos are ungated. A failed model does not block the next.
cd ~/pilot-repo
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p ~/out /workspace/hf
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
log start vllm-install
python -m venv /workspace/vllm-venv && /workspace/vllm-venv/bin/pip install -q vllm -e . > ~/out/vllm-install.log 2>&1 \
  || { log VLLM_INSTALL_FAILED; touch ~/out/DONE; exit 1; }
/workspace/vllm-venv/bin/python -c 'import vllm; print(vllm.__version__)' >> ~/out/steps.log
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt
step() { name=$1 model=$2 rev=$3; shift 3; log "start $name"
  /workspace/vllm-venv/bin/python entrypoints/probe_vllm.py --out ~/out/"$name" --lean none --model "$model" --revision "$rev" --samples 2 "$@" > ~/out/"$name".log 2>&1
  log "end $name exit=$?"; }
fetch() { log "start download $1"; /workspace/vllm-venv/bin/hf download "$1" --revision "$2" --exclude 'original/*' --exclude 'metal/*' > ~/out/download-"${1##*/}".log 2>&1; log "end download $1 exit=$?"; }

M=Qwen/Qwen3-32B-AWQ; R=0499c3ac83fdef8810b907a23894ba91e95eddd8
fetch "$M" "$R"
step qwen32-nothink "$M" "$R" --max-tokens 1024 --max-model-len 2048 --max-seconds 900 --template-kwargs '{"enable_thinking": false}'
step qwen32-think   "$M" "$R" --max-tokens 8192 --max-model-len 9216 --max-seconds 1800 --template-kwargs '{"enable_thinking": true}'

M=openai/gpt-oss-20b; R=6cee5e81ee83917806bbde320786a8fb61efebee
fetch "$M" "$R"
step oss20-low  "$M" "$R" --max-tokens 2048 --max-model-len 3072 --max-seconds 900 --template-kwargs '{"reasoning_effort": "low"}'
step oss20-high "$M" "$R" --max-tokens 8192 --max-model-len 9216 --max-seconds 1800 --template-kwargs '{"reasoning_effort": "high"}'

M=openai/gpt-oss-120b; R=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
fetch "$M" "$R"
step oss120-low  "$M" "$R" --max-tokens 2048 --max-model-len 3072 --max-seconds 900 --template-kwargs '{"reasoning_effort": "low"}'
step oss120-high "$M" "$R" --max-tokens 8192 --max-model-len 9216 --max-seconds 1800 --template-kwargs '{"reasoning_effort": "high"}'
touch ~/out/DONE
