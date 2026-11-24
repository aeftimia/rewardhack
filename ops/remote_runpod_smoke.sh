#!/usr/bin/env bash
# Preflight smoke test for the funded training run (docs/preflight-plan.md, step 3). One RunPod pod with a
# 80-96 GB card, about two hours. NOT YET RUN ON A POD: written on 2026-11-21 from the earlier probe scripts,
# so expect the first launch to find environment problems; every step logs and failures stop the run. Model-written Lean IS executed here (training needs the reward), unlike the inference probes.
#
# Answers before any of the $800 line is spent:
#   1. Only with JUDGE_KIND=pod: does a judge served on this pod agree with the model-read labels? (judge-validation)
#   2. Does TRL GRPO train gpt-oss-20b at all: reward variation, real updates, stable output? (train-*)
#   3. What does a step cost? seconds per step -> dollars per run -> 15 runs against $800.
set -euo pipefail
cd ~/pilot-repo
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
mkdir -p ~/out /workspace/hf
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; exit 1; }
JUDGE_PID=
cleanup() { if [[ -n $JUDGE_PID ]]; then kill "$JUDGE_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
MODEL=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee
JUDGE=${JUDGE_MODEL:-Qwen/Qwen3-8B}; JUDGE_REV=${JUDGE_REV:-b968826d9c46dd6066d109eabc6255188de91218}
# The accepted stage 1 judge (2026-11-24) is gpt-5.4-nano with the checklist prompt, called over the API. The driver
# forwards the key into ~/.judge_key (mode 600) only when asked to; Lean never sees it (env -i, no network).
# JUDGE_KIND=pod serves the small open model here instead, which has not been accepted for a reward.
JUDGE_KIND=${JUDGE_KIND:-api}
if [[ $JUDGE_KIND == api ]]; then
  [[ -s ~/.judge_key ]] || fail 'NO_JUDGE_KEY: run the driver with FORWARD_JUDGE_KEY=1, or set JUDGE_KIND=pod'
  export OPENAI_API_KEY; OPENAI_API_KEY=$(cat ~/.judge_key); rm -f ~/.judge_key
fi

# --- Lean, isolated. Proof bodies are model-written, so Lean runs with an empty environment and no network.
log start lean
mkdir -p /workspace/lean && curl -fsSL https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst | tar --zstd -xf - -C /workspace/lean \
  || fail LEAN_DOWNLOAD_FAILED
LEAN_BIN=/workspace/lean/lean-4.19.0-linux/bin/lean
if unshare -rn true 2>/dev/null; then ISOLATE='unshare -rn'
elif [[ ${LEAN_ISOLATION:-} == none ]]; then ISOLATE=''; log 'WARNING: no network namespace; running Lean with an empty environment only (LEAN_ISOLATION=none)'
else fail 'NO_NETWORK_NAMESPACE: this container cannot unshare. Rerun with LEAN_ISOLATION=none only if running model-written Lean on a disposable pod without network isolation is acceptable.'
fi
printf '#!/bin/bash\nexec env -i PATH=/usr/bin:/bin %s %s "$@"\n' "$ISOLATE" "$LEAN_BIN" > ~/lean-safe && chmod +x ~/lean-safe
~/lean-safe --version >> ~/out/steps.log 2>&1 || fail LEAN_WRAPPER_FAILED

# --- Python: the pinned training stack, and vLLM in its own environment for the judge server.
log start install
python -m venv /workspace/train-venv && /workspace/train-venv/bin/pip install -q torch -r requirements.txt > ~/out/install-train.log 2>&1 || fail TRAIN_INSTALL_FAILED
/workspace/train-venv/bin/pip install -q 'huggingface_hub[cli]' > ~/out/install-hf.log 2>&1
HF=/workspace/train-venv/bin/hf
if [[ $JUDGE_KIND == pod ]]; then python -m venv /workspace/vllm-venv && /workspace/vllm-venv/bin/pip install -q vllm > ~/out/install-vllm.log 2>&1 || fail VLLM_INSTALL_FAILED; fi
PY=/workspace/train-venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt
$PY entrypoints/run.py controls --out ~/out/controls --lean ~/lean-safe > ~/out/controls.log 2>&1 || fail VERIFIER_CONTROLS_FAILED
log 'verifier controls passed'

# --- Downloads (both ungated; no Hugging Face token on this pod).
PAIRS=("$MODEL $REV"); [[ $JUDGE_KIND == pod ]] && PAIRS+=("$JUDGE $JUDGE_REV")
for pair in "${PAIRS[@]}"; do set -- $pair
  log "start download $1"; $HF download "$1" --revision "$2" --exclude 'original/*' --exclude 'metal/*' > ~/out/download-"${1##*/}".log 2>&1 || fail "DOWNLOAD_FAILED $1"
done

if [[ $JUDGE_KIND == api ]]; then
JUDGE_ARGS=(--judge api --judge-model gpt-5.4-nano --judge-usd-in 0.20 --judge-usd-out 1.25 --judge-max-usd "${JUDGE_MAX_USD:-1}")
else
# --- Judge server: a small model in a fixed slice of the card, OpenAI-compatible on localhost.
log start judge-server
/workspace/vllm-venv/bin/vllm serve "$JUDGE" --revision "$JUDGE_REV" --port 8001 --host 127.0.0.1 --max-model-len 3072 \
  --gpu-memory-utilization "${JUDGE_GPU_FRACTION:-0.22}" > ~/out/judge-server.log 2>&1 &
JUDGE_PID=$!
for _ in $(seq 90); do curl -sf http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8001/v1/models > /dev/null || fail JUDGE_SERVER_DID_NOT_START
export NO_KEY=none
JUDGE_ARGS=(--judge api --judge-model "$JUDGE" --judge-url http://127.0.0.1:8001/v1 --judge-key-env NO_KEY --judge-extra '{"chat_template_kwargs": {"enable_thinking": false}}')

# Question 1. 60 rows of each reading category; minutes at vLLM speed, and it gates the reward below.
log start judge-validation
$PY entrypoints/validate_judge.py --out ~/out/judge-validation --per-category 60 --lean ~/lean-safe --require-acceptance "${JUDGE_ARGS[@]}" > ~/out/judge-validation.log 2>&1 || fail JUDGE_VALIDATION_FAILED
log "end judge-validation passed"
fi

# Questions 2 and 3. Low reasoning effort: the unprompted base rate (12/60) was measured there, and high
# effort truncated 89/120 at 8k tokens. Hard task set, neutral prompt, no hint: unseeded exploration.
BASE=(--lean ~/lean-safe --model "$MODEL" --revision "$REV" --reasoning-effort low --thinking unsupported --taskset hard
      --max-completion-length 2048 --num-generations 8 --eval-limit 12 --eval-samples 2 --save-every 8 --stop-no-variation 99
      --arm report_correct --penalty 0.5 "${JUDGE_ARGS[@]}")
# Exploration is measured before updates, on stratified training groups with eight samples each.
log start probe
$PY entrypoints/run.py probe --out ~/out/probe "${BASE[@]}" --train-limit 12 --max-seconds 600 > ~/out/probe.log 2>&1 || fail PROBE_FAILED
step() { name=$1; shift; log "start $name"
  $PY entrypoints/run.py train --out ~/out/"$name" "${BASE[@]}" "$@" > ~/out/"$name".log 2>&1 || fail "$name FAILED"
  log "end $name passed"; nvidia-smi --query-gpu=memory.used --format=csv,noheader >> ~/out/steps.log; }
step train-lr1e-4 --steps 12 --learning-rate 1e-4 --max-seconds 2400 --final-eval-seconds 600
# The shared remote runner bounds total process time; if time remains: the same at a tenth of the learning rate, to see whether 1e-4 is what breaks output format.
step train-lr1e-5 --steps 8 --learning-rate 1e-5 --max-seconds 1200 --final-eval-seconds 420
# ops/remote_job.sh records EXIT_CODE and DONE after this script exits.
