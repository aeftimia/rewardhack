#!/usr/bin/env bash
# Preflight smoke test for the funded training run (docs/preflight-plan.md, step 3). One RunPod pod with a
# 80-96 GB card, about two hours. NOT YET RUN ON A POD: written on 2026-11-21 from the earlier probe scripts,
# so expect the first launch to find environment problems; every step logs and a failed step does not block
# the next. Model-written Lean IS executed here (training needs the reward), unlike the inference probes.
#
# Answers three questions before any of the $800 line is spent:
#   1. Does the judge served on this pod agree with the model-read labels?           (judge-validation)
#   2. Does TRL GRPO train gpt-oss-20b at all: reward variation, real updates, stable output? (train-*)
#   3. What does a step cost? seconds per step -> dollars per run -> 15 runs against $800.
cd ~/pilot-repo
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_ATTN
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
mkdir -p ~/out /workspace/hf
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; touch ~/out/DONE; exit 1; }
MODEL=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee
JUDGE=${JUDGE_MODEL:-Qwen/Qwen3-8B}; JUDGE_REV=${JUDGE_REV:-b968826d9c46dd6066d109eabc6255188de91218}

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
python -m venv /workspace/vllm-venv && /workspace/vllm-venv/bin/pip install -q vllm > ~/out/install-vllm.log 2>&1 || fail VLLM_INSTALL_FAILED
PY=/workspace/train-venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt
$PY entrypoints/run.py controls --out ~/out/controls --lean ~/lean-safe > ~/out/controls.log 2>&1 || fail VERIFIER_CONTROLS_FAILED
log 'verifier controls passed'

# --- Downloads (both ungated; no Hugging Face token on this pod).
for pair in "$MODEL $REV" "$JUDGE $JUDGE_REV"; do set -- $pair
  log "start download $1"; /workspace/vllm-venv/bin/hf download "$1" --revision "$2" --exclude 'original/*' --exclude 'metal/*' > ~/out/download-"${1##*/}".log 2>&1 || fail "DOWNLOAD_FAILED $1"
done

# --- Judge server: a small model in a fixed slice of the card, OpenAI-compatible on localhost.
log start judge-server
/workspace/vllm-venv/bin/vllm serve "$JUDGE" --revision "$JUDGE_REV" --port 8001 --host 127.0.0.1 --max-model-len 3072 \
  --gpu-memory-utilization "${JUDGE_GPU_FRACTION:-0.22}" > ~/out/judge-server.log 2>&1 &
for _ in $(seq 90); do curl -sf http://127.0.0.1:8001/v1/models > /dev/null && break; sleep 5; done
curl -sf http://127.0.0.1:8001/v1/models > /dev/null || fail JUDGE_SERVER_DID_NOT_START
export NO_KEY=none
JUDGE_ARGS=(--judge api --judge-model "$JUDGE" --judge-url http://127.0.0.1:8001/v1 --judge-key-env NO_KEY --judge-extra '{"chat_template_kwargs": {"enable_thinking": false}}')

# Question 1. 60 rows of each reading category; minutes at vLLM speed, and it gates the reward below.
log start judge-validation
$PY entrypoints/validate_judge.py --out ~/out/judge-validation --per-category 60 "${JUDGE_ARGS[@]}" > ~/out/judge-validation.log 2>&1
log "end judge-validation exit=$?"

# Questions 2 and 3. Low reasoning effort: the unprompted base rate (12/60) was measured there, and high
# effort truncated 89/120 at 8k tokens. Hard task set, neutral prompt, no hint: unseeded exploration.
BASE=(--lean ~/lean-safe --model "$MODEL" --revision "$REV" --reasoning-effort low --thinking unsupported --taskset hard
      --max-completion-length 2048 --num-generations 8 --eval-limit 12 --eval-samples 2 --save-every 8 --stop-no-variation 99
      --arm report_correct --penalty 0.5 "${JUDGE_ARGS[@]}")
step() { name=$1; shift; log "start $name"
  $PY entrypoints/run.py train --out ~/out/"$name" "${BASE[@]}" "$@" > ~/out/"$name".log 2>&1
  log "end $name exit=$?"; nvidia-smi --query-gpu=memory.used --format=csv,noheader >> ~/out/steps.log; }
step train-lr1e-4 --steps 12 --learning-rate 1e-4 --max-seconds 2700 --final-eval-seconds 600
# Only if the first run left time: the same at a tenth of the learning rate, to see whether 1e-4 is what breaks output format.
step train-lr1e-5 --steps 8 --learning-rate 1e-5 --max-seconds 1500 --final-eval-seconds 420
pkill -f 'vllm serve' || true
touch ~/out/DONE
