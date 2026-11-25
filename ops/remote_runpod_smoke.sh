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
cleanup() { rm -f ~/.judge_key; if [[ -n $JUDGE_PID ]]; then kill "$JUDGE_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
MODEL=openai/gpt-oss-20b; REV=6cee5e81ee83917806bbde320786a8fb61efebee
JUDGE=${JUDGE_MODEL:-Qwen/Qwen3-8B}; JUDGE_REV=${JUDGE_REV:-b968826d9c46dd6066d109eabc6255188de91218}
# The accepted stage 1 judge (2026-11-24) is gpt-5.4-nano with the checklist prompt, called over the API. The driver
# forwards the key into ~/.judge_key (mode 600) only when asked to. The key is held in an unexported shell variable
# and given only to the Python processes that call the judge (`judged`), so installers and downloads never have it.
# Lean runs with an empty environment, no network and, where the container allows, its own /proc, so a proof body
# cannot read another process's environment; file-reading Lean syntax is also refused before Lean runs.
# JUDGE_KIND=pod serves the small open model here instead, which has not been accepted for a reward.
JUDGE_KIND=${JUDGE_KIND:-api}
if [[ $JUDGE_KIND == api ]]; then
  [[ -s ~/.judge_key ]] || fail 'NO_JUDGE_KEY: run the driver with FORWARD_JUDGE_KEY=1, or set JUDGE_KIND=pod'
  JUDGE_KEY=$(cat ~/.judge_key); rm -f ~/.judge_key
fi
judged() { OPENAI_API_KEY=${JUDGE_KEY:-} "$@"; }

# --- Lean, isolated. Proof bodies are model-written, so Lean runs with an empty environment and no network.
log start lean
mkdir -p /workspace/lean && curl -fsSL https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst | tar --zstd -xf - -C /workspace/lean \
  || fail LEAN_DOWNLOAD_FAILED
LEAN_BIN=/workspace/lean/lean-4.19.0-linux/bin/lean
if unshare -rnpf --mount-proc true 2>/dev/null; then ISOLATE='unshare -rnpf --mount-proc'
elif unshare -rn true 2>/dev/null; then ISOLATE='unshare -rn'; log 'WARNING: no private /proc for Lean; relying on the syntax screen to keep proofs from reading process environments'
elif [[ ${LEAN_ISOLATION:-} == none ]] && setpriv --reuid=65534 --regid=65534 --clear-groups true 2>/dev/null; then
  # No namespaces in this container. An unprivileged user cannot read another user's /proc/<pid>/environ,
  # which is where the judge key lives; the network stays reachable, which is what LEAN_ISOLATION=none accepts.
  ISOLATE='setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs'; READABLE=1
  log 'WARNING: no network namespace; Lean runs as an unprivileged user with an empty environment (LEAN_ISOLATION=none)'
elif [[ ${LEAN_ISOLATION:-} == none ]]; then ISOLATE=''; log 'WARNING: no network namespace; running Lean with an empty environment only (LEAN_ISOLATION=none)'
else fail 'NO_NETWORK_NAMESPACE: this container cannot unshare. Rerun with LEAN_ISOLATION=none only if running model-written Lean on a disposable pod without network isolation is acceptable.'
fi
# The verifier writes each source into a private temporary folder; the unprivileged user needs to read that one file.
OPEN=''; [[ ${READABLE:-0} == 1 ]] && OPEN='for a in "$@"; do [[ -f $a ]] && chmod a+rx "$(dirname "$a")" && chmod a+r "$a"; done'
mkdir -p /workspace/bin
printf '#!/bin/bash\n%s\nexec env -i PATH=/usr/bin:/bin %s %s "$@"\n' "$OPEN" "$ISOLATE" "$LEAN_BIN" > /workspace/bin/lean-safe && chmod 755 /workspace/bin/lean-safe
ln -sf /workspace/bin/lean-safe ~/lean-safe
~/lean-safe --version >> ~/out/steps.log 2>&1 || fail LEAN_WRAPPER_FAILED

# --- Python: the pinned training stack, and vLLM in its own environment for the judge server.
log start install
python -m venv /workspace/train-venv && /workspace/train-venv/bin/pip install -q torch -r requirements.txt > ~/out/install-train.log 2>&1 || fail TRAIN_INSTALL_FAILED
/workspace/train-venv/bin/pip install -q 'huggingface_hub[cli]' > ~/out/install-hf.log 2>&1 || fail HF_CLI_INSTALL_FAILED
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
# JUDGE_MAX_USD bounds the whole script: the cap is enforced per process, and three processes call the judge.
JUDGE_ARGS=(--judge api --judge-model gpt-5.4-nano --judge-usd-in 0.20 --judge-usd-out 1.25
            --judge-max-usd "$(awk -v cap="${JUDGE_MAX_USD:-1}" 'BEGIN { printf "%.4f", cap / 3 }')")
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
      --max-completion-length 2048 --num-generations 8 --eval-limit "${SMOKE_EVAL_LIMIT:-12}" --eval-samples 2 --save-every 8 --stop-no-variation 99
      --arm report_correct --penalty 0.5 "${JUDGE_ARGS[@]}")
# Exploration is measured before updates, on stratified training groups with eight samples each.
# SMOKE_* settings exist because the first pod run measured generation at about 2.5x the time these defaults
# assumed (gpt-oss has only eager attention): a relaunch on the same pod can skip a finished probe and size
# the training step to the measured speed. SMOKE_TOKEN_LIMIT then carries the probe's token-limit decision.
if [[ ${SMOKE_SKIP_PROBE:-0} == 1 ]]; then
  LIMIT=${SMOKE_TOKEN_LIMIT:?SMOKE_SKIP_PROBE=1 needs SMOKE_TOKEN_LIMIT, the limit an earlier probe on this model called for}
  log "probe skipped (run earlier); training with --max-completion-length $LIMIT"
else
log start probe
judged $PY entrypoints/run.py probe --out ~/out/probe "${BASE[@]}" --train-limit "${SMOKE_PROBE_TASKS:-12}" --max-seconds "${SMOKE_PROBE_SECONDS:-1200}" > ~/out/probe.log 2>&1 || fail PROBE_FAILED
# Owner's rule: more than 5% of answers cut off at the token limit means the limit goes up before anything else changes.
LIMIT=2048
if grep -q 'raise the token limit: True' ~/out/probe.log; then LIMIT=4096; fi
log "$(grep 'probe cut-off rate' ~/out/probe.log | tail -1); training with --max-completion-length $LIMIT"
# Owner's rule: more than 5% of answers the verifier could not decide means looking at the verifier before paying for training.
log "$(grep 'probe undecided rate' ~/out/probe.log | tail -1)"
if grep -q 'check the verifier: True' ~/out/probe.log; then fail PROBE_VERIFIER_UNDECIDED; fi
fi
step() { name=$1; shift; log "start $name"
  judged $PY entrypoints/run.py train --out ~/out/"$name" "${BASE[@]}" --max-completion-length "$LIMIT" "$@" > ~/out/"$name".log 2>&1 || fail "$name FAILED"
  log "end $name passed"; nvidia-smi --query-gpu=memory.used --format=csv,noheader >> ~/out/steps.log; }
step train-lr1e-4 --steps "${SMOKE_STEPS:-12}" --learning-rate 1e-4 --max-seconds "${SMOKE_TRAIN_SECONDS:-2400}" --final-eval-seconds "${SMOKE_FINAL_EVAL_SECONDS:-600}"
if [[ -n ${SMOKE_MEMORY_LIMIT:-} ]]; then
  # Memory question only: two steps at a longer token limit. Eager attention holds batch x heads x length^2 per layer,
  # so the limit the cut-off rule asks for may not fit this card at eight generations. It runs last: a failure here costs nothing above.
  step "train-limit$SMOKE_MEMORY_LIMIT" --steps 2 --learning-rate 1e-4 --max-completion-length "$SMOKE_MEMORY_LIMIT" --eval-limit 2 --max-seconds "${SMOKE_MEMORY_SECONDS:-900}" --final-eval-seconds 300
elif [[ ${SMOKE_SECOND_RUN:-1} == 1 ]]; then
# The shared remote runner bounds total process time; if time remains: the same at a tenth of the learning rate, to see whether 1e-4 is what breaks output format.
step train-lr1e-5 --steps 8 --learning-rate 1e-5 --max-seconds 1200 --final-eval-seconds 420
fi
# ops/remote_job.sh records EXIT_CODE and DONE after this script exits.
