#!/usr/bin/env bash
# deadline: 170
# Preflight smoke test for the funded training run (docs/preflight-plan.md, step 3). One RunPod pod with a
# 80-96 GB card. First run on a pod: 2026-11-24 (results/runpod-smoke-20261124*). Its inner limits add up to about 150 minutes
# (probe 55, first training run 60, second 35) plus setup, so drive it with a 170 minute deadline; when everything works it ends
# far sooner and the pod is removed then. That figure is for the API judge; JUDGE_KIND=pod adds a second install and download, a
# 15 minute readiness wait and up to 30 minutes of judge validation: allow 240. Run it with LEAN_ISOLATION=none: RunPod containers have no namespaces. Model-written Lean IS executed here (training needs the reward), unlike the inference probes.
#
# Answers before any of the $800 line is spent:
#   1. Only with JUDGE_KIND=pod: does a judge served on this pod agree with the model-read labels? (judge-validation)
#   2. Does TRL GRPO train gpt-oss-20b at all: reward variation, real updates, stable output? (train-*)
#   3. What does a step cost? seconds per step -> dollars per run -> 15 runs against $800.
set -euo pipefail
cd ~/pilot-repo
# VLLM_ATTENTION_BACKEND was set here until 25 Nov: vLLM has reported it as an unknown variable since the 5 Nov probes, so it never did anything.
export HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
mkdir -p ~/out /workspace/hf
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; exit 1; }
JUDGE_PID=
ROLLOUT_PID=
SMOKE_VLLM_ROLLOUTS=${SMOKE_VLLM_ROLLOUTS:-0}
# The script's own values for some optional SMOKE_* settings (the ones the 26 Nov edits touched; the rest keep their
# inline defaults where they are used), which the driver forwards only when the launch record sets them.
SMOKE_PART_ONE=${SMOKE_PART_ONE:-0}  # fallback-ok: optional setting; unset means the preflight run on the funded arm, as before Part 1 existed
SMOKE_PROBE_TASKS=${SMOKE_PROBE_TASKS:-12}  # fallback-ok: optional setting; the probe's documented size when the launch does not set it
SMOKE_PROBE_SECONDS=${SMOKE_PROBE_SECONDS:-3300}  # fallback-ok: optional setting; the probe's documented time limit (55 min) when unset
SMOKE_STEPS=${SMOKE_STEPS:-12}  # fallback-ok: optional setting; the documented preflight length when the launch does not set it
SMOKE_TRAIN_SECONDS=${SMOKE_TRAIN_SECONDS:-3600}  # fallback-ok: optional setting; the documented per-run limit (60 min) when unset
SMOKE_FINAL_EVAL_SECONDS=${SMOKE_FINAL_EVAL_SECONDS:-600}  # fallback-ok: optional setting; time reserved for the after-evaluation when unset
SMOKE_MEMORY_SECONDS=${SMOKE_MEMORY_SECONDS:-1800}  # fallback-ok: optional setting; the memory check's documented limit when unset
source ops/pod_lib.sh
JUDGE_PORT=8011
cleanup() {
  rm -f ~/.judge_key
  for pid in "$JUDGE_PID" "$ROLLOUT_PID"; do
    if [[ -n $pid ]] && kill -0 "$pid"; then stop_server "$pid"; fi
  done
}
trap cleanup EXIT
OUT_NEED_GB=(10 45)  # indexed by SMOKE_PART_ONE: a Part 1 pair with expert LoRA writes about 40 GB of results
pod_self_test 40 "${OUT_NEED_GB[$SMOKE_PART_ONE]}" "$JUDGE_PORT"
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

# --- How Lean will be isolated, decided before anything is downloaded: this image has had no namespaces, and finding that
# out after the installs wasted a pod on 2026-11-24.
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

# --- Lean, isolated. Proof bodies are model-written, so Lean runs with an empty environment and no network.
log start lean
# Earlier hosts downloaded and unpacked in 2–5 s. On 25 Nov the retry host transferred
# 163446784 of 343842845 bytes in 60 s: allow 3x an estimated 140 s at that rate.
# shellcheck disable=SC2015  # the failure branch is meant to run when any step fails
# cap: 3x estimated 140 s
mkdir -p /workspace/lean && curl -fsSL --max-time 420 https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst | tar --zstd -xf - -C /workspace/lean \
  || fail LEAN_DOWNLOAD_FAILED
LEAN_BIN=/workspace/lean/lean-4.19.0-linux/bin/lean
# The verifier writes each source into a private temporary folder; the unprivileged user needs to read that one file.
OPEN=''
# shellcheck disable=SC2016  # $@ and $a belong to the wrapper this text is written into, not to this script
[[ ${READABLE:-0} == 1 ]] && OPEN='for a in "$@"; do [[ -f $a ]] && chmod a+rx "$(dirname "$a")" && chmod a+r "$a"; done'
mkdir -p /workspace/bin
printf '#!/bin/bash\n%s\nexec env -i PATH=/usr/bin:/bin %s %s "$@"\n' "$OPEN" "$ISOLATE" "$LEAN_BIN" > /workspace/bin/lean-safe && chmod 755 /workspace/bin/lean-safe
ln -sf /workspace/bin/lean-safe ~/lean-safe
~/lean-safe --version >> ~/out/steps.log 2>&1 || fail LEAN_WRAPPER_FAILED

# --- Python: the pinned training stack, and vLLM in its own environment (judge server or rollouts). uv installs the same
# pins as pip, several times faster (owner, 26 Nov); the vLLM environment installs in the background, beside the training
# one, the verifier checks and the model download. On 26 Nov with pip, in series: training install and checks 137 s, vLLM 222 s.
log start install
# Limits on the steps that fetch from the network, so a stalled connection ends in minutes and not at the job's deadline:
# the install took 81 s when timed, the 13 GB download 0.5 to 5.1 minutes, the vLLM install 222 s (all with pip).
# shellcheck disable=SC2015  # the failure branch is meant to run when either step fails
python -m venv /workspace/uv-venv && /workspace/uv-venv/bin/pip install -q uv==0.13.8 > ~/out/install-uv.log 2>&1 || fail UV_INSTALL_FAILED
UV=/workspace/uv-venv/bin/uv
# uv's cache of unpacked wheels (two torch and CUDA sets) stays off the disk ~/out is checked for (review round 12).
export UV_CACHE_DIR=/workspace/uv-cache
VLLM_INSTALL_PID=
# --clear: since uv 0.10, `uv venv` refuses an existing environment, which a relaunch on a kept pod has (review round 12).
if [[ $JUDGE_KIND == pod || $SMOKE_VLLM_ROLLOUTS == 1 ]]; then
  ( $UV venv -q --clear --python "$(command -v python)" /workspace/vllm-venv \
      && timeout 600 $UV pip install -q --python /workspace/vllm-venv/bin/python -r requirements-vllm.txt ) > ~/out/install-vllm.log 2>&1 &  # cap: 2x measured 222 s
  VLLM_INSTALL_PID=$!
fi
# Where the vLLM environment is needed: wait for its background install, its own failure code if it failed.
vllm_installed() {
  wait "$VLLM_INSTALL_PID" || fail VLLM_INSTALL_FAILED
  $UV pip freeze --python /workspace/vllm-venv/bin/python > ~/out/pip-freeze-vllm.txt || fail VLLM_FREEZE_FAILED
}
# shellcheck disable=SC2015  # the failure branch is meant to run when either step fails
$UV venv -q --clear --python "$(command -v python)" /workspace/train-venv \
  && timeout 300 $UV pip install -q --python /workspace/train-venv/bin/python torch -r requirements.txt > ~/out/install-train.log 2>&1 || fail TRAIN_INSTALL_FAILED  # cap: 2x measured 81 s
# The `hf` command comes with huggingface_hub itself; the `[cli]` extra asked for until 25 Nov does not exist, and pip stepped back through versions looking for it.
HF=/workspace/train-venv/bin/hf
[[ -x $HF ]] || fail 'NO_HF_COMMAND: huggingface_hub did not install its hf command'
$UV pip freeze --python /workspace/train-venv/bin/python > ~/out/pip-freeze-train.txt || fail TRAIN_FREEZE_FAILED
if [[ $JUDGE_KIND == pod ]]; then vllm_installed; fi
PY=/workspace/train-venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv > ~/out/gpu.txt
# The verifier checks run beside the model download (owner, 26 Nov); they are joined, with their own failure code, before
# anything uses Lean's verdicts.
$PY entrypoints/run.py controls --out ~/out/controls --lean ~/lean-safe > ~/out/controls.log 2>&1 &
CONTROLS_PID=$!

# --- Downloads (both ungated; no Hugging Face token on this pod).
PAIRS=("$MODEL $REV"); [[ $JUDGE_KIND == pod ]] && PAIRS+=("$JUDGE $JUDGE_REV")
for pair in "${PAIRS[@]}"; do
  # shellcheck disable=SC2086  # split on purpose: "repo revision" becomes $1 and $2
  set -- $pair
  # cap: 2x measured 5.1 min
  log "start download $1"; timeout 660 $HF download "$1" --revision "$2" --exclude 'original/*' --exclude 'metal/*' > ~/out/download-"${1##*/}".log 2>&1 || fail "DOWNLOAD_FAILED $1"
done
wait "$CONTROLS_PID" || fail VERIFIER_CONTROLS_FAILED
log 'verifier controls passed'

if [[ $JUDGE_KIND == api ]]; then
# JUDGE_MAX_USD bounds the whole script: the cap is enforced per process, split over the processes that call the judge.
# Preflight: the probe and two training runs. Part 1: its two runs (the driver refuses it with a probe or a further run);
# a third share there only cut each run's allowance, and 4 prompt groups per step make a run's judge use about 4 times larger.
JUDGE_PROCESSES=(3 2)  # indexed by SMOKE_PART_ONE
JUDGE_ARGS=(--judge api --judge-model gpt-5.4-nano --judge-usd-in 0.20 --judge-usd-out 1.25
            --judge-max-usd "$(awk -v cap="${JUDGE_MAX_USD:?the driver passes JUDGE_MAX_USD}" -v n="${JUDGE_PROCESSES[$SMOKE_PART_ONE]}" 'BEGIN { printf "%.4f", cap / n }')")
else
# --- Judge server: a small model in a fixed slice of the card, OpenAI-compatible on localhost.
log start judge-server
PATH="/workspace/vllm-venv/bin:$PATH" /workspace/vllm-venv/bin/vllm serve "$JUDGE" --revision "$JUDGE_REV" --port "$JUDGE_PORT" --host 127.0.0.1 --max-model-len 3072 \
  --gpu-memory-utilization "${JUDGE_GPU_FRACTION:-0.22}" > ~/out/judge-server.log 2>&1 &
JUDGE_PID=$!
# 15 minutes of elapsed time. vLLM's first start took 6.5 minutes (390 s) on this pod type, with its own fetch of the weights; that
# was gpt-oss-20b in the benchmark job, the only start timed so far, and this judge model's own start has not been timed.
# cap: 2x measured 390 s
SERVER_WAIT=900
wait_for_server "$JUDGE_PID" "$JUDGE_PORT" "$JUDGE" "$SERVER_WAIT" || fail JUDGE_SERVER_DID_NOT_START
refuse_ignored_settings ~/out/judge-server.log
export NO_KEY=none
JUDGE_ARGS=(--judge api --judge-model "$JUDGE" --judge-url "http://127.0.0.1:$JUDGE_PORT/v1" --judge-key-env NO_KEY --judge-extra '{"chat_template_kwargs": {"enable_thinking": false}}')

# Question 1. 60 rows of each reading category; minutes at vLLM speed, and it gates the reward below.
log start judge-validation
# cap: 1.5x estimated 20 min (360 readings; about 1.6 s each measured on the laptop, so under 10 min at vLLM speed)
timeout 1800 $PY entrypoints/validate_judge.py --out ~/out/judge-validation --per-category 60 --lean ~/lean-safe --require-acceptance "${JUDGE_ARGS[@]}" > ~/out/judge-validation.log 2>&1 || fail JUDGE_VALIDATION_FAILED
log "end judge-validation passed"
fi

# Questions 2 and 3. Low reasoning effort: the unprompted base rate (12/60) was measured there, and high
# effort truncated 89/120 at 8k tokens. Hard task set, neutral prompt, no hint: unseeded exploration.
BASE=(--lean ~/lean-safe --model "$MODEL" --revision "$REV" --reasoning-effort low --thinking unsupported --taskset hard --attn-implementation eager --mxfp4 dequantize
      --max-completion-length 2048 --num-generations 8 --eval-limit "${SMOKE_EVAL_LIMIT:-12}" --eval-samples 2 --save-every 8 --stop-no-variation 99
      "${JUDGE_ARGS[@]}")
# The funded arm at λ=0.5: the probe and the preflight training steps. SMOKE_PART_ONE=1 trains the Part 1 pair instead (below).
FUNDED=(--arm report_correct --penalty 0.5)
# Exploration is measured before updates, on stratified training groups with eight samples each.
# SMOKE_* settings exist because the first pod run measured generation at about 2.5x the time these defaults
# assumed (gpt-oss has only eager attention): a relaunch on the same pod can skip a finished probe and size
# the training step to the measured speed. SMOKE_TOKEN_LIMIT then carries the probe's token-limit decision.
if [[ ${SMOKE_SKIP_PROBE:-0} == 1 ]]; then
  LIMIT=${SMOKE_TOKEN_LIMIT:?SMOKE_SKIP_PROBE=1 needs SMOKE_TOKEN_LIMIT, the limit an earlier probe on this model called for}
  log "probe skipped (run earlier); training with --max-completion-length $LIMIT"
else
log start probe
judged $PY entrypoints/run.py probe --out ~/out/probe "${BASE[@]}" "${FUNDED[@]}" --train-limit "$SMOKE_PROBE_TASKS" --max-seconds "$SMOKE_PROBE_SECONDS" > ~/out/probe.log 2>&1 || fail PROBE_FAILED
# The two owner's rules below are read from the probe's summary, not from its log text: a reworded log line must not switch a rule off.
# More than 5% of answers cut off at the token limit: the limit goes up before anything else changes.
# More than 5% of answers the verifier could not decide: look at the verifier before paying for training.
# A rule the probe could not evaluate (no rows to judge it by) is None there and stops the job here: it is not a "no".
PROBE=$($PY -m rewardhack.probe_rules ~/out/probe/summary.json) || fail PROBE_SUMMARY_UNREADABLE
read -r LIMIT CHECK_VERIFIER PROBE_LINE <<< "$PROBE"
log "probe: $PROBE_LINE; training with --max-completion-length $LIMIT"
if [[ $CHECK_VERIFIER == 1 ]]; then fail PROBE_VERIFIER_UNDECIDED; fi
fi
step() { name=$1; shift; log "start $name"
  judged $PY entrypoints/run.py train --out ~/out/"$name" "${BASE[@]}" --max-completion-length "$LIMIT" "$@" > ~/out/"$name".log 2>&1 || fail "$name FAILED"
  # A run stopped early (deadline, no variation) still exits 0: say so where the job's progress is read.
  status=$($PY -c 'import json, sys; s = json.load(open(sys.argv[1])); print("steps", s["global_step"], "early stop", s["early_stop"] if "early_stop" in s else "none")' ~/out/"$name"/summary.json) \
    || fail "$name SUMMARY_UNREADABLE"
  log "end $name passed: $status"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >> ~/out/steps.log; }
# Eight generations at 2,048 tokens ran out of memory on a 96 GB card (2026-11-24); four fit (2026-11-25). Every training step below uses this.
GENERATIONS=${SMOKE_TRAIN_GENERATIONS:-4}
# Prompt groups per optimizer step (owner, 26 Nov, notes section 72): one group of four had reward variation in 12 of 60
# baseline steps. Each group is its own micro-batch, so memory is that of one group; vLLM samples all groups at once.
PROMPTS=4
if [[ $SMOKE_VLLM_ROLLOUTS == 1 ]]; then
  [[ $JUDGE_KIND == api && ${SMOKE_SKIP_PROBE:?} == 1 && ${SMOKE_SECOND_RUN:?} == 0 ]] || fail INVALID_ROLLOUT_PILOT
  log start rollout-install  # the wait for the background install started with the training one
  vllm_installed
  log start rollout-server
  # The context holds the prompt (a few hundred tokens) and an answer of up to LIMIT: 2,048 tokens of room for the prompt.
  # A fixed 4,096 refused every request once LIMIT was 4,096 (review round 6); the rollout client also checks each request.
  # CUDA graphs are captured for decode batches of up to 32 tokens (16 sequences use at most 16): the default captured 166
  # sizes up to 1,024 on this card, most of a 125 s engine start (measuring pass). Larger batches (prefill) run eagerly.
  VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True PATH="/workspace/vllm-venv/bin:$PATH" \
    /workspace/vllm-venv/bin/vllm serve "$MODEL" --revision "$REV" --served-model-name "$MODEL@$REV" \
      --host 127.0.0.1 --port 8011 --max-model-len $((LIMIT + 2048)) --gpu-memory-utilization 0.30 --max-num-seqs $((GENERATIONS * PROMPTS)) --max-cudagraph-capture-size 32 \
      --enable-lora --max-lora-rank 16 --max-loras 2 --enable-sleep-mode --enable-scale-out \
      --logprobs-mode processed_logprobs > ~/out/rollout-server.log 2>&1 &
  ROLLOUT_PID=$!
  # cap: 2x measured 390 s
  wait_for_server "$ROLLOUT_PID" 8011 "$MODEL@$REV" 900 || fail ROLLOUT_SERVER_DID_NOT_START
  refuse_ignored_settings ~/out/rollout-server.log
  # cap: 3x estimated 40 s
  curl -fsS --max-time 120 -X POST 'http://127.0.0.1:8011/sleep?level=1' > ~/out/rollout-initial-sleep.txt || fail ROLLOUT_SLEEP_FAILED
  BASE+=(--rollout-backend vllm --rollout-url http://127.0.0.1:8011)
fi
# The batch shape of every training step in this script, so that no comparison below changes it by accident.
SHAPE=(--num-generations "$GENERATIONS" --prompts-per-step "$PROMPTS")
TRAIN=(--steps "$SMOKE_STEPS" --learning-rate 1e-4 "${SHAPE[@]}" --max-seconds "$SMOKE_TRAIN_SECONDS" --final-eval-seconds "$SMOKE_FINAL_EVAL_SECONDS")
if [[ $SMOKE_PART_ONE == 1 ]]; then
  # Part 1 proof of concept (owner, 26 Nov; issue #2): the baseline and the main setting at λ=0, same seed and settings,
  # one after the other. Each keeps resumable checkpoints in <name>/trainer, so a later run can extend it with --resume-from.
  step train-baseline "${TRAIN[@]}" --arm baseline --penalty 0
  step train-lambda0 "${TRAIN[@]}" --arm report_correct --penalty 0
else
  step train-lr1e-4 "${TRAIN[@]}" "${FUNDED[@]}"
fi
if [[ -n ${SMOKE_MEMORY_LIMIT:-} ]]; then
  # Memory question only: two steps at a longer token limit. Eager attention holds batch x heads x length^2 per layer,
  # so the limit the cut-off rule asks for may not fit this card at eight generations. It runs last: a failure here costs nothing above.
  step "train-limit$SMOKE_MEMORY_LIMIT" --steps 2 --learning-rate 1e-4 "${SHAPE[@]}" --max-completion-length "$SMOKE_MEMORY_LIMIT" --eval-limit 2 --max-seconds "$SMOKE_MEMORY_SECONDS" --final-eval-seconds 450 "${FUNDED[@]}"
elif [[ ${SMOKE_SECOND_RUN:-1} == 1 ]]; then
# The shared remote runner bounds total process time; if time remains: the same at a tenth of the learning rate, to see whether 1e-4 is what breaks output format.
# 8 steps at 61 s, plus a before and an after evaluation of up to 4.6 minutes each: about 17 minutes measured, so twice that.
step train-lr1e-5 --steps 8 --learning-rate 1e-5 "${SHAPE[@]}" --max-seconds 2100 --final-eval-seconds 600 "${FUNDED[@]}"
fi
# ops/remote_job.sh records EXIT_CODE and DONE after this script exits.
