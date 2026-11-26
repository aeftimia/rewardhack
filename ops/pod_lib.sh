#!/usr/bin/env bash
# Shared by the pod job scripts (sourced, not run). Each function exists because its absence cost a paid run.
# Callers define `log` and `fail`.
set -euo pipefail
source ops/pod_pins.env  # models, revisions, versions and ports: stated once there
# The rollout server and a judge served on the pod have their own ports (pod_pins.env), so the two are never confused.
# 15 minutes for a vLLM server to answer. Its first start took 6.5 minutes (390 s) on this pod type with its own weight fetch.
# shellcheck disable=SC2034  # read by the job scripts that source this file
# cap: 2x measured 390 s
SERVER_WAIT=900

# Wait until the card is free: a process that just ended (a relaunch's predecessor, a stopped server, a finished stage)
# gives its memory back within seconds. Usage: wait_card_free <who is waiting, for the failure message>.
# The limit, CARD_FREE_SECONDS, is in ops/pod_pins.env, where the driver also counts it: the whole wait, each query's own
# limit (NVIDIA_SMI_SECONDS) included, ends within it.
# Every nvidia-smi query goes through here: it can hang on a wedged card.
smi() { timeout "$NVIDIA_SMI_SECONDS" nvidia-smi "$@"; }
# The card's memory in use, in MiB. A failed query, or an empty or non-numeric reading, is a failure with its reason in
# the log (the callers fail on it), not a free card.
card_used_mib() {
  local used
  used=$(smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) || { log "nvidia-smi failed or did not answer in $NVIDIA_SMI_SECONDS s"; return 1; }
  [[ $used =~ ^[0-9]+$ ]] || { log "nvidia-smi read the card's memory as '$used'"; return 1; }
  echo "$used"
}
wait_card_free() {
  local used end=$((SECONDS + CARD_FREE_SECONDS))
  used=$(card_used_mib) || fail NVIDIA_SMI_FAILED
  while ((used >= 1024 && SECONDS + 2 + NVIDIA_SMI_SECONDS <= end)); do
    sleep 2
    used=$(card_used_mib) || fail NVIDIA_SMI_FAILED
  done
  ((used < 1024)) || fail "CARD_NOT_FREED: $used MiB still in use after $CARD_FREE_SECONDS s; waiting: $1"
}

# Every Python environment a job makes is /workspace/<name>-venv. Usage: env_bin <name> <command>.
env_bin() { echo "/workspace/$1-venv/bin/$2"; }

# Create (or, on a relaunch, recreate: `uv venv` refuses an existing one without --clear) an environment, install into
# it and record what was installed in ~/out/pip-freeze-<name>.txt. Usage: INSTALL_WAIT=<s> make_env <name> <uv pip args>,
# with the wait's basis in a cap comment beside the call. Needs UV and UV_CACHE_DIR from the caller.
make_env() {
  local name=$1; shift
  "$UV" venv -q --clear --python "$(command -v python)" "/workspace/$name-venv" \
    && timeout "${INSTALL_WAIT:?the caller sets INSTALL_WAIT}" "$UV" pip install -q --python "$(env_bin "$name" python)" "$@" \
    && "$UV" pip freeze --python "$(env_bin "$name" python)" > ~/out/pip-freeze-"$name".txt
}

# Run a command of an environment with that environment's bin first on PATH. vLLM's FlashInfer builds kernels with the
# environment's ninja: the 5 Nov probes failed without it, and the 26 Nov gpt-oss-120b stage was written without it
# (review round 25), because the prefix was copied at each launch instead of stated here.
# Usage: in_env <name> [timeout <seconds>] <command> [args...]; the command must be the environment's own. A timeout sends
# TERM at <seconds> and KILL KILL_AFTER_SECONDS later (ops/pod_pins.env): a process stuck in a CUDA call ignores TERM.
in_env() {
  local bin wrap=()
  bin=$(env_bin "$1" ''); shift
  if [[ $1 == timeout ]]; then wrap=(timeout --kill-after="$KILL_AFTER_SECONDS" "$2"); shift 2; fi
  [[ -x $bin$1 ]] || fail "NO_COMMAND: $1 is not in $bin"
  PATH="$bin:$PATH" ${wrap[@]+"${wrap[@]}"} "$bin$1" "${@:2}"  # the + form: bash 3.2 calls an empty array unbound
}

# Fetch a model's weights at a pinned revision into HF_HOME, without the copies vLLM and transformers do not read.
# Usage: DOWNLOAD_WAIT=<s> fetch_model <repo> <revision>, with a cap comment beside the call; the log is
# ~/out/download-<repo name>.log. Needs HF (an `hf` command) from the caller.
fetch_model() {
  timeout "${DOWNLOAD_WAIT:?the caller sets DOWNLOAD_WAIT}" "$HF" download "$1" --revision "$2" --exclude 'original/*' --exclude 'metal/*' \
    > ~/out/download-"${1##*/}".log 2>&1
}

# Seconds-long checks of what the job assumes about the machine, before any install or download is paid for.
# Usage: pod_self_test <cards needed> <free GB needed under /workspace> <free GB needed for ~/out> <port that must be free>...
pod_self_test() {
  local gpus=$1 need_gb=$2 out_need_gb=$3 tool port free_gb out_free_gb cards
  shift 3
  for tool in runpodctl timeout setsid pkill pgrep curl tar zstd nvidia-smi python; do
    command -v "$tool" > /dev/null || fail "SELF_TEST: $tool is not on this pod"
  done
  python -c 'import sys, venv, ensurepip; assert sys.version_info >= (3, 12), sys.version' || fail 'SELF_TEST: python 3.12 with venv and ensurepip is required'
  for port in "$@"; do
    # 2026-11-25: the image's nginx owns 8001 and answered a readiness check meant for our server. Binding the port is the
    # test: asking it a question would pass for a listener that is slow or does not speak HTTP.
    # SO_REUSEADDR: a server killed a moment ago leaves closing connections that would refuse the bind; a live listener still does.
    python -c 'import socket, sys; s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1])))' "$port" \
      || fail "SELF_TEST: port $port is already in use"
  done
  # A process killed a moment ago (a relaunch) can take some seconds to give the card back.
  wait_card_free 'SELF_TEST (an earlier job'"'"'s process?)'
  # A two-card model on a one-card pod would fail only at the rollout server's start, after the download.
  cards=$(smi --query-gpu=name --format=csv,noheader | wc -l) || fail NVIDIA_SMI_FAILED
  ((cards >= gpus)) || fail "SELF_TEST: this pod has $cards card(s), the job's model needs $gpus"
  free_gb=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  [[ $free_gb -ge $need_gb ]] || fail "SELF_TEST: /workspace has ${free_gb} GB free, the job needs $need_gb"
  # Results go to ~/out: on the container disk, or with SMOKE_VOLUME=1 through a link to the network volume (so the disk is
  # read at "$HOME/out/", which follows the link), and a relaunch keeps the earlier run beside it (out.prev.*). With expert
  # LoRA each Part 1 run writes about 20 GB there (adapter snapshots about 1 GB each, checkpoints about 3 GB): a full disk
  # fails a save late in a run, so the room is checked before anything is paid for beyond setup.
  # On one disk the two needs add up.
  if [[ $(df --output=target /workspace | tail -1) == $(df --output=target "$HOME/out/" | tail -1) ]]; then
    ((free_gb >= need_gb + out_need_gb)) || fail "SELF_TEST: /workspace and ~/out share a disk with ${free_gb} GB free, the job needs $need_gb + $out_need_gb"
  else
    out_free_gb=$(df --output=avail -BG "$HOME/out/" | tail -1 | tr -dc 0-9)
    [[ $out_free_gb -ge $out_need_gb ]] || fail "SELF_TEST: the disk holding ~/out has ${out_free_gb} GB free, the job needs $out_need_gb"
  fi
  log "self-test passed: $(smi --query-gpu=name,memory.total --format=csv,noheader), ${free_gb} GB free"
}

# Wait for an OpenAI-compatible server we started. Ready means: our process is alive AND the port's model list names our
# model. Usage: wait_for_server <pid> <port> <model> <seconds>; returns 1 if it died or did not answer in that time.
# The time is elapsed time by the shell's clock, not a count of tries: a try is a 5 s sleep plus a request of up to 5 s, so
# "180 tries of 5 s" was up to 30 minutes where its callers said 15. The last check starts before the time is up, so the
# function can return up to 10 s after it.
wait_for_server() {
  local pid=$1 port=$2 model=$3 seconds=$4 end listed
  [[ $seconds =~ ^[1-9][0-9]*$ ]] || fail "wait_for_server: the wait must be a whole number of seconds, got '$seconds'"
  end=$((SECONDS+seconds))
  while ((SECONDS < end)); do
    kill -0 "$pid" || return 1
    # The reply is captured and then searched. Piped, `grep -q` exits at the first match, curl can then fail on its closed
    # pipe, and under pipefail the pipeline reports that failure for a server that had answered correctly.
    if listed=$(curl -sf --max-time 5 "http://127.0.0.1:$port/v1/models") && grep -q "$model" <<< "$listed"; then return 0; fi
    sleep 5
  done
  return 1
}

# vLLM ignores SIGTERM while it is fetching weights, and its engine is a child process that holds the card's memory. A
# server started through in_env is a grandchild of the job (the backgrounded function's shell is its parent), so the
# whole tree under the given process goes, deepest first.
stop_server() {
  local child
  for child in $(pgrep -P "$1"); do stop_server "$child"; done
  # A parent can exit by itself once its child is killed; it is then already gone, which is what this wanted.
  kill -9 "$1" || [[ ! -e /proc/$1 ]]
}

# A setting the library does not know is ignored with one warning line. That is a silent fallback: stop on it.
refuse_ignored_settings() {
  if grep -i 'unknown vllm environment variable' "$1"; then fail "IGNORED_SETTING: vLLM does not know a variable this script sets (see $1)"; fi
}
