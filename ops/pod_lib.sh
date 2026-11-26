#!/usr/bin/env bash
# Shared by the pod job scripts (sourced, not run). Each function exists because its absence cost a paid run.
# Callers define `log` and `fail`.
set -euo pipefail
source ops/pod_pins.env  # models, revisions, versions and ports: stated once there
# The rollout server and a judge served on the pod have their own ports (pod_pins.env), so the two are never confused.
# 15 minutes for a vLLM server to answer. Its first start took 6.5 minutes (390 s) on this pod type with its own weight fetch.
# cap: 2x measured 390 s
SERVER_WAIT=900

# Seconds-long checks of what the job assumes about the machine, before any install or download is paid for.
# Usage: pod_self_test <free GB needed under /workspace> <free GB needed for ~/out> <port that must be free>...
pod_self_test() {
  local need_gb=$1 out_need_gb=$2 tool port free_gb out_free_gb used_mib
  shift 2
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
  for _ in $(seq 30); do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ $used_mib -ge 1024 ]] || break
    sleep 1
  done
  [[ $used_mib -lt 1024 ]] || fail "SELF_TEST: the card still has ${used_mib} MiB in use after 30 s (an earlier job's process?)"
  free_gb=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  [[ $free_gb -ge $need_gb ]] || fail "SELF_TEST: /workspace has ${free_gb} GB free, the job needs $need_gb"
  # Results go to ~/out, on the container disk, and a relaunch keeps the earlier run beside it (out.prev.*). With expert
  # LoRA a Part 1 pair writes about 40 GB there (adapter snapshots about 1 GB each, checkpoints about 3 GB): a full disk
  # fails a save late in a run, so the room is checked before anything is paid for beyond setup.
  # On one disk the two needs add up.
  if [[ $(df --output=target /workspace | tail -1) == $(df --output=target "$HOME" | tail -1) ]]; then
    ((free_gb >= need_gb + out_need_gb)) || fail "SELF_TEST: /workspace and ~/out share a disk with ${free_gb} GB free, the job needs $need_gb + $out_need_gb"
  else
    out_free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
    [[ $out_free_gb -ge $out_need_gb ]] || fail "SELF_TEST: the disk holding ~/out has ${out_free_gb} GB free, the job needs $out_need_gb"
  fi
  log "self-test passed: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), ${free_gb} GB free"
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

# vLLM ignores SIGTERM while it is fetching weights, and its engine is a child process that holds the card's memory.
stop_server() {
  if ! pkill -9 -P "$1"; then log "server $1 had no child processes to stop"; fi
  kill -9 "$1"
}

# A setting the library does not know is ignored with one warning line. That is a silent fallback: stop on it.
refuse_ignored_settings() {
  if grep -i 'unknown vllm environment variable' "$1"; then fail "IGNORED_SETTING: vLLM does not know a variable this script sets (see $1)"; fi
}
