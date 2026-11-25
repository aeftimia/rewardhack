#!/usr/bin/env bash
# Shared by the pod job scripts (sourced, not run). Each function exists because its absence cost a paid run.
# Callers define `log` and `fail`.
set -euo pipefail

# Seconds-long checks of what the job assumes about the machine, before any install or download is paid for.
# Usage: pod_self_test <free GB needed under /workspace> <port that must be free>...
pod_self_test() {
  local need_gb=$1 tool port free_gb used_mib
  shift
  for tool in runpodctl timeout setsid pkill pgrep curl tar zstd nvidia-smi python; do
    command -v "$tool" > /dev/null || fail "SELF_TEST: $tool is not on this pod"
  done
  python -c 'import sys, venv, ensurepip; assert sys.version_info >= (3, 12), sys.version' || fail 'SELF_TEST: python 3.12 with venv and ensurepip is required'
  for port in "$@"; do
    # 2026-11-25: the image's nginx owns 8001 and answered a readiness check meant for our server. Binding the port is the
    # test: asking it a question would pass for a listener that is slow or does not speak HTTP.
    python -c 'import socket, sys; socket.socket().bind(("127.0.0.1", int(sys.argv[1])))' "$port" || fail "SELF_TEST: port $port is already in use"
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
  log "self-test passed: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), ${free_gb} GB free"
}

# Wait for an OpenAI-compatible server we started. Ready means: our process is alive AND the port's model list names our
# model. Usage: wait_for_server <pid> <port> <model> <tries of 5 s>; returns 1 if it died or never answered.
wait_for_server() {
  local pid=$1 port=$2 model=$3 tries=$4
  for _ in $(seq "$tries"); do
    kill -0 "$pid" || return 1
    if curl -sf --max-time 5 "http://127.0.0.1:$port/v1/models" | grep -q "$model"; then return 0; fi
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
