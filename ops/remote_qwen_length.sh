#!/usr/bin/env bash
# deadline: 125
# Inference only: same pinned base model and evaluation draws as the 8k run.
set -euo pipefail
cd ~/pilot-repo
source ops/pod_lib.sh
configure_model_environment
mkdir -p ~/out "$HF_HOME"
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; exit 1; }
SERVER_PID=
cleanup() { if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID"; then stop_server "$SERVER_PID"; fi; }
trap cleanup EXIT
pod_self_test 1 95 2 "$ROLLOUT_PORT"
log install
install_uv
grep '^numpy==' requirements-vllm.txt > /workspace/probe-override.txt
# cap: 2x measured 222 s
INSTALL_WAIT=600 make_env probe -r requirements-vllm.txt -e . --override /workspace/probe-override.txt > ~/out/install-probe.log 2>&1
HF=$(env_bin probe hf)
log download
# cap: 2x measured 232 s (latest Qwen download, 27 Nov)
DOWNLOAD_WAIT=480 fetch_model "$QWEN3_32B" "$QWEN3_32B_REV"
LIMIT=${SMOKE_TOKEN_LIMIT:?}
[[ $LIMIT == 16384 ]] || fail EXPECTED_16384_TOKEN_DIAGNOSTIC
log server
serve_rollouts probe 1 "$LIMIT" "$DEFAULT_TRAIN_GENERATIONS" > ~/out/rollout-server.log 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_PID" "$ROLLOUT_PORT" "$QWEN3_32B@$QWEN3_32B_REV" "$SERVER_WAIT"
refuse_ignored_settings ~/out/rollout-server.log
log inference
# cap: 1.5x estimated 3600 s (24x16k generation, allowing longer-context slowdown)
in_env probe timeout 5400 python entrypoints/qwen_length_measure.py --out ~/out/inference \
  --model "$QWEN3_32B" --revision "$QWEN3_32B_REV" --rollout-url "http://127.0.0.1:$ROLLOUT_PORT" \
  --max-completion-length "$LIMIT" --seed "$PROBE_SEED" > ~/out/inference.log 2>&1
log completed
touch ~/out/DONE
