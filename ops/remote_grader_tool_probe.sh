#!/usr/bin/env bash
# deadline: 60
# Grader-tool smoke test (entrypoints/grader_tool_probe.py): untrained gpt-oss-20b episodes with the read-only run_grader
# tool, sampled from the vLLM rollout server exactly as a training run's evaluations are (the trainer's environment, the
# server's context and memory share), with model-written Lean behind the tool and the verifier. No trainer, no judge.
# Stages in series, each with its own limit: Lean 7 min, training install 5 min, vLLM install beside the 11 min download,
# server start 15 min, episodes 15 min: about 53 minutes at worst; typically about 20.
set -euo pipefail
cd ~/pilot-repo
source ops/pod_lib.sh
configure_model_environment
mkdir -p ~/out "$HF_HOME"
rm -f ~/out/DONE
log() { echo "$(date -u +%FT%TZ) $*" >> ~/out/steps.log; }
fail() { log "$1"; exit 1; }
SERVER_PID=
VLLM_INSTALL_PID=
cleanup() { if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID"; then stop_server "$SERVER_PID"; fi; }
trap cleanup EXIT
M=0  # gpt-oss-20b's column in ops/pod_pins.env
MODEL=${MODEL_REPOS[$M]} REV=${MODEL_REVS[$M]}
LIMIT=${SMOKE_TOKEN_LIMIT:?the answer budget per episode, as a training run sets it}
RUNS=${SMOKE_GRADER_RUNS:?how many drafts the tool may grade, as grader_runs in an experiment}
[[ $LIMIT =~ ^[1-9][0-9]{0,5}$ && $RUNS =~ ^[1-8]$ ]] || fail 'SMOKE_TOKEN_LIMIT must be a positive integer and SMOKE_GRADER_RUNS 1 to 8'
pod_self_test 1 "${MODEL_WORKSPACE_GB[$M]}" 2 "$ROLLOUT_PORT"
choose_lean_isolation
install_lean
log start install
install_uv
INSTALL_WAIT=600 make_env vllm -r requirements-vllm.txt > ~/out/install-vllm.log 2>&1 &  # cap: 2x measured 222 s
VLLM_INSTALL_PID=$!
INSTALL_WAIT=300 make_env train torch -r requirements.txt > ~/out/install-train.log 2>&1 || fail TRAIN_INSTALL_FAILED  # cap: 2x measured 81 s
HF=$(env_bin train hf)
PY=$(env_bin train python)
log "start download $MODEL"
DOWNLOAD_WAIT=${MODEL_DOWNLOAD_LIMITS[$M]} fetch_model "$MODEL" "$REV" || fail "DOWNLOAD_FAILED $MODEL"
wait "$VLLM_INSTALL_PID" || fail VLLM_INSTALL_FAILED
# The context a training run's server would have: the prompt room, the answer budget and the tool results' reserve.
RESULTS=$("$PY" -c 'import sys; from rewardhack.grader_tool import result_reserve; print(result_reserve(int(sys.argv[1])))' "$RUNS") \
  || fail RESULT_RESERVE_UNREADABLE
log "start rollout-server (answer budget $LIMIT, tool results $RESULTS)"
serve_rollouts vllm "$M" "$LIMIT" "$DEFAULT_TRAIN_GENERATIONS" "$RESULTS" > ~/out/rollout-server.log 2>&1 &
SERVER_PID=$!
wait_for_server "$SERVER_PID" "$ROLLOUT_PORT" "$MODEL@$REV" "$SERVER_WAIT" || fail ROLLOUT_SERVER_DID_NOT_START
refuse_ignored_settings ~/out/rollout-server.log
log start episodes
# cap: 1.5x estimated 600 s (8 episodes of up to 5 turns sharing a 16k budget each, and their Lean runs)
timeout --kill-after="$KILL_AFTER_SECONDS" 900 "$PY" entrypoints/grader_tool_probe.py --out ~/out/probe --lean ~/lean-safe \
  --model "$MODEL" --revision "$REV" --rollout-url "http://127.0.0.1:$ROLLOUT_PORT" --max-completion-length "$LIMIT" \
  --grader-runs "$RUNS" > ~/out/probe.log 2>&1 || fail PROBE_FAILED
log completed
touch ~/out/DONE
