#!/usr/bin/env bash
# Drive an already-running RunPod pod over SSH: sync the repo, start the remote script, pull results
# until DONE or the deadline. Billing has a time cap that does not depend on this machine: the remote
# runner refuses to start unless the pod can remove itself, and removes it a grace period after the
# deadline or after the job ends (ops/remote_job.sh). This script also removes the pod when it exits,
# once the results are copied; KEEP_POD=1 skips that, for a relaunch on the same pod within the grace period.
# Pods for this project live in the RunPod space dedicated to it (docs/preflight-plan.md); do not point this at any other account's pod.
# Usage: drive_runpod.sh <ssh host> <ssh port> <remote script> <results dir> <deadline minutes>
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 5 ]] || { echo 'need host port remote-script results-dir deadline-minutes'; exit 2; }
HOST=$1 PORT=$2 REMOTE=$3 RESULTS=$4 DEADLINE=$5
[[ $REMOTE =~ ^ops/[a-zA-Z0-9_-]+\.sh$ && $DEADLINE =~ ^[1-9][0-9]*$ ]] || exit 2
if [[ $REMOTE == ops/remote_runpod_smoke.sh && ${JUDGE_KIND:-api} == api ]]; then
  [[ ${FORWARD_JUDGE_KEY:-} == 1 && -n ${OPENAI_API_KEY:-} ]] || {
    echo 'API smoke needs FORWARD_JUDGE_KEY=1 and OPENAI_API_KEY before starting'; exit 2;
  }
fi
[[ ${JUDGE_KIND:-api} == api || ${JUDGE_KIND:-api} == pod ]] || exit 2
[[ ${LEAN_ISOLATION:-required} == required || ${LEAN_ISOLATION:-} == none ]] || exit 2
[[ ${JUDGE_MAX_USD:-1} =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo 'JUDGE_MAX_USD must be a number of dollars'; exit 2; }
# SMOKE_* settings for the remote smoke script are passed through, and only if they are plain numbers.
SMOKE_ENV=''
while IFS='=' read -r name value; do
  [[ $name =~ ^SMOKE_[A-Z_]+$ && $value =~ ^[0-9]+$ ]] || { echo "invalid $name"; exit 2; }
  SMOKE_ENV+="$name=$value "
done < <(env | grep '^SMOKE_' || true)
mkdir -p "$RESULTS"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=.tools/known_hosts -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
pull() { rsync -az --partial --timeout=60 -e "${SSH[*]}" "root@$HOST:out/" "$RESULTS/" 2>/dev/null; }
finish() {
  echo "copying results"; pull || true
  "${SSH[@]}" "root@$HOST" 'rm -f ~/.judge_key' 2>/dev/null || true
  date -u +%FT%TZ > "$RESULTS/driver-exit.txt"
  if [[ ${KEEP_POD:-0} == 1 ]]; then echo "KEEP_POD=1: pod left running; its own timer removes it (out/autostop.log)"
  elif "${SSH[@]}" "root@$HOST" 'bash pilot-repo/ops/remote_job.sh --remove-now' 2>/dev/null; then echo "pod removal requested; confirm in the RunPod console that it is gone"
  else echo "REMINDER: could not remove the pod from here. Terminate it in the RunPod console now"; fi
}
trap finish EXIT
for _ in $(seq 20); do "${SSH[@]}" "root@$HOST" true 2>/dev/null && break; sleep 15; done
"${SSH[@]}" "root@$HOST" true || { echo 'ssh failed'; exit 1; }
# results/ stays behind except the model-read labels, which judge validation on the pod needs (7 MB).
rsync -az --exclude .tools --exclude spending --include 'results/' --include 'results/report-reading/***' --exclude 'results/*' -e "${SSH[*]}" ./ "root@$HOST:pilot-repo/"
# The API judge needs the OpenAI key on the pod. It is sent only on request, over the SSH channel into a mode-600
# file that the remote script reads once and deletes; it never appears in a command line or in the repo.
if [[ ${FORWARD_JUDGE_KEY:-} == 1 && $REMOTE == ops/remote_runpod_smoke.sh && ${JUDGE_KIND:-api} == api ]]; then
  printf '%s' "${OPENAI_API_KEY:?FORWARD_JUDGE_KEY=1 needs OPENAI_API_KEY}" | "${SSH[@]}" "root@$HOST" 'umask 077; cat > ~/.judge_key'
fi
# A fresh out/ every run: a reused pod must not hand back the previous run's summaries. Models live in /workspace.
# The remote log goes inside it so the runner's own errors are pulled with the results.
"${SSH[@]}" "root@$HOST" "rm -rf out; mkdir -p out; JUDGE_KIND=${JUDGE_KIND:-api} LEAN_ISOLATION=${LEAN_ISOLATION:-required} JUDGE_MAX_USD=${JUDGE_MAX_USD:-1} REQUIRE_AUTOSTOP=1 AUTOSTOP_GRACE_SECONDS=${AUTOSTOP_GRACE_SECONDS:-600} ${SMOKE_ENV}nohup bash pilot-repo/ops/remote_job.sh pilot-repo/$REMOTE $((DEADLINE*60)) > out/remote.log 2>&1 < /dev/null &"
date -u +%FT%TZ > "$RESULTS/started-at.txt"
END=$((SECONDS+DEADLINE*60))
while ((SECONDS < END)); do
  sleep 60
  if code=$("${SSH[@]}" "root@$HOST" 'cat out/EXIT_CODE' 2>/dev/null); then
    [[ $code =~ ^[0-9]+$ && $code -le 255 ]] || { echo 'invalid remote exit status'; exit 1; }
    echo "remote finished with exit=$code"
    exit "$code"
  fi
  if ((SECONDS % 300 < 60)); then pull || true; fi
done
echo 'driver deadline reached; the remote timeout and the pod removal timer are armed'
exit 1
