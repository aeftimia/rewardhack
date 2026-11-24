#!/usr/bin/env bash
# Drive an already-running RunPod pod over SSH: sync the repo, start the remote script, pull results
# until DONE or the deadline. The pod has no OS shutdown timer, so termination is done in the console
# by whoever launched it; this script prints a loud reminder on every exit.
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
mkdir -p "$RESULTS"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=.tools/known_hosts -o ConnectTimeout=10)
pull() { rsync -az --partial -e "${SSH[*]}" "root@$HOST:out/" "$RESULTS/" 2>/dev/null; }
finish() { echo "copying results"; pull || true; date -u +%FT%TZ > "$RESULTS/driver-exit.txt"; echo "REMINDER: terminate the pod in the RunPod console now"; }
trap finish EXIT
for _ in $(seq 20); do "${SSH[@]}" "root@$HOST" true 2>/dev/null && break; sleep 15; done
"${SSH[@]}" "root@$HOST" true || { echo 'ssh failed'; exit 1; }
# results/ stays behind except the model-read labels, which judge validation on the pod needs (7 MB).
rsync -az --exclude .tools --exclude spending --include 'results/' --include 'results/report-reading/***' --exclude 'results/*' -e "${SSH[*]}" ./ "root@$HOST:pilot-repo/"
# The API judge needs the OpenAI key on the pod. It is sent only on request, over the SSH channel into a mode-600
# file that the remote script reads once and deletes; it never appears in a command line or in the repo.
if [[ ${FORWARD_JUDGE_KEY:-} == 1 ]]; then
  printf '%s' "${OPENAI_API_KEY:?FORWARD_JUDGE_KEY=1 needs OPENAI_API_KEY}" | "${SSH[@]}" "root@$HOST" 'umask 077; cat > ~/.judge_key'
fi
"${SSH[@]}" "root@$HOST" "mkdir -p out; rm -f out/DONE out/EXIT_CODE out/EXIT_CODE.tmp; JUDGE_KIND=${JUDGE_KIND:-api} LEAN_ISOLATION=${LEAN_ISOLATION:-required} nohup bash pilot-repo/ops/remote_job.sh pilot-repo/$REMOTE $((DEADLINE*60)) > remote.log 2>&1 < /dev/null &"
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
echo 'driver deadline reached; remote process timeout is armed, but terminate the pod now to stop billing'
exit 1
