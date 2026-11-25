#!/usr/bin/env bash
# Shared remote runner: bounds process runtime and atomically records the real exit code.
# On a RunPod pod it also arms a billing time cap: the pod removes itself with RunPod's own CLI a grace
# period after the job's deadline, and again a grace period after the job ends, whichever comes first.
# Neither needs the laptop or the console. With REQUIRE_AUTOSTOP=1 the job refuses to start unless that
# cap could be armed. Elsewhere (AWS has an OS shutdown timer) this stops the job, not the billing.
set -uo pipefail
OUT=${REMOTE_JOB_OUT:-$HOME/out}
GRACE=${AUTOSTOP_GRACE_SECONDS:-600}  # time for the driver to copy results before the pod is removed
mkdir -p "$OUT"
# An SSH session does not inherit the container's environment, where RunPod puts the pod's id and its own key.
pod_env() { [[ -r /proc/1/environ ]] && tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -1; }
POD_ID=${RUNPOD_POD_ID:-$(pod_env RUNPOD_POD_ID)}
export RUNPOD_API_KEY=${RUNPOD_API_KEY:-$(pod_env RUNPOD_API_KEY)}
AUTOSTOP=0
remove_after() {  # detached, so it outlives this script, the SSH session and a killed job
  nohup setsid bash -c "sleep $1; runpodctl remove pod '$POD_ID'" >> "$OUT/autostop.log" 2>&1 < /dev/null &
}
finish() {
  code=$?
  rm -f "$HOME/.judge_key"  # never outlives the job, whichever remote script ran and however it ended
  printf '%s\n' "$code" > "$OUT/EXIT_CODE.tmp"
  mv "$OUT/EXIT_CODE.tmp" "$OUT/EXIT_CODE"
  touch "$OUT/DONE"
  if ((AUTOSTOP)); then echo "$(date -u +%FT%TZ) job ended with $code; removing pod in ${GRACE}s" >> "$OUT/autostop.log"; remove_after "$GRACE"; fi
}
if [[ ${1:-} == --remove-now ]]; then  # the driver's last act once the results are copied
  [[ -n $POD_ID ]] && command -v runpodctl >/dev/null || { echo 'cannot remove this pod from inside it'; exit 3; }
  exec runpodctl remove pod "$POD_ID"
fi
trap finish EXIT
[[ $# == 2 && $2 =~ ^[1-9][0-9]*$ && $GRACE =~ ^[0-9]+$ ]] || exit 2
command -v timeout >/dev/null || { echo 'GNU timeout is required'; exit 2; }
if [[ -n $POD_ID ]] && command -v runpodctl >/dev/null && runpodctl get pod "$POD_ID" > "$OUT/autostop.log" 2>&1; then
  AUTOSTOP=1
  # A relaunch on the same pod replaces the earlier job's timers; otherwise the old one would remove the pod mid-run.
  pkill -f 'runpodctl remove pod'; [[ $? -le 1 ]] || { echo 'could not replace earlier removal timers' | tee -a "$OUT/autostop.log"; exit 3; }  # 1: none were running
  echo "$(date -u +%FT%TZ) armed: pod $POD_ID is removed at most $(($2+30+GRACE))s from now" >> "$OUT/autostop.log"
  remove_after "$(($2+30+GRACE))"
elif [[ ${REQUIRE_AUTOSTOP:-0} == 1 ]]; then
  echo 'NO_AUTOSTOP: this pod cannot remove itself (no pod id, no runpodctl, or its key was refused); not starting' | tee -a "$OUT/autostop.log"
  exit 3
fi
timeout --signal=TERM --kill-after=30 "$2" bash "$1"
