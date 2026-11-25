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
  # Retried until RunPod accepts it: one API error must not leave the pod billing.
  nohup setsid bash -c "sleep $1; until runpodctl remove pod '$POD_ID'; do sleep 30; done" >> "$OUT/autostop.log" 2>&1 < /dev/null &
}
arm() {  # $1: seconds until removal. Replaces earlier timers, then proves the new one is running before saying so.
  pkill -f 'runpodctl remove pod'; [[ $? -le 1 ]] || { echo 'could not replace earlier removal timers' | tee -a "$OUT/autostop.log"; exit 3; }  # 1: none were running
  remove_after "$1"
  pgrep -f 'runpodctl remove pod' > /dev/null || { echo 'NO_AUTOSTOP: the removal timer did not start' | tee -a "$OUT/autostop.log"; exit 3; }
  echo "$(date -u +%FT%TZ) armed: pod $POD_ID is removed at most ${1}s from now" >> "$OUT/autostop.log"
}
can_remove() { [[ -n $POD_ID ]] && command -v runpodctl >/dev/null && runpodctl get pod "$POD_ID" > "$OUT/autostop.log" 2>&1; }
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
if [[ ${1:-} == --arm-only ]]; then  # the driver's first act on a new pod: a cap for the minutes before the job starts
  [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || { echo '--arm-only needs a number of seconds'; exit 2; }
  can_remove || { echo 'NO_AUTOSTOP: this pod cannot remove itself'; exit 3; }
  arm "$2"
  exit 0
fi
trap finish EXIT
[[ $# == 2 && $2 =~ ^[1-9][0-9]*$ && $GRACE =~ ^[0-9]+$ ]] || { echo 'need a job script and a whole number of seconds; AUTOSTOP_GRACE_SECONDS must be a number'; exit 2; }
command -v timeout >/dev/null || { echo 'GNU timeout is required'; exit 2; }
command -v setsid >/dev/null || { echo 'setsid is required'; exit 2; }
# A relaunch must not share the GPU with an earlier job that is still running, and that job's exit must not write this
# run's EXIT_CODE or arm a removal in the middle of it. Its runner goes first, without its exit trap, then its whole process group.
PIDS=${REMOTE_JOB_PIDS:-$HOME/.remote_job.pids}
if [[ -s $PIDS ]]; then
  read -r OLD_RUNNER OLD_GROUP < "$PIDS"
  if kill -0 "$OLD_RUNNER"; then echo "stopping earlier runner $OLD_RUNNER"; kill -9 "$OLD_RUNNER"; fi
  if kill -0 -- "-$OLD_GROUP"; then echo "stopping earlier job group $OLD_GROUP"; kill -9 -- "-$OLD_GROUP"; fi
fi
if can_remove; then
  AUTOSTOP=1
  # A relaunch on the same pod replaces the earlier job's timers; otherwise the old one would remove the pod mid-run.
  arm "$(($2+30+GRACE))"
elif [[ ${REQUIRE_AUTOSTOP:-0} == 1 ]]; then
  echo 'NO_AUTOSTOP: this pod cannot remove itself (no pod id, no runpodctl, or its key was refused); not starting' | tee -a "$OUT/autostop.log"
  exit 3
fi
# Its own session, so everything the job starts (servers, their workers) is one process group a relaunch can stop.
setsid timeout --signal=TERM --kill-after=30 "$2" bash "$1" &
JOB=$!
printf '%s %s\n' "$$" "$JOB" > "$PIDS"
wait "$JOB"
