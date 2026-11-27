#!/usr/bin/env bash
# Shared remote runner: bounds process runtime and atomically records the real exit code.
# On a RunPod pod it also arms a billing time cap: the pod removes itself with RunPod's own CLI a grace
# period after the job's deadline, and again a grace period after the job ends, whichever comes first.
# Neither needs the laptop or the console. With REQUIRE_AUTOSTOP=1 the job refuses to start unless that
# cap could be armed. Elsewhere this stops the job, not the billing.
set -euo pipefail
# The wait between TERM and KILL is ops/pod_pins.env's KILL_AFTER_SECONDS, written out here because the driver also sends
# this script over SSH on stdin (--remove-now, --arm-only), where there is no file beside it to source;
# tests/test_shared_constants.py keeps the two equal.
KILL_AFTER_SECONDS=30
OUT=${REMOTE_JOB_OUT:-$HOME/out}
GRACE=${AUTOSTOP_GRACE_SECONDS:-600}  # time for the driver to copy results before the pod is removed
mkdir -p "$OUT" || { echo "cannot create runner output directory: $OUT" >&2; exit 3; }
# An SSH session does not inherit the container's environment, where RunPod puts the pod's id and its own key.
pod_env() { if [[ -r /proc/1/environ ]]; then tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -1; fi; }
POD_ID=${RUNPOD_POD_ID:-$(pod_env RUNPOD_POD_ID)}
export RUNPOD_API_KEY=${RUNPOD_API_KEY:-$(pod_env RUNPOD_API_KEY)}
AUTOSTOP=0
remove_after() {  # detached, so it outlives this script, the SSH session and a killed job
  # Asked again every 30 s for as long as this pod exists. It does not rely on what runpodctl's exit status means, which has
  # never been seen for a refusal: the loop ends when the pod, and this process with it, is gone.
  nohup setsid bash -c "sleep $1; while true; do runpodctl remove pod '$POD_ID'; sleep 30; done" >> "$OUT/autostop.log" 2>&1 < /dev/null &
}
# Whether process $1 is running this pod's removal loop itself: anchored at the start, since nohup and setsid carry the
# loop's text in their own arguments until they exec into it. Captured first: a pipe into `grep -q` could end pgrep early.
timer_runs() {
  local running
  running=$(pgrep -f "^bash -c sleep [0-9]+; while true; do runpodctl remove pod '$POD_ID'") || [[ $? == 1 ]]  # pgrep's 1 means no matches; other errors propagate
  [[ $'\n'$running$'\n' == *$'\n'"$1"$'\n'* ]]
}
arm() {  # $1: seconds until removal. The new timer is proven alive first; only then are the earlier ones stopped.
  local earlier new pid
  # The bracket keeps the pattern from matching any command line that merely quotes it (a shell running this check); the timer's own
  # command line, `bash -c "sleep N; while true; do runpodctl remove pod ..."`, still matches. Only this pod's timers: a pod has
  # no other, and tests that run in parallel on one machine (each with its own pod id) cannot stop each other's.
  earlier=$(pgrep -f "[r]unpodctl remove pod '$POD_ID'") || [[ $? == 1 ]]  # pgrep's 1 means no matches; other errors propagate
  remove_after "$1"
  new=$!
  # Proven by what its process runs, not by its being alive at one instant: nohup and setsid exec into the timer's loop in
  # the same process, so its command line becomes the loop's. A one-off `kill -0` after a second passed a timer that was
  # still starting on a loaded machine and failed afterwards (the 26 Nov flaky test, review round 33).
  # cap: 5x estimated 1 s (nohup, setsid and bash exec'ing into the loop; slower when the machine is busy)
  for _ in $(seq 50); do
    if timer_runs "$new" || ! kill -0 "$new"; then break; fi
    sleep 0.1
  done
  timer_runs "$new" || { echo 'NO_AUTOSTOP: the removal timer did not start; earlier timers were left running' | tee -a "$OUT/autostop.log"; exit 3; }
  for pid in $earlier; do
    if kill -0 "$pid"; then kill "$pid"; fi
  done
  echo "$(date -u +%FT%TZ) armed: pod $POD_ID is removed at most ${1}s from now (timer pid $new)" >> "$OUT/autostop.log"
}
can_remove() { [[ -n $POD_ID ]] && command -v runpodctl >/dev/null && runpodctl get pod "$POD_ID" >> "$OUT/autostop.log" 2>&1; }
finish() {
  code=$?
  trap - EXIT
  # Arm removal before cleanup: even an error writing the status must not leave a pod billing.
  if ((AUTOSTOP)); then echo "$(date -u +%FT%TZ) job ended with $code; removing pod in ${GRACE}s" >> "$OUT/autostop.log"; remove_after "$GRACE"; fi
  # Whatever ended the runner, the job must not outlive it: the driver would report this exit code, copy partial results
  # and remove the pod while the job still ran.
  if [[ -n ${JOB:-} ]] && kill -0 -- "-$JOB"; then kill -9 -- "-$JOB"; fi
  # By session as well: a `timeout` inside the job puts its command in a process group of its own, which the line above misses.
  if [[ -n ${JOB:-} ]]; then pkill -9 -s "$JOB" || [[ $? == 1 ]]; fi  # pkill's 1 means the session is already gone
  rm -f "$HOME/.judge_key"  # never outlives the job, whichever remote script ran and however it ended
  # The pids file names a running job for a relaunch to stop; once this job has ended it names nothing (review round 30).
  # Only this runner's own line: a relaunch that already replaced it kills this runner with KILL, which runs no trap.
  if [[ -n ${PIDS:-} && -s $PIDS ]] && read -r runner _ < "$PIDS" && [[ $runner == "$$" ]]; then rm -f "$PIDS"; fi
  printf '%s\n' "$code" > "$OUT/EXIT_CODE.tmp"
  mv "$OUT/EXIT_CODE.tmp" "$OUT/EXIT_CODE"
  touch "$OUT/DONE"
  exit "$code"
}
if [[ ${1:-} == --remove-now ]]; then  # the driver's last act once the results are copied
  # shellcheck disable=SC2015  # refuse when either the pod id or runpodctl is missing
  [[ -n $POD_ID ]] && command -v runpodctl >/dev/null || { echo 'cannot remove this pod from inside it'; exit 3; }
  # The same proven detached timer retries even when this SSH session ends.
  arm 1
  echo "removal armed for pod $POD_ID; confirm it disappears in the console"
  exit 0
fi
if [[ ${1:-} == --arm-only ]]; then  # the driver's first act on a new pod: a cap for the minutes before the job starts
  [[ ${2:-} =~ ^[1-9][0-9]*$ ]] || { echo '--arm-only needs a number of seconds'; exit 2; }
  can_remove || { echo 'NO_AUTOSTOP: this pod cannot remove itself'; exit 3; }
  arm "$2"
  exit 0
fi
trap finish EXIT
trap 'exit 143' TERM  # nohup covers only a hang-up; a TERM must end through `finish`, not skip it
[[ $# == 2 && $2 =~ ^[1-9][0-9]*$ && $GRACE =~ ^[0-9]+$ ]] || { echo 'need a job script and a whole number of seconds; AUTOSTOP_GRACE_SECONDS must be a number'; exit 2; }
command -v timeout >/dev/null || { echo 'GNU timeout is required'; exit 2; }
command -v setsid >/dev/null || { echo 'setsid is required'; exit 2; }
# No default: left unset, a job started by any route but the driver on a machine that cannot remove itself would run with no cap on billing and nothing saying so.
[[ ${REQUIRE_AUTOSTOP:-} == 0 || ${REQUIRE_AUTOSTOP:-} == 1 ]] || {
  echo 'REQUIRE_AUTOSTOP must be set to 1 or 0. 1 (every RunPod pod; ops/drive_runpod.sh sets it): the job does not start unless this machine has armed its own removal. 0: the job starts even if no removal timer can be armed, which is only right where something else bounds the billing.' | tee -a "$OUT/autostop.log"
  exit 2
}
# A relaunch must not share the GPU with an earlier job that is still running, and that job's exit must not write this
# run's EXIT_CODE or arm a removal in the middle of it. Its runner goes first, without its exit trap, then its whole process group.
PIDS=${REMOTE_JOB_PIDS:-$HOME/.remote_job.pids}
echo "runner state file: $PIDS (REMOTE_JOB_PIDS overrides the standard HOME path)"
if [[ -s $PIDS ]]; then
  read -r OLD_RUNNER OLD_GROUP < "$PIDS"
  if kill -0 "$OLD_RUNNER"; then echo "stopping earlier runner $OLD_RUNNER"; kill -9 "$OLD_RUNNER"; fi
  if kill -0 -- "-$OLD_GROUP"; then echo "stopping earlier job group $OLD_GROUP"; kill -9 -- "-$OLD_GROUP"; fi
  pkill -9 -s "$OLD_GROUP" || [[ $? == 1 ]]  # the session may already be gone; other signalling errors stop the relaunch
  # A killed 60 GB process takes a moment to give the card back; the new job's self-test reads the card's memory.
  for _ in $(seq 30); do
    kill -0 -- "-$OLD_GROUP" || break
    sleep 1
  done
  rm -f "$PIDS"  # names a stopped job now; a later relaunch must not signal its ids once they are reused
fi
if can_remove; then
  AUTOSTOP=1
  # A relaunch on the same pod replaces the earlier job's timers; otherwise the old one would remove the pod mid-run.
  arm "$(($2+KILL_AFTER_SECONDS+GRACE))"
elif [[ $REQUIRE_AUTOSTOP == 1 ]]; then
  echo 'NO_AUTOSTOP: this pod cannot remove itself (no pod id, no runpodctl, or its key was refused); not starting' | tee -a "$OUT/autostop.log"
  exit 3
else
  echo "$(date -u +%FT%TZ) NO_AUTOSTOP: REQUIRE_AUTOSTOP=0 and this machine cannot remove itself (no pod id, no runpodctl, or its key was refused). The job starts with NO removal timer: this script stops the job after ${2}s but does not stop the billing." | tee -a "$OUT/autostop.log"
fi
# Its own session, so everything the job starts (servers, their workers) is one process group a relaunch can stop.
setsid timeout --signal=TERM --kill-after="$KILL_AFTER_SECONDS" "$2" bash "$1" &
JOB=$!
printf '%s %s\n' "$$" "$JOB" > "$PIDS"
wait "$JOB"
