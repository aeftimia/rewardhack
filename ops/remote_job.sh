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
echo "runner output directory: $OUT (REMOTE_JOB_OUT overrides the standard HOME/out path)"
GRACE=${AUTOSTOP_GRACE_SECONDS:-600}  # time for the driver to copy results before the pod is removed
echo "runner grace period: ${GRACE}s (AUTOSTOP_GRACE_SECONDS overrides the standard copy window)"
mkdir -p "$OUT" || { echo "cannot create runner output directory: $OUT" >&2; exit 3; }
# An SSH session does not inherit the container's environment, where RunPod puts the pod's id and its own key.
pod_env() { if [[ -r /proc/1/environ ]]; then tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -1; fi; }
if [[ -n ${RUNPOD_POD_ID:-} ]]; then
  POD_ID=$RUNPOD_POD_ID
  echo 'pod identity source: RUNPOD_POD_ID in the runner environment'
else
  POD_ID=$(pod_env RUNPOD_POD_ID)
  echo 'pod identity source: container PID 1 environment (SSH did not supply RUNPOD_POD_ID)'
fi
export RUNPOD_API_KEY=${RUNPOD_API_KEY:-$(pod_env RUNPOD_API_KEY)}
AUTOSTOP=0
remove_after() {  # detached, so it outlives this script, the SSH session and a killed job
  # Asked again every 30 s for as long as this pod exists. It does not rely on what runpodctl's exit status means, which has
  # never been seen for a refusal: the loop ends when the pod, and this process with it, is gone.
  nohup setsid bash -c "sleep $1; while true; do runpodctl remove pod '$POD_ID'; sleep 30; done" >> "$OUT/autostop.log" 2>&1 < /dev/null &
  TIMER_PID=$!
  # Confirm detachment before releasing runner ownership: an unscheduled child still has
  # the runner's argv, so the next launch's timer search could miss it until it starts.
  # cap: 5x estimated 1 s (nohup, setsid and bash exec'ing into the loop; slower when the machine is busy)
  for _ in $(seq 50); do
    if timer_runs "$TIMER_PID" || ! kill -0 "$TIMER_PID"; then break; fi
    sleep 0.1
  done
  if ! timer_runs "$TIMER_PID"; then
    if kill -0 "$TIMER_PID"; then kill -9 "$TIMER_PID"; fi
    echo 'NO_AUTOSTOP: the removal timer did not start; earlier timers were left running' | tee -a "$OUT/autostop.log"
    return 1
  fi
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
  earlier=$(pgrep -f "[r]unpodctl remove pod '$POD_ID'") || [[ $? == 1 ]]  # no matches is normal; other errors stop arming
  remove_after "$1" || { echo 'could not arm a replacement removal timer'; exit 3; }
  new=$TIMER_PID
  for pid in $earlier; do
    if kill -0 "$pid"; then kill "$pid"; fi
  done
  echo "$(date -u +%FT%TZ) armed: pod $POD_ID is removed at most ${1}s from now (timer pid $new)" >> "$OUT/autostop.log"
}
can_remove() { [[ -n $POD_ID ]] && command -v runpodctl >/dev/null && runpodctl get pod "$POD_ID" >> "$OUT/autostop.log" 2>&1; }
finish() {
  code=$?
  trap - EXIT
  # The runner and every job process share one session, including inner timeouts' process
  # groups. Timers use their own detached sessions. Keep only this writer alive for cleanup.
  for pid in $(pgrep -s "$$"); do
    if [[ $pid != "$$" ]] && kill -0 "$pid"; then kill -9 "$pid"; fi
  done
  rm -f "$HOME/.judge_key"  # never outlives the job, whichever remote script ran and however it ended
  printf '%s\n' "$code" > "$OUT/EXIT_CODE.tmp"
  mv "$OUT/EXIT_CODE.tmp" "$OUT/EXIT_CODE"
  touch "$OUT/DONE"
  if ((AUTOSTOP)); then
    echo "$(date -u +%FT%TZ) job ended with $code; removing pod in ${GRACE}s" >> "$OUT/autostop.log"
    if ! remove_after "$GRACE"; then echo 'grace timer failed; the earlier deadline timer still bounds billing' >> "$OUT/autostop.log"; fi
  fi
  # Keep ownership until the last output and timer write: a relaunch may rotate out as soon
  # as this record disappears. Only remove this runner's record.
  if [[ -n ${PIDS:-} && -s $PIDS ]] && read -r runner _ < "$PIDS" && [[ $runner == "$$" ]]; then rm -f "$PIDS"; fi
  exit "$code"
}
stop_previous() {
  PIDS=${REMOTE_JOB_PIDS:-$HOME/.remote_job.pids}
  echo "runner state file: $PIDS (REMOTE_JOB_PIDS overrides the standard HOME path)"
  [[ -s $PIDS ]] || return 0
  local old_session
  read -r old_session < "$PIDS"
  [[ $old_session =~ ^[1-9][0-9]+$ ]] || { echo 'invalid earlier job session record'; return 1; }
  if pgrep -s "$old_session" >/dev/null; then pkill -9 -s "$old_session" || { echo 'could not stop earlier job session'; return 1; }; fi
  # cap: 5x estimated 6 s (a killed GPU job and its runner releasing their processes)
  for _ in $(seq 30); do
    # Zombies cannot write or hold the GPU; reaping them belongs to their parent, often pod PID 1.
    if ! pgrep -r D,I,R,S,T,t,W -s "$old_session" >/dev/null; then
      rm -f "$PIDS"
      return 0
    fi
    sleep 1
  done
  echo 'earlier runner or job session is still present; refusing a relaunch'
  return 1
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
LAUNCH_ID=''
if [[ ${1:-} == --launch ]]; then
  [[ $# == 4 && $2 =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'need launch id, script and seconds'; exit 2; }
  LAUNCH_ID=$2
  shift 2
fi
[[ $# == 2 && $2 =~ ^[1-9][0-9]*$ && $GRACE =~ ^[0-9]+$ ]] || { echo 'need a job script and a whole number of seconds; AUTOSTOP_GRACE_SECONDS must be a number'; exit 2; }
command -v timeout >/dev/null || { echo 'GNU timeout is required'; exit 2; }
command -v setsid >/dev/null || { echo 'setsid is required'; exit 2; }
# Re-exec once as a session leader. `--wait` preserves exit status if setsid must fork.
# Check the actual session on entry, including after re-exec; a broken setsid must fail, not loop.
if ! pgrep -s "$$" >/dev/null; then
  [[ ${REMOTE_JOB_SESSION_ENTRY:-0} == 0 ]] || { echo 'setsid did not establish the runner session'; exit 2; }
  export REMOTE_JOB_SESSION_ENTRY=1
  if [[ -n $LAUNCH_ID ]]; then exec setsid --wait bash "$0" --launch "$LAUNCH_ID" "$@"
  else exec setsid --wait bash "$0" "$@"; fi
fi
# No default: left unset, a job started by any route but the driver on a machine that cannot remove itself would run with no cap on billing and nothing saying so.
[[ ${REQUIRE_AUTOSTOP:-} == 0 || ${REQUIRE_AUTOSTOP:-} == 1 ]] || {
  echo 'REQUIRE_AUTOSTOP must be set to 1 or 0. 1 (every RunPod pod; ops/drive_runpod.sh sets it): the job does not start unless this machine has armed its own removal. 0: the job starts even if no removal timer can be armed, which is only right where something else bounds the billing.' | tee -a "$OUT/autostop.log"
  exit 2
}
# Record the whole session before any setup or child spawn, so replacement has no unowned
# startup window. A previous runner and all of its writers are stopped before output moves.
stop_previous || { echo 'cannot start while earlier job remains'; exit 1; }
if ! printf '%s\n' "$$" > "$PIDS.$$" || ! mv "$PIDS.$$" "$PIDS"; then echo 'could not publish job session ownership'; exit 1; fi
if [[ -n $LAUNCH_ID ]]; then
  [[ -f $1 && ! -e $OUT.prev.$LAUNCH_ID ]] || { echo 'job missing or launch archive already exists'; exit 1; }
  if [[ -e $OUT ]]; then mv "$OUT" "$OUT.prev.$LAUNCH_ID" || { echo 'could not archive previous output'; exit 1; }; fi
  mkdir "$OUT" || { echo 'could not create launch output'; exit 1; }
  exec > "$OUT/remote.log" 2>&1
fi
trap finish EXIT
trap 'exit 143' TERM
if [[ -n $LAUNCH_ID ]]; then
  if [[ -f $HOME/.judge_key.$LAUNCH_ID ]]; then mv "$HOME/.judge_key.$LAUNCH_ID" "$HOME/.judge_key" || { echo 'could not activate judge key'; exit 1; }; fi
  printf '%s\n' "$LAUNCH_ID" > "$OUT/LAUNCH_ID" || { echo 'could not publish launch receipt'; exit 1; }
fi
if can_remove; then
  # A relaunch on the same pod replaces the earlier job's timers; otherwise the old one would remove the pod mid-run.
  arm "$(($2+KILL_AFTER_SECONDS+GRACE))"
  AUTOSTOP=1
elif [[ $REQUIRE_AUTOSTOP == 1 ]]; then
  echo 'NO_AUTOSTOP: this pod cannot remove itself (no pod id, no runpodctl, or its key was refused); not starting' | tee -a "$OUT/autostop.log"
  exit 3
else
  echo "$(date -u +%FT%TZ) NO_AUTOSTOP: REQUIRE_AUTOSTOP=0 and this machine cannot remove itself (no pod id, no runpodctl, or its key was refused). The job starts with NO removal timer: this script stops the job after ${2}s but does not stop the billing." | tee -a "$OUT/autostop.log"
fi
# Keep the job in this session: a relaunch can stop it even between spawn and `$!` assignment.
timeout --signal=TERM --kill-after="$KILL_AFTER_SECONDS" "$2" bash "$1" &
JOB=$!
wait "$JOB"
