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
[[ $REMOTE =~ ^ops/[a-zA-Z0-9_-]+\.sh$ && -f $REMOTE && $DEADLINE =~ ^[1-9][0-9]*$ ]] || { echo 'the remote script must be an existing ops/*.sh and the deadline a whole number of minutes'; exit 2; }
if [[ $REMOTE == ops/remote_runpod_smoke.sh && ${JUDGE_KIND:-api} == api ]]; then
  [[ ${FORWARD_JUDGE_KEY:-} == 1 && -n ${OPENAI_API_KEY:-} ]] || {
    echo 'API smoke needs FORWARD_JUDGE_KEY=1 and OPENAI_API_KEY before starting'; exit 2;
  }
fi
# The copy to and from the pod passes its ssh options to rsync as one string, which rsync splits at spaces: a results path with
# a space sends ssh half of the known-hosts path as the host. The sync then fails after the pod is armed, and so do the final copies.
[[ $RESULTS != *[[:space:]]* ]] || { echo "the results directory must not contain spaces (give it relative to the repo): $RESULTS"; exit 2; }
if [[ ${SMOKE_SKIP_PROBE:-0} == 1 && -z ${SMOKE_TOKEN_LIMIT:-} ]]; then
  echo 'SMOKE_SKIP_PROBE=1 needs SMOKE_TOKEN_LIMIT, the token limit an earlier probe on this model called for; without it the job stops on the pod after its install and download'; exit 2
fi
[[ ${JUDGE_KIND:-api} == api || ${JUDGE_KIND:-api} == pod ]] || { echo 'JUDGE_KIND must be api or pod'; exit 2; }
# The smoke script runs model-written Lean, and how that is isolated is chosen by name for every run. Left unset it used to mean
# `required`, which RunPod containers cannot satisfy: the job then failed only after a pod boot and a sync had been paid for.
if [[ $REMOTE == ops/remote_runpod_smoke.sh && -z ${LEAN_ISOLATION:-} ]]; then
  echo 'LEAN_ISOLATION must be set for the smoke script. required: Lean runs only inside a network namespace, which RunPod containers cannot create, so on a RunPod pod the job stops before any install. none: Lean runs as an unprivileged user with an empty environment but with the network reachable, which is acceptable only on a disposable pod.'
  exit 2
fi
[[ ${LEAN_ISOLATION:-required} == required || ${LEAN_ISOLATION:-} == none ]] || { echo 'LEAN_ISOLATION must be required or none'; exit 2; }
[[ ${JUDGE_MAX_USD:-1} =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo 'JUDGE_MAX_USD must be a number of dollars'; exit 2; }
# The grace is how long the pod outlives its job, and the results are only on the pod until this script has copied them. This script
# looks for the job's end every 60 s, connecting takes up to 10 s, and the final copy is up to three attempts 10 s apart, so under
# 180 s the pod can be gone before the copy is done; at 0 it is gone before the next look. The runner's own default is 600.
# The ceiling is the runner's default: the grace is billed time after the job, and the budget check counts ten minutes of it.
GRACE_FLOOR=180 GRACE_CEILING=600
if [[ -n ${AUTOSTOP_GRACE_SECONDS:-} ]]; then
  [[ $AUTOSTOP_GRACE_SECONDS =~ ^[1-9][0-9]*$ && $AUTOSTOP_GRACE_SECONDS -ge $GRACE_FLOOR && $AUTOSTOP_GRACE_SECONDS -le $GRACE_CEILING ]] || {
    echo "AUTOSTOP_GRACE_SECONDS must be a whole number of seconds, at least $GRACE_FLOOR and at most $GRACE_CEILING: the pod is removed that long after the job ends, and this script needs a 60 s poll plus the copy to get the results off it first. Got '$AUTOSTOP_GRACE_SECONDS'."
    exit 2
  }
fi
# SMOKE_* settings for the remote smoke script are passed through, and only if they are plain numbers.
# Only the names the chosen remote script itself mentions are accepted, read from its text so the list cannot drift: a misspelt
# name, or any SMOKE_* setting with a script that reads none, would be forwarded and ignored.
SMOKE_READ=$(awk '{ while (match($0, /SMOKE_[A-Z_]+/)) { print substr($0, RSTART, RLENGTH); $0 = substr($0, RSTART+RLENGTH) } }' "$REMOTE" | sort -u | tr '\n' ' ')
SMOKE_ENV=''
while IFS='=' read -r name value; do
  [[ $name =~ ^SMOKE_[A-Z_]+$ && $value =~ ^[0-9]+$ ]] || { echo "invalid $name"; exit 2; }
  if [[ " $SMOKE_READ" != *" $name "* ]]; then
    if [[ -n $SMOKE_READ ]]; then echo "$name is set, but $REMOTE does not read it and would ignore it. The SMOKE_* settings it reads: $SMOKE_READ"
    else echo "$name is set, but $REMOTE reads no SMOKE_* setting at all and would ignore it. Unset it, or drive the script it is meant for."; fi
    exit 2
  fi
  SMOKE_ENV+="$name=$value "
done < <(env | grep '^SMOKE_' || true)
# Everything above refuses a command that could only fail later, on a pod that is already billing. DRIVE_CHECK_ONLY=1 runs just
# those checks, with placeholder host and port if need be, so they can be made before a pod is deployed (`entrypoints/gate.py ready`).
if [[ ${DRIVE_CHECK_ONLY:-0} == 1 ]]; then echo 'check passed'; exit 0; fi
mkdir -p "$RESULTS"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$RESULTS/known_hosts" -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
pull() { rsync -az --partial --timeout=60 -e "${SSH[*]}" "root@$HOST:out/" "$RESULTS/"; }
say() { echo "$*" >> "$RESULTS/driver.log"; echo "$*"; }
# The results are only on the pod until this machine copies them, and the pod removes itself a grace period after the job ends:
# a laptop that goes to sleep in between loses them. macOS: hold off idle sleep for as long as this script lives (closing the lid
# still sleeps it). Elsewhere there is nothing here to do that, and the log says so.
if command -v caffeinate > /dev/null; then
  caffeinate -i -w $$ >> "$RESULTS/driver.log" 2>&1 < /dev/null &
else
  # `say` writes the log first; with the terminal gone its printed copy fails, and that must not stop the driver here.
  if ! say 'NOTE: no caffeinate on this machine, so nothing keeps it awake: if it sleeps before the final copy, the results go with the pod'; then
    echo 'the note above could not be printed: this script has no terminal' >> "$RESULTS/driver.log"
  fi
fi
finish() {
  code=$?
  # From here nothing may stop this function: with the terminal gone a plain `echo` fails, and under `set -e` that
  # ended the cleanup before the copy and the removal. Every step below reports its own failure.
  set +e
  say "copying results"
  COPIED=0
  for attempt in 1 2 3; do
    if pull; then COPIED=1; break; fi
    say "copy attempt $attempt failed"; sleep 10
  done
  if [[ $COPIED == 0 ]]; then say 'FINAL COPY FAILED three times: the results are only on the pod, so it is not removed from here'; fi
  if ! "${SSH[@]}" "root@$HOST" 'rm -f ~/.judge_key'; then say 'could not reach the pod to delete the judge key file'; fi
  date -u +%FT%TZ > "$RESULTS/driver-exit.txt"
  if [[ ${KEEP_POD:-0} == 1 || $COPIED == 0 ]]; then
    # Left running on purpose: say so only if a removal timer is really armed there.
    # The bracket keeps the pattern from matching the shell that runs this very command, which made the check always pass.
    if "${SSH[@]}" "root@$HOST" "pgrep -f '[r]unpodctl remove pod' > /dev/null"; then say "pod left running; a removal timer is running on it (out/autostop.log)"
    else say "REMINDER: the pod is running with NO removal timer. Terminate it in the RunPod console now"; code=97; fi
  elif "${SSH[@]}" "root@$HOST" 'bash -s -- --remove-now' < ops/remote_job.sh; then  # sent over the connection: the repo may not be on the pod yet
    # Accepted is not gone: the pod is gone when it stops answering.
    GONE=0
    for _ in $(seq 12); do
      if ! "${SSH[@]}" "root@$HOST" true 2>/dev/null; then GONE=1; break; fi  # fallback-ok: a refused connection is the expected outcome being waited for, and its message is noise
      sleep 5
    done
    if [[ $GONE == 1 ]]; then say "pod removed: it no longer answers"
    else say "REMINDER: RunPod accepted the removal but the pod still answers after a minute. Check the console now"; code=97; fi
  else say "REMINDER: could not remove the pod from here. Terminate it in the RunPod console now"; code=97; fi
  if [[ $COPIED == 0 && $code == 0 ]]; then code=96; fi
  exit "$code"
}
trap finish EXIT
for _ in $(seq 20); do "${SSH[@]}" "root@$HOST" true 2>/dev/null && break; sleep 15; done
"${SSH[@]}" "root@$HOST" true || { echo 'ssh failed'; exit 1; }
# Before anything else on a billing pod: a provisional removal timer, so an interrupted copy or a sleeping laptop is bounded.
# The job's own timer replaces it when the job starts.
"${SSH[@]}" "root@$HOST" 'bash -s -- --arm-only 900' < ops/remote_job.sh || { echo 'could not arm a provisional removal timer on the pod'; exit 1; }
# results/ stays behind except the model-read labels, which judge validation on the pod needs (7 MB).
rsync -az --timeout=120 --exclude .git --exclude .tools --exclude .claude --exclude spending --include 'results/' --include 'results/report-reading/***' --exclude 'results/*' -e "${SSH[*]}" ./ "root@$HOST:pilot-repo/"
# The API judge needs the OpenAI key on the pod. It is sent only on request, over the SSH channel into a mode-600
# file that the remote script reads once and deletes; it never appears in a command line or in the repo.
if [[ ${FORWARD_JUDGE_KEY:-} == 1 && $REMOTE == ops/remote_runpod_smoke.sh && ${JUDGE_KIND:-api} == api ]]; then
  printf '%s' "${OPENAI_API_KEY:?FORWARD_JUDGE_KEY=1 needs OPENAI_API_KEY}" | "${SSH[@]}" "root@$HOST" 'umask 077; cat > ~/.judge_key'
fi
# A fresh out/ every run: a reused pod must not hand back the previous run's summaries. Models live in /workspace.
# The remote log goes inside it so the runner's own errors are pulled with the results.
"${SSH[@]}" "root@$HOST" "if [ -e out ]; then mv out out.prev.\$(date +%s); fi; mkdir -p out; JUDGE_KIND=${JUDGE_KIND:-api} LEAN_ISOLATION=${LEAN_ISOLATION:-required} JUDGE_MAX_USD=${JUDGE_MAX_USD:-1} REQUIRE_AUTOSTOP=1 AUTOSTOP_GRACE_SECONDS=${AUTOSTOP_GRACE_SECONDS:-600} ${SMOKE_ENV}nohup bash pilot-repo/ops/remote_job.sh pilot-repo/$REMOTE $((DEADLINE*60)) > out/remote.log 2>&1 < /dev/null &"
date -u +%FT%TZ > "$RESULTS/started-at.txt"
END=$((SECONDS+DEADLINE*60+90))  # the remote timeout is DEADLINE; the extra time is for it to write EXIT_CODE
while ((SECONDS < END)); do
  sleep 60
  if code=$("${SSH[@]}" "root@$HOST" 'cat out/EXIT_CODE' 2>/dev/null); then
    [[ $code =~ ^[0-9]+$ && $code -le 255 ]] || { echo 'invalid remote exit status'; exit 1; }
    echo "remote finished with exit=$code"
    exit "$code"
  fi
  if ((SECONDS % 300 < 60)) && ! pull; then echo 'interim copy failed; will try again'; fi
done
echo 'driver deadline reached; the remote timeout and the pod removal timer are armed'
exit 1
