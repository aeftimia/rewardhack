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
# Every call that could hold a billing pod is bounded with it (try_ssh, the arm call, on_pod): without it each would fail
# as "not found" and the pod would be given up on with no timer, so it is checked before anything, check-only included.
# The driver reads nothing from its terminal: every command's stdin is /dev/null unless it is given one. Under
# `timeout` ssh runs in a background process group, and one that read the terminal (a key pressed while the driver runs
# in it, or a driver started with `&`) was stopped and its call failed or hung (review rounds 36 and 37).
exec < /dev/null
command -v timeout > /dev/null || { echo 'the driver needs a timeout command on this machine (macOS 27 has /usr/bin/timeout; else coreutils)'; exit 2; }
SMOKE_VLLM_ROLLOUTS=${SMOKE_VLLM_ROLLOUTS:-0}
[[ $SMOKE_VLLM_ROLLOUTS == 0 || $SMOKE_VLLM_ROLLOUTS == 1 ]] || { echo 'SMOKE_VLLM_ROLLOUTS must be 0 or 1'; exit 2; }
if [[ $SMOKE_VLLM_ROLLOUTS == 1 ]]; then
  [[ ${JUDGE_KIND:?vLLM pilot needs JUDGE_KIND=api} == api && ${SMOKE_SKIP_PROBE:?vLLM pilot needs SMOKE_SKIP_PROBE=1} == 1 && ${SMOKE_SECOND_RUN:?vLLM pilot needs SMOKE_SECOND_RUN=0} == 0 ]] || {
    echo 'vLLM pilot needs API judge, SMOKE_SKIP_PROBE=1 and SMOKE_SECOND_RUN=0'; exit 2;
  }
  # The vLLM server's context is sized to SMOKE_TOKEN_LIMIT: a longer memory step could never be sampled there.
  [[ -z ${SMOKE_MEMORY_LIMIT+set} ]] || { echo 'SMOKE_MEMORY_LIMIT is a question for the transformers sampler: not with SMOKE_VLLM_ROLLOUTS=1'; exit 2; }
fi
source ops/pod_pins.env  # PART_ONE_RUNS: the Part 1 jobs and their runs
JUDGE_KIND=${JUDGE_KIND:-api}
if [[ -n ${SMOKE_PART_ONE+set} ]]; then PART_ONE=$SMOKE_PART_ONE; else PART_ONE=0; fi  # unset: the preflight run
# No leading zeros: bash reads 08 as octal, and 00 would pass the string tests below as a Part 1 job of row 0, with no runs.
# At most three digits: a 20-digit number wraps to a negative index, which bash 4.3+ reads from the end of the table.
[[ $PART_ONE =~ ^(0|[1-9][0-9]{0,2})$ && -n ${PART_ONE_RUNS[$PART_ONE]+set} ]] || { echo "SMOKE_PART_ONE must index a job in ops/pod_pins.env's PART_ONE_RUNS (0 to $((${#PART_ONE_RUNS[@]} - 1)))"; exit 2; }
read -ra PART_ONE_JOB <<< "${PART_ONE_RUNS[$PART_ONE]}"
if [[ -n ${SMOKE_PROBE_LARGE+set} ]]; then PROBE_LARGE=$SMOKE_PROBE_LARGE; else PROBE_LARGE=0; fi  # unset: no inference probe
[[ $PROBE_LARGE =~ ^(0|[1-9][0-9]{0,2})$ && -n ${PROBE_SET_NAMES[$PROBE_LARGE]+set} ]] || { echo 'SMOKE_PROBE_LARGE must index PROBE_SET_NAMES in ops/pod_pins.env'; exit 2; }
# A Part 1 job with no runs is the probe alone; without the probe it would do nothing but setup.
[[ $PART_ONE == 0 || ${#PART_ONE_JOB[@]} -gt 0 || $PROBE_LARGE != 0 ]] || { echo "SMOKE_PART_ONE=$PART_ONE has no training runs: it needs a nonzero SMOKE_PROBE_LARGE stage set"; exit 2; }
# The probe runs after a Part 1 job's training, on the vLLM path that job already requires.
[[ $PROBE_LARGE == 0 || $PART_ONE != 0 ]] || { echo 'SMOKE_PROBE_LARGE=1 runs after a Part 1 job: set SMOKE_PART_ONE too'; exit 2; }
# The job's training model (ops/pod_pins.env). The 20b permission probe takes a whole card after a one-card job's training.
MODEL_INDEX=${PART_ONE_MODEL[$PART_ONE]}
[[ $PROBE_LARGE == 0 || ${MODEL_GPUS[$MODEL_INDEX]} == 1 ]] || { echo "SMOKE_PROBE_LARGE=1 follows a one-card job; SMOKE_PART_ONE=$PART_ONE trains ${MODEL_REPOS[$MODEL_INDEX]} on ${MODEL_GPUS[$MODEL_INDEX]} cards"; exit 2; }
# The preflight run's own limits add up to about 150 minutes plus setup (ops/remote_runpod_smoke.sh's header); the script's
# `# deadline:` line is the floor for every job, so this minimum is held here (notes section 80).
[[ $REMOTE != ops/remote_runpod_smoke.sh || $PART_ONE != 0 || ! $DEADLINE =~ ^[0-9]+$ || $DEADLINE -ge 170 ]] || { echo 'the preflight run (SMOKE_PART_ONE unset or 0) needs a deadline of at least 170 minutes'; exit 2; }
if [[ $PART_ONE != 0 ]]; then
  # A Part 1 job's runs are the whole training (the inference probe may follow): no preflight probe, no funded-arm runs, and its length and time limit stated, not
  # taken from the preflight defaults (12 steps, 60 minutes). ${X+set} tests whether the launch set X (bash 3.2 has no -v).
  # vLLM samples the rollouts and evaluations: without it the runs would use the trainer's own sampler, many times
  # slower, and record a run on the wrong backend (review round 16).
  [[ -n ${SMOKE_SKIP_PROBE+set} && $SMOKE_SKIP_PROBE == 1 && -n ${SMOKE_SECOND_RUN+set} && $SMOKE_SECOND_RUN == 0 && $SMOKE_VLLM_ROLLOUTS == 1
     && -z ${SMOKE_MEMORY_LIMIT+set} && -n ${SMOKE_STEPS+set} && -n ${SMOKE_TRAIN_SECONDS+set} && -n ${SMOKE_FINAL_EVAL_SECONDS+set} ]] || {
    echo "SMOKE_PART_ONE=$PART_ONE needs SMOKE_SKIP_PROBE=1, SMOKE_SECOND_RUN=0, SMOKE_VLLM_ROLLOUTS=1, no SMOKE_MEMORY_LIMIT, and SMOKE_STEPS, SMOKE_TRAIN_SECONDS and SMOKE_FINAL_EVAL_SECONDS set"; exit 2;
  }
  # The setup allowance below is the API judge's; a judge served on the pod adds its download, start and validation.
  [[ -z ${JUDGE_KIND+set} || $JUDGE_KIND == api ]] || { echo "SMOKE_PART_ONE=$PART_ONE is sized for JUDGE_KIND=api"; exit 2; }
  # The runs one after the other, each allowed SMOKE_TRAIN_SECONDS, after setup and the vLLM server start (per model, with
  # its basis, in ops/pod_pins.env): a shorter job deadline would end the last run before its after-evaluation.
  PART1_SETUP_MINUTES=${MODEL_SETUP_LIMITS[$MODEL_INDEX]}
  # Whole numbers before any arithmetic: bash evaluates text inside (( )) as an expression. At most six digits (about 11
  # days of seconds): a longer one overflows bash's arithmetic and turned the deadline check below into a pass (round 28).
  [[ $DEADLINE =~ ^[1-9][0-9]{0,5}$ && $SMOKE_TRAIN_SECONDS =~ ^[1-9][0-9]{0,5}$ && $SMOKE_FINAL_EVAL_SECONDS =~ ^[1-9][0-9]{0,5}$ ]] || {
    echo "SMOKE_PART_ONE=$PART_ONE needs the deadline, SMOKE_TRAIN_SECONDS and SMOKE_FINAL_EVAL_SECONDS as whole numbers of at most six digits"; exit 2;
  }
  # The after-evaluation's reserve comes out of each run's limit and must leave the before-evaluation and some training.
  ((SMOKE_FINAL_EVAL_SECONDS < SMOKE_TRAIN_SECONDS)) || {
    echo "SMOKE_PART_ONE=$PART_ONE needs SMOKE_FINAL_EVAL_SECONDS below SMOKE_TRAIN_SECONDS"; exit 2;
  }
  # A run stopped at its own limit exits only once the answers already being scored finish (outcomes.score_concurrently):
  # one row's worst case, a grader check (10 s), three verifier checks with retries (3 x 80 s) and a judge request with its
  # retries (4 x 60 s): 490 s, and the OpenAI client's waits between retries on top (review round 14).
  # cap: 1.5x estimated 540 s
  SCORING_TAIL_SECONDS=810
  # The selected probe stages use the model already fetched during setup; each has its own card wait and outer limit.
  TRAINING_SECONDS=$((${#PART_ONE_JOB[@]} * (SMOKE_TRAIN_SECONDS + SCORING_TAIL_SECONDS)))
  # Before the training, its environment's check (with the kill-after of its timeout).
  PROBE_LARGE_SECONDS=$((PROBE_ENV_CHECK_SECONDS + KILL_AFTER_SECONDS))
  for stage in "${PROBE_LARGE_STAGES[@]}"; do
    read -r _ seconds _ _ _ stage_set _ <<< "$stage"
    [[ $stage_set == "$PROBE_LARGE" ]] || continue
    PROBE_LARGE_SECONDS=$((PROBE_LARGE_SECONDS + CARD_FREE_SECONDS + seconds + PROBE_LARGE_EXIT_SECONDS + KILL_AFTER_SECONDS))
  done
  RUNS_SECONDS=$((TRAINING_SECONDS + (PROBE_LARGE != 0) * PROBE_LARGE_SECONDS))
  ((DEADLINE*60 >= PART1_SETUP_MINUTES*60 + RUNS_SECONDS)) || {
    echo "SMOKE_PART_ONE=$PART_ONE needs a deadline of at least $((PART1_SETUP_MINUTES + (RUNS_SECONDS+59)/60)) minutes: setup, ${#PART_ONE_JOB[@]} run(s) of SMOKE_TRAIN_SECONDS each with its scoring tail, and the selected inference probe if set"; exit 2;
  }
fi
[[ $REMOTE =~ ^ops/[a-zA-Z0-9_-]+\.sh$ && -f $REMOTE && $DEADLINE =~ ^[1-9][0-9]*$ ]] || { echo 'the remote script must be an existing ops/*.sh and the deadline a whole number of minutes'; exit 2; }
if [[ $REMOTE == ops/remote_runpod_smoke.sh && $JUDGE_KIND == api ]]; then
  [[ ${FORWARD_JUDGE_KEY:-} == 1 && -n ${OPENAI_API_KEY:-} ]] || {
    echo 'API smoke needs FORWARD_JUDGE_KEY=1 and OPENAI_API_KEY before starting'; exit 2;
  }
fi
# The copy to and from the pod passes its ssh options to rsync as one string, which rsync splits at spaces: a results path with
# a space sends ssh half of the known-hosts path as the host. The sync then fails after the pod is armed, and so do the final copies.
[[ $RESULTS != *[[:space:]]* ]] || { echo "the results directory must not contain spaces (give it relative to the repo): $RESULTS"; exit 2; }
# A run owns its local directory too: rsync preserves absent files, and reusing a directory would retain an earlier
# run's rows while overwriting its launch commit. Reject even an empty directory or a broken symlink.
[[ ! -e $RESULTS && ! -L $RESULTS ]] || { echo "the results directory must be new: $RESULTS"; exit 2; }
if [[ ${SMOKE_SKIP_PROBE:-0} == 1 && -z ${SMOKE_TOKEN_LIMIT:-} ]]; then
  echo 'SMOKE_SKIP_PROBE=1 needs SMOKE_TOKEN_LIMIT, the token limit an earlier probe on this model called for; without it the job stops on the pod after its install and download'; exit 2
fi
[[ $JUDGE_KIND == api || $JUDGE_KIND == pod ]] || { echo 'JUDGE_KIND must be api or pod'; exit 2; }
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
# The grace is billed time after the job, and the launch record's budget check counts all of it. The ceiling is the runner's
# default, except for a pod kept for a relaunch (KEEP_POD=1): there the grace must also hold the copy of several GB of
# checkpoints, sizing the next launch from the results, `gate.py ready` (about 2 minutes of tests) and the relaunch itself.
if [[ -n ${KEEP_POD+set} ]]; then KEEP=$KEEP_POD; else KEEP=0; fi  # unset: the pod is removed after the copy
[[ $KEEP == 0 || $KEEP == 1 ]] || { echo 'KEEP_POD must be 0 or 1'; exit 2; }
GRACE_CEILINGS=(600 1800)  # indexed by KEEP
GRACE_FLOOR=180 GRACE_CEILING=${GRACE_CEILINGS[$KEEP]}
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
SMOKE_EXPORTED=$(env | awk '/^SMOKE_/')
while IFS='=' read -r name value; do
  [[ -n $name ]] || continue  # a here-string for an empty environment selection contains one blank line
  [[ $name =~ ^SMOKE_[A-Z_]+$ && $value =~ ^[0-9]+$ ]] || { echo "invalid $name"; exit 2; }
  if [[ " $SMOKE_READ" != *" $name "* ]]; then
    if [[ -n $SMOKE_READ ]]; then echo "$name is set, but $REMOTE does not read it and would ignore it. The SMOKE_* settings it reads: $SMOKE_READ"
    else echo "$name is set, but $REMOTE reads no SMOKE_* setting at all and would ignore it. Unset it, or drive the script it is meant for."; fi
    exit 2
  fi
  SMOKE_ENV+="$name=$value "
done <<< "$SMOKE_EXPORTED"
# With the judge served on the pod the smoke script has a second install and download, a 15 minute readiness wait and up to 30 minutes
# of judge validation on top of its 170 minute budget.
if [[ $REMOTE == ops/remote_runpod_smoke.sh && $JUDGE_KIND == pod && $DEADLINE -lt 240 ]]; then
  echo 'JUDGE_KIND=pod needs a deadline of at least 240 minutes for the smoke script: its limits add up to about 234'; exit 2
fi
# Everything above refuses a command that could only fail later, on a pod that is already billing. DRIVE_CHECK_ONLY=1 runs just
# those checks, with placeholder host and port if need be, so they can be made before a pod is deployed (`entrypoints/gate.py ready`).
if [[ ${DRIVE_CHECK_ONLY:-0} == 1 ]]; then echo 'check passed'; exit 0; fi
# Reserve the destination atomically before consuming the launch record. Two drivers given the same path cannot
# both pass this mkdir, even if each saw it absent above. A failed gate leaves this empty reservation for inspection.
mkdir -p -- "$(dirname -- "$RESULTS")" || { echo "could not create the results parent: $RESULTS" >&2; exit 2; }
mkdir -- "$RESULTS" || { echo "could not reserve a new results directory: $RESULTS" >&2; exit 2; }
# Before contacting a pod: the launch record (`entrypoints/gate.py ready`) must cover this launch, its commit,
# ledger, remote script, deadline and the driver settings in this environment, and must not have started one already. It is
# marked used here. No host is exempt: tests run a copy of ops/ whose gate.sh is a stub (tests/conftest.py).
bash ops/gate.sh launch --remote "$REMOTE" --deadline "$DEADLINE" || { echo 'not started: the launch check above refused it, before any pod was contacted' >&2; exit 2; }
# The commit the pod runs (the launch record requires it clean), kept with the results: the run archive records it (notes section 80).
git rev-parse HEAD > "$RESULTS/launch-commit.txt"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$RESULTS/known_hosts" -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
# rsync never deletes here, so every resumable checkpoint an interim copy saw would stay on this machine (the trainer keeps
# two on the pod; with expert LoRA each is about 3 GB). Interim copies leave them out; the final copy takes the ones left.
# The vLLM hand-over adapter (rewritten every step) and the reload-check adapter are scratch, about 1 GB each: never copied.
# Enter the owned directory before checking it, so a later rename cannot redirect the copy.
pull() { rsync -az --partial --timeout=60 -e "${SSH[*]}" --rsync-path="$OWN_OUTPUT && rsync" --exclude='adapter-rollout/' --exclude='adapter-sync-check/' "$@" "root@$HOST:./" "$RESULTS/"; }
# Sizes in the final copy, smallest first: the pod removes itself at most 10 minutes after the job ends.
SMALL=(--exclude='adapter*/' --exclude='trainer/checkpoint-*/')  # logs, rows, summaries, evaluations
ADAPTERS=(--exclude='trainer/checkpoint-*/')                      # and the saved adapters, about 1 GB each
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
# Asked of a pod this run armed no timer on (see `finish`): whether any job was ever launched there. POSIX sh, as the
# pod's login shell runs it; tests/test_round30_followups.py runs it against a real failed --arm-only.
# shellcheck disable=SC2016  # expanded by the pod's shell, not here
# A job's trace is a remote.log, in out/ or in an out.prev.* a relaunch moved aside; an out.prev.* holding only what an
# arm call wrote (autostop.log) is none (review round 48).
JOB_TRACES='set -- out.prev.*/remote.log; if [ -e out/remote.log ] || [ -e "$1" ]; then echo held; else echo empty; fi'
# One command on the pod, bounded as a whole: the cleanup below asks the RunPod API too (`--remove-now`), and a stalled
# API must end in the reminders, not hold the driver with the pod billing (review round 35). ARM_SECONDS: ops/pod_pins.env.
# shellcheck disable=SC2329  # run by `finish` below
on_pod() { timeout "$ARM_SECONDS" "${SSH[@]}" "root@$HOST" "$@"; }
# shellcheck disable=SC2329  # run by `trap finish EXIT` below
finish() {
  code=$?
  # From here nothing may stop this function: with the terminal gone a plain `echo` fails, and under `set -e` that
  # ended the cleanup before the copy and the removal. Every step below reports its own failure.
  set +e
  # Only a confirmed receipt authorizes copying out/. Before any launch was sent, ask whether the pod holds
  # earlier results so cleanup can preserve them. Take the last line to ignore a login-shell banner.
  HELD=unknown
  if [[ $LAUNCH_SENT == 0 ]]; then
    HELD=$(on_pod "$JOB_TRACES" | tail -n 1)
    if [[ -z $HELD ]]; then HELD=$(on_pod "$JOB_TRACES" | tail -n 1); fi
  fi
  if [[ $LAUNCH_SENT == 1 && $LAUNCHED == 0 ]]; then
    say "this launch's receipt was not confirmed, so whether this run's job started is unknown: out/ is not copied and the pod is not removed from here. Its removal timer bounds it; inspect launch.$LAUNCH_ID.log and out/LAUNCH_ID on the pod"
    COPIED=1 CHECKPOINTS=0 SMALL_COPIED=1
  elif [[ $LAUNCHED == 0 && $HELD == empty ]]; then
    say 'no job was launched on this pod: nothing to copy'
    COPIED=1 CHECKPOINTS=1 SMALL_COPIED=1
  elif [[ $LAUNCHED == 0 ]]; then
    if [[ -z $HELD ]]; then say 'this run launched no job, and the pod did not answer whether one ever ran there: nothing is copied, and it is not removed from here'
    else say "this run launched no job, and the pod holds an earlier job's results: they are not copied into this run's folder, and the pod is not removed from here"; fi
    COPIED=1 CHECKPOINTS=0 SMALL_COPIED=1
  else
  say "copying results"
  # The small files, then the adapters, then the resumable checkpoints (several GB with expert LoRA): the pod removes
  # itself at most 10 minutes after the job ends, and a slow copy must cost the checkpoints, never the results.
  # Results and checkpoints are recorded apart, so that a failed checkpoint copy is not reported as lost results.
  COPIED=0 CHECKPOINTS=0 SMALL_COPIED=0
  for attempt in 1 2 3; do
    if pull "${SMALL[@]}"; then SMALL_COPIED=1; fi
    if [[ $SMALL_COPIED == 1 ]] && pull "${ADAPTERS[@]}"; then COPIED=1; break; fi
    say "copy attempt $attempt failed"; sleep 10
  done
  if [[ $SMALL_COPIED == 0 ]]; then say 'FINAL COPY FAILED three times: the results are only on the pod, so it is not removed from here'
  elif [[ $COPIED == 0 ]]; then say 'the logs, rows, summaries and evaluations are copied but the saved adapters are not: the pod is not removed from here, so they can still be fetched until its own timer removes it'
  else
    for attempt in 1 2 3; do
      if pull; then CHECKPOINTS=1; break; fi
      say "checkpoint copy attempt $attempt failed"; sleep 10
    done
    if [[ $CHECKPOINTS == 0 ]]; then say 'the results are copied but the resumable checkpoints are not: the pod is not removed from here, so they can still be fetched until its own timer removes it'; fi
  fi
  fi  # the copy
  # An unconfirmed detached launch may still need its staged key. The pod's timer bounds that file's lifetime.
  if [[ -n ${LAUNCH_ID:-} && ( $LAUNCH_SENT == 0 || $LAUNCHED == 1 ) ]] && ! on_pod "rm -f ~/.judge_key.$LAUNCH_ID"; then
    say "could not reach the pod to delete this launch's staged judge key"
  fi
  date -u +%FT%TZ > "$RESULTS/driver-exit.txt"
  # A later launch must never be removed by this driver's cleanup.
  REMOVE_COMMAND='bash -s -- --remove-now'
  if [[ $LAUNCHED == 1 ]]; then REMOVE_COMMAND="$OWN_OUTPUT && $REMOVE_COMMAND"; fi
  # A failed arm on a fresh, empty pod requires immediate removal even with KEEP_POD. Preserve earlier results
  # on a reused pod, but verify that a removal timer remains. out/ alone proves nothing: arming creates it too.
  if [[ $ARMED == 0 && -z $HELD ]] && ! on_pod true; then
    if [[ $SSH_UP == 1 ]]; then why='the pod stopped answering after it accepted SSH, before this run armed a removal timer on it'
    else why='the pod never accepted SSH, so this run armed no removal timer on it'; fi
    say "REMINDER: $why (a kept pod keeps its earlier one). Terminate it in the RunPod console now unless out/autostop.log there shows one"; code=97
  elif [[ ( $ARMED == 1 && ( $KEEP == 1 || $CHECKPOINTS == 0 ) ) || ( $ARMED == 0 && $HELD != empty ) ]]; then
    # Left running on purpose: say so only if a removal timer is really armed there. A pod that no longer answers has
    # most likely removed itself on its timer; that is said as such, not as a pod running without one.
    # The bracket keeps the pattern from matching the shell that runs this very command, which made the check always pass.
    if ! on_pod true; then
      if [[ $ARMED == 1 ]]; then
        say "the pod no longer answers (ssh error above): it has removed itself on its timer, or the connection failed and its timer (armed $ARMED_AT, at most the provisional 15 minutes before the job's own replaces it) removes it. Confirm in the RunPod console"
      else  # a kept pod with an earlier job's results, which this run armed no timer on (review round 47)
        say "REMINDER: the pod no longer answers, and this run armed no removal timer on it; a kept pod keeps its earlier one. Check in the RunPod console now that it is gone or has a timer (out/autostop.log there)"; code=97
      fi
    elif on_pod "pgrep -f '[r]unpodctl remove pod' > /dev/null"; then say "pod left running; a removal timer is running on it (out/autostop.log)"
    else say "REMINDER: the pod is running with NO removal timer. Terminate it in the RunPod console now"; code=97; fi
  elif on_pod "$REMOVE_COMMAND" < ops/remote_job.sh; then  # sent over the connection: the repo may not be on the pod yet
    # Accepted is not gone: the pod is gone when it stops answering.
    GONE=0
    for _ in $(seq 12); do
      if ! on_pod true 2>> "$RESULTS/ssh-removal.log"; then GONE=1; break; fi  # a refused connection is what is waited for; its message is kept there
      sleep 5
    done
    if [[ $GONE == 1 ]]; then say "pod removed: it no longer answers"
    else say "REMINDER: RunPod accepted the removal but the pod still answers after a minute. Check the console now"; code=97; fi
  else say "REMINDER: removal was not confirmed. Check the pod's launch ownership and removal timer now before terminating it: a different launch must be preserved"; code=97; fi
  if [[ $SMALL_COPIED == 0 && $code == 0 ]]; then code=96; fi
  if [[ $SMALL_COPIED == 1 && $COPIED == 0 && $code == 0 ]]; then code=94; fi
  if [[ $COPIED == 1 && $CHECKPOINTS == 0 && $code == 0 ]]; then code=95; fi
  exit "$code"
}
ARMED=0
# Set once the pod has accepted SSH: a try that connects ends the wait, and only a wait that never connected makes the
# final try (review round 37); `finish` tells a pod that never answered from one that stopped answering (round 45).
SSH_UP=0
LAUNCH_SENT=0  # set just before the launch call
LAUNCHED=0  # set only after the pod confirms this launch's receipt
trap finish EXIT
# Within SSH_UP_SECONDS (ops/pod_pins.env), which the launch gate counts: each try is bounded as a whole (ssh's
# ConnectTimeout covers only the connection; BatchMode keeps it from waiting on a prompt) and followed by a 15 s pause.
# The refusals while the pod boots are kept in ssh-wait.log; the last try shows its own (review rounds 31 and 32).
# cap: 2.5x estimated 10 s (one ssh try that answers: connect, key exchange, `true`)
SSH_TRY_SECONDS=25
try_ssh() { timeout "$SSH_TRY_SECONDS" "${SSH[@]}" "root@$HOST" true; }
# Bounded by time, not by a count of tries: a booting pod usually refuses at once, so a count gave up after a third of
# the wait (review round 33). The last try starts early enough that it, a pause and the final try end within the wait;
# the count only keeps a stubbed `sleep` (the tests) from spinning.
SSH_END=$((SECONDS + SSH_UP_SECONDS - 2 * SSH_TRY_SECONDS - 15))
for _ in $(seq $((SSH_UP_SECONDS / 15))); do
  ((SECONDS < SSH_END)) || break
  if try_ssh 2>> "$RESULTS/ssh-wait.log"; then SSH_UP=1; break; fi
  sleep 15
done
((SSH_UP)) || try_ssh || { echo "ssh failed: the pod did not accept SSH within $SSH_UP_SECONDS s (each refusal is in $RESULTS/ssh-wait.log: a key that needs a passphrase fails here too)"; exit 1; }
SSH_UP=1
# Before anything else on a billing pod: a provisional removal timer, so an interrupted copy or a sleeping laptop is bounded.
# The job's own timer replaces it when the job starts.
# Bounded as a whole (ARM_SECONDS, ops/pod_pins.env, counted by the gate): it asks the RunPod API whether the pod can
# remove itself, and a stalled API would otherwise hold the driver here with no timer armed (review round 34).
timeout "$ARM_SECONDS" "${SSH[@]}" "root@$HOST" 'bash -s -- --arm-only 900' < ops/remote_job.sh || { echo 'could not arm a provisional removal timer on the pod'; exit 1; }
ARMED=1  # from here a removal timer runs on the pod, so a pod that stops answering has most likely removed itself
ARMED_AT=$(date -u +%FT%TZ)
# results/ stays behind except the model-read labels, which judge validation on the pod needs (7 MB).
rsync -az --timeout=120 --exclude .git --exclude .tools --exclude .claude --exclude spending --include 'results/' --include 'results/report-reading/***' --exclude 'results/*' -e "${SSH[*]}" ./ "root@$HOST:pilot-repo/"
# The API judge needs the OpenAI key on the pod. It is sent only on request, over the SSH channel into a mode-600
# file that the remote script reads once and deletes; it never appears in a command line or in the repo.
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM-$RANDOM"
OWN_OUTPUT="cd out && [ \"\$(cat LAUNCH_ID)\" = '$LAUNCH_ID' ]"
printf '%s\n' "$LAUNCH_ID" > "$RESULTS/launch-id.txt"
if [[ ${FORWARD_JUDGE_KEY:-} == 1 && $REMOTE == ops/remote_runpod_smoke.sh && $JUDGE_KIND == api ]]; then
  printf '%s' "${OPENAI_API_KEY:?FORWARD_JUDGE_KEY=1 needs OPENAI_API_KEY}" | on_pod "umask 077; cat > ~/.judge_key.$LAUNCH_ID"
fi
# The values the pod gets for settings left unset: the documented defaults the checks above validated.
if [[ -n ${JUDGE_KIND:-} ]]; then SENT_JUDGE_KIND=$JUDGE_KIND; else SENT_JUDGE_KIND=api; fi
if [[ -n ${LEAN_ISOLATION:-} ]]; then SENT_ISOLATION=$LEAN_ISOLATION; else SENT_ISOLATION=required; fi
if [[ -n ${JUDGE_MAX_USD:-} ]]; then SENT_JUDGE_MAX_USD=$JUDGE_MAX_USD; else SENT_JUDGE_MAX_USD=1; fi
if [[ -n ${AUTOSTOP_GRACE_SECONDS:-} ]]; then SENT_GRACE=$AUTOSTOP_GRACE_SECONDS; else SENT_GRACE=600; fi
LAUNCH_SENT=1
if ! on_pod "JUDGE_KIND=$SENT_JUDGE_KIND LEAN_ISOLATION=$SENT_ISOLATION JUDGE_MAX_USD=$SENT_JUDGE_MAX_USD REQUIRE_AUTOSTOP=1 AUTOSTOP_GRACE_SECONDS=$SENT_GRACE ${SMOKE_ENV}nohup bash pilot-repo/ops/remote_job.sh --launch $LAUNCH_ID pilot-repo/$REMOTE $((DEADLINE*60)) > launch.$LAUNCH_ID.log 2>&1 < /dev/null &"; then
  say 'launch SSH failed; checking the launch receipt on the pod'
fi
# An SSH success only acknowledges a background shell. Always wait for the detached transaction
# to stop the old writer, prepare a clean directory and record this launch's exact identity.
# cap: 2.5x estimated 18 s (stopping an earlier job and preparing its replacement)
RECEIPT_WAIT_SECONDS=45
for attempt in 1 2; do
  if receipt=$(on_pod "for i in \$(seq $RECEIPT_WAIT_SECONDS); do if [ -r out/LAUNCH_ID ] && [ \"\$(cat out/LAUNCH_ID)\" = '$LAUNCH_ID' ]; then echo '$LAUNCH_ID'; exit 0; fi; sleep 1; done; exit 1" | tail -n 1) && [[ $receipt == "$LAUNCH_ID" ]]; then
    LAUNCHED=1
    break
  fi
done
[[ $LAUNCHED == 1 ]] || { echo "launch receipt unconfirmed; inspect launch.$LAUNCH_ID.log on the pod"; exit 1; }
say 'launch receipt confirmed: following this job'
date -u +%FT%TZ > "$RESULTS/started-at.txt"
END=$((SECONDS+DEADLINE*60+90))  # the remote timeout is DEADLINE; the extra time is for it to write EXIT_CODE
while ((SECONDS < END)); do
  sleep 60
  if code=$(on_pod "if $OWN_OUTPUT; then cat EXIT_CODE; else echo ownership-lost; fi" 2>> "$RESULTS/ssh-poll.log"); then
    if [[ $code == ownership-lost ]]; then
      LAUNCHED=0
      echo 'the output directory no longer belongs to this launch; stopping without copying or removing the pod'
      exit 1
    fi
    [[ $code =~ ^[0-9]+$ && $code -le 255 ]] || { echo 'invalid remote exit status'; exit 1; }
    echo "remote finished with exit=$code"
    exit "$code"
  fi
  if ((SECONDS % 300 < 60)) && ! pull "${ADAPTERS[@]}"; then echo 'interim copy failed; will try again'; fi
done
echo 'driver deadline reached; the remote timeout and the pod removal timer are armed'
exit 1
