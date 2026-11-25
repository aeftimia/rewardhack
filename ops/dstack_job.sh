#!/usr/bin/env bash
# The command of a dstack task (src/rewardhack/dstack_jobs.py): on the pod, what ops/drive_runpod.sh does there over SSH,
# then ops/remote_job.sh unchanged. A prototype: no paid run has used it yet (docs/notes-2026-11-21.md, section 62).
# dstack removes the pod when this script ends or max_duration runs out, but only while its server runs. The cap that
# holds without that server is remote_job.sh's own: armed here before anything else is done, and again for the job.
# Usage: dstack_job.sh <ops/remote_runpod_*.sh> <deadline seconds>
set -euo pipefail
[[ $# == 2 && $1 =~ ^ops/remote_runpod_[a-z_]+\.sh$ && $2 =~ ^[1-9][0-9]*$ ]] || { echo 'need an ops/remote_runpod_*.sh script and a deadline in whole seconds'; exit 2; }
# Every setting comes from the task configuration by name; none is left to a default here.
for name in JUDGE_KIND LEAN_ISOLATION JUDGE_MAX_USD AUTOSTOP_GRACE_SECONDS RESULTS_MODE RESULTS_HOLD_SECONDS DSTACK_REPO_DIR DSTACK_RUN_NAME; do
  [[ -n ${!name:-} ]] || { echo "$name is not set: start this through entrypoints/dstack_launch.py"; exit 2; }
done
[[ $RESULTS_HOLD_SECONDS =~ ^[0-9]+$ ]] || { echo 'RESULTS_HOLD_SECONDS must be a whole number of seconds'; exit 2; }
# dstack's runner writes every variable of the job, the judge key among them, to /dstack/profile with mode 644 (runner
# executor.go, writeDstackProfile, read 2026-11-25). With LEAN_ISOLATION=none model-written Lean runs as an unprivileged
# user on this pod, so the file is made unreadable to it and the key's line removed before anything else runs.
if [[ -e /dstack/profile ]]; then
  chmod 600 /dstack/profile
  sed -i '/^export OPENAI_API_KEY=/d' /dstack/profile
fi
cd "$DSTACK_REPO_DIR"
[[ -f $1 ]] || { echo "$1 is not in this checkout"; exit 2; }
# The job scripts read ~/pilot-repo and write ~/out, the paths the bash driver gives them.
[[ ! -e ~/pilot-repo && ! -e ~/out ]] || { echo '~/pilot-repo or ~/out already exists: this is not a fresh pod'; exit 2; }
ln -s "$DSTACK_REPO_DIR" ~/pilot-repo
results_volume() {  # a network volume mounted by dstack: the results outlive the pod, whoever removes it
  mountpoint -q /results || { echo 'RESULTS_MODE=volume but nothing is mounted at /results'; exit 2; }
  local folder
  folder="/results/$DSTACK_RUN_NAME-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir "$folder"
  ln -s "$folder" ~/out
}
results_hold() { mkdir ~/out; }  # on the pod's disk, for `dstack attach` and a copy during the hold below
case $RESULTS_MODE in
  volume) results_volume ;;
  hold) results_hold ;;
  *) echo "RESULTS_MODE must be volume or hold, not $RESULTS_MODE"; exit 2 ;;
esac
# The judge key leaves the environment before anything else is started; it is held in an unexported variable until written.
JUDGE_KEY=${OPENAI_API_KEY:-}
unset OPENAI_API_KEY
# A provisional removal timer first, as the bash driver arms over SSH: the job's own timer replaces it when the job starts.
# cap: 3x estimated 5 min (from here to the job's start is seconds; 15 min is what the bash driver arms)
bash ops/remote_job.sh --arm-only 900
# The API judge's key, forwarded only on request, into the mode-600 file the smoke script reads once and deletes.
if [[ $JUDGE_KIND == api && ${FORWARD_JUDGE_KEY:-0} == 1 ]]; then
  [[ -n $JUDGE_KEY ]] || { echo 'FORWARD_JUDGE_KEY=1 but the task carries no OPENAI_API_KEY'; exit 2; }
  (umask 077; printf '%s' "$JUDGE_KEY" > ~/.judge_key)
fi
unset JUDGE_KEY
STATUS=0
REQUIRE_AUTOSTOP=1 bash ops/remote_job.sh "$1" "$2" 2>&1 | tee -a ~/out/remote.log || STATUS=$?
echo "job ended with exit $STATUS"
if ((RESULTS_HOLD_SECONDS > 0)); then
  echo "results are on this pod for ${RESULTS_HOLD_SECONDS}s more, then the pod's own timer removes it: copy them with dstack attach"
  sleep "$RESULTS_HOLD_SECONDS"
fi
exit "$STATUS"
