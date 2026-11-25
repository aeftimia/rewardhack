#!/usr/bin/env bash
# Launch, run a remote script, copy results, terminate, verify. Termination runs on any exit, and the
# OS timer set at boot still terminates the instance if this machine dies. One exception: if the final copy
# fails while the instance is still up, it is left to that timer (it holds the only copy) and this exits 96.
# Exit 97: the teardown check did not succeed.
# Usage: drive_run.sh <remote script> <results dir> <deadline minutes> <launch_aws.sh args, incl. --approved>
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=$1 RESULTS=$2 DEADLINE=$3; shift 3
REGION=us-west-2
mkdir -p "$RESULTS"
OUTPUT=$(bash ops/launch_aws.sh "$@") || { echo "$OUTPUT"; exit 1; }
echo "$OUTPUT"
LAST=$(echo "$OUTPUT" | tail -1); ID=${LAST%% *}; IP=${LAST##* }
[[ $ID == i-* ]] || { echo 'no instance launched'; exit 1; }
finish() {
  code=$?
  if [[ $JOB_STARTED == 0 ]]; then
    echo 'the remote script was never started, so there are no results to copy'; COPIED=1
  else
    echo "copying results"
    COPIED=0
    for attempt in 1 2 3; do
      if pull; then COPIED=1; break; fi
      echo "copy attempt $attempt failed"; sleep 10
    done
  fi
  aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].[InstanceType,LaunchTime,InstanceLifecycle]' --output text > "$RESULTS/instance.txt"
  if [[ $COPIED == 0 ]]; then
    # As in drive_runpod.sh: a machine holding the only copy of the results is not terminated from here. Only an instance that is
    # known to be gone already has nothing left to keep; a state that could not be read counts as still there.
    GONE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
    if [[ $GONE != terminated && $GONE != shutting-down ]]; then
      echo "FINAL COPY FAILED three times: the results are only on instance $ID (state: '$GONE'), so it is NOT terminated from here. Its OS shutdown timer, set at boot, still ends it. Copy out/ from ec2-user@$IP before then, then run: bash ops/teardown_check.sh $REGION"
      if [[ $code == 0 ]]; then code=96; fi
      exit "$code"
    fi
    echo "FINAL COPY FAILED three times and instance $ID is already $GONE: whatever was not copied earlier is lost"
    if [[ $code == 0 ]]; then code=96; fi
  fi
  bash ops/teardown_check.sh "$REGION" | tee "$RESULTS/teardown.txt"
  TEARDOWN=${PIPESTATUS[0]}
  date -u +%FT%TZ > "$RESULTS/terminated-at.txt"
  aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].[State.Name,StateTransitionReason]' --output text | tee -a "$RESULTS/teardown.txt"
  if [[ $TEARDOWN != 0 ]]; then
    echo "TEARDOWN DID NOT SUCCEED (ops/teardown_check.sh exit $TEARDOWN): something billable may be left in $REGION. Check the EC2 console now. $RESULTS/teardown.txt has what it found."
    code=97
  fi
  exit "$code"
}
SSH=(ssh -i .tools/rewardhack.pem -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=.tools/known_hosts -o ConnectTimeout=10)
pull() { rsync -az --partial --exclude trainer -e "${SSH[*]}" "ec2-user@$IP:out/" "$RESULTS/"; }
JOB_STARTED=0  # until the remote script is running there is nothing on the instance worth keeping it for
trap finish EXIT
for _ in $(seq 40); do "${SSH[@]}" "ec2-user@$IP" true 2>/dev/null && break; sleep 15; done
rsync -az --exclude .tools --exclude spending --exclude results -e "${SSH[*]}" ./ "ec2-user@$IP:pilot-repo/"
SYNCED=$?
# This script has no `set -e`: a failed sync went unnoticed and the driver then waited the whole deadline for a job that never started.
[[ $SYNCED == 0 ]] || { echo "copying the repo to the instance failed (rsync exit $SYNCED); stopping, and the exit handler terminates the instance"; exit 1; }
"${SSH[@]}" "ec2-user@$IP" "sudo dnf install -y -q zstd rsync >/dev/null 2>&1; nohup bash pilot-repo/$REMOTE > remote.log 2>&1 < /dev/null & disown"
STARTED=$?
[[ $STARTED == 0 ]] || { echo "starting the remote script failed (ssh exit $STARTED); stopping, and the exit handler terminates the instance"; exit 1; }
JOB_STARTED=1
END=$((SECONDS+DEADLINE*60))
while ((SECONDS < END)); do
  sleep 60
  "${SSH[@]}" "ec2-user@$IP" 'test -f out/DONE' 2>/dev/null && { echo 'remote finished'; exit 0; }
  ((SECONDS % 600 < 60)) && pull
  STATE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
  [[ $STATE == running ]] || { echo "instance is $STATE"; exit 1; }
done
echo 'driver deadline reached'
