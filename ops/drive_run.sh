#!/usr/bin/env bash
# Launch, run a remote script, copy results, terminate, verify. Termination runs on any exit, and the
# OS timer set at boot still terminates the instance if this machine dies.
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
  echo "copying results"; pull
  aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].[InstanceType,LaunchTime,InstanceLifecycle]' --output text > "$RESULTS/instance.txt"
  bash ops/teardown_check.sh "$REGION" | tee "$RESULTS/teardown.txt"
  date -u +%FT%TZ > "$RESULTS/terminated-at.txt"
  aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].[State.Name,StateTransitionReason]' --output text | tee -a "$RESULTS/teardown.txt"
}
SSH=(ssh -i .tools/rewardhack.pem -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=.tools/known_hosts -o ConnectTimeout=10)
pull() { rsync -az --partial --exclude trainer -e "${SSH[*]}" "ec2-user@$IP:out/" "$RESULTS/"; }
trap finish EXIT
for _ in $(seq 40); do "${SSH[@]}" "ec2-user@$IP" true 2>/dev/null && break; sleep 15; done
rsync -az --exclude .tools --exclude spending --exclude results -e "${SSH[*]}" ./ "ec2-user@$IP:pilot-repo/"
"${SSH[@]}" "ec2-user@$IP" "sudo dnf install -y -q zstd rsync >/dev/null 2>&1; nohup bash pilot-repo/$REMOTE > remote.log 2>&1 < /dev/null & disown"
END=$((SECONDS+DEADLINE*60))
while ((SECONDS < END)); do
  sleep 60
  "${SSH[@]}" "ec2-user@$IP" 'test -f out/DONE' 2>/dev/null && { echo 'remote finished'; exit 0; }
  ((SECONDS % 600 < 60)) && pull
  STATE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
  [[ $STATE == running ]] || { echo "instance is $STATE"; exit 1; }
done
echo 'driver deadline reached'
