#!/usr/bin/env bash
# Terminate this project's instances, then confirm nothing billable is left. Exit 1 if anything remains.
set -euo pipefail
REGION=${1:-us-west-2}
FILTER=Name=tag:project,Values=rewardhack
IDS=$(aws ec2 describe-instances --region "$REGION" --filters "$FILTER" Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
if [[ -n $IDS ]]; then
  # One throttled or dropped call must not leave the instances running: asked up to five times, 10 s apart. Whatever happens here,
  # the counts below are still taken and decide the exit status.
  ACCEPTED=0
  for attempt in 1 2 3 4 5; do
    # shellcheck disable=SC2086
    if aws ec2 terminate-instances --region "$REGION" --instance-ids $IDS >/dev/null; then ACCEPTED=1; break; fi
    echo "terminate-instances attempt $attempt of 5 failed for: $IDS"
    sleep 10
  done
  if [[ $ACCEPTED == 0 ]]; then echo "TERMINATION WAS NOT ACCEPTED in 5 attempts; still running and billing: $IDS"
  else
    # shellcheck disable=SC2086
    if ! aws ec2 wait instance-terminated --region "$REGION" --instance-ids $IDS; then echo "waiting for termination failed; the counts below say what is left: $IDS"; fi
  fi
fi
LIVE=$(aws ec2 describe-instances --region "$REGION" --filters Name=instance-state-name,Values=pending,running,stopping,stopped --query 'length(Reservations[].Instances[])' --output text)
VOLUMES=$(aws ec2 describe-volumes --region "$REGION" --query 'length(Volumes[])' --output text)
SNAPSHOTS=$(aws ec2 describe-snapshots --region "$REGION" --owner-ids self --query 'length(Snapshots[])' --output text)
ADDRESSES=$(aws ec2 describe-addresses --region "$REGION" --query 'length(Addresses[])' --output text)
echo "$REGION: instances=$LIVE volumes=$VOLUMES snapshots=$SNAPSHOTS elastic_ips=$ADDRESSES"
[[ $LIVE == 0 && $VOLUMES == 0 && $SNAPSHOTS == 0 && $ADDRESSES == 0 ]]
