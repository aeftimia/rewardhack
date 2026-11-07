#!/usr/bin/env bash
# Terminate this project's instances, then confirm nothing billable is left. Exit 1 if anything remains.
set -euo pipefail
REGION=${1:-us-west-2}
FILTER=Name=tag:project,Values=rewardhack
IDS=$(aws ec2 describe-instances --region "$REGION" --filters "$FILTER" Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
if [[ -n $IDS ]]; then
  # shellcheck disable=SC2086
  aws ec2 terminate-instances --region "$REGION" --instance-ids $IDS >/dev/null
  # shellcheck disable=SC2086
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids $IDS
fi
LIVE=$(aws ec2 describe-instances --region "$REGION" --filters Name=instance-state-name,Values=pending,running,stopping,stopped --query 'length(Reservations[].Instances[])' --output text)
VOLUMES=$(aws ec2 describe-volumes --region "$REGION" --query 'length(Volumes[])' --output text)
SNAPSHOTS=$(aws ec2 describe-snapshots --region "$REGION" --owner-ids self --query 'length(Snapshots[])' --output text)
ADDRESSES=$(aws ec2 describe-addresses --region "$REGION" --query 'length(Addresses[])' --output text)
echo "$REGION: instances=$LIVE volumes=$VOLUMES snapshots=$SNAPSHOTS elastic_ips=$ADDRESSES"
[[ $LIVE == 0 && $VOLUMES == 0 && $SNAPSHOTS == 0 && $ADDRESSES == 0 ]]
