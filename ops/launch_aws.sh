#!/usr/bin/env bash
# Launch one self-terminating GPU instance. Prints the plan; does nothing paid without --approved.
# Protections: budget guard, spot price ceiling, OS shutdown timer set at boot, terminate-on-shutdown,
# root volume deleted on termination, SSH open only to this machine's IP, IMDS limited to the host.
set -euo pipefail
cd "$(dirname "$0")/.."
REGION=us-west-2 TYPE= MARKET=on-demand MAX_PRICE= HOURS= MODEL= HOURLY= DISK=80 APPROVED=0 ZONE=
while (($#)); do case "$1" in
  --region) REGION=$2; shift 2;; --type) TYPE=$2; shift 2;; --market) MARKET=$2; shift 2;; --zone) ZONE=$2; shift 2;;
  --max-price) MAX_PRICE=$2; shift 2;; --hours) HOURS=$2; shift 2;; --model) MODEL=$2; shift 2;;
  --hourly) HOURLY=$2; shift 2;; --disk) DISK=$2; shift 2;; --approved) APPROVED=1; shift;;
  *) echo "unknown $1"; exit 2;; esac; done
[[ -n $TYPE && -n $HOURS && -n $MODEL && -n $HOURLY ]] || { echo 'need --type --hours --model --hourly'; exit 2; }
PROFILE_ACCOUNT=428141338262
[[ $(aws sts get-caller-identity --query Account --output text) == "$PROFILE_ACCOUNT" ]] || { echo 'wrong AWS account'; exit 1; }
# gp3 at about $0.08/GB-month; $0.25 allowance for data transfer and shutdown overhead.
OTHER=$(python3 -c "print(round($DISK*0.08/730*$HOURS+0.25,2))")
python3 entrypoints/budget.py check --provider aws --model "$MODEL" --hourly "$HOURLY" --max-hours "$HOURS" --other-usd "$OTHER"
MINUTES=$(python3 -c "print(int($HOURS*60))")
AMI=$(aws ssm get-parameter --region "$REGION" --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id --query Parameter.Value --output text)
USER_DATA=$(printf '#!/bin/bash\nshutdown -h +%s "budget cap"\n' "$MINUTES")
TAGS="ResourceType=instance,Tags=[{Key=project,Value=rewardhack}]"
MARKET_ARGS=()
if [[ $MARKET == spot ]]; then
  [[ -n $MAX_PRICE ]] || { echo 'spot needs --max-price'; exit 2; }
  MARKET_ARGS=(--instance-market-options "MarketType=spot,SpotOptions={MaxPrice=$MAX_PRICE,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}")
fi
COMMON=(--region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --count 1
        --instance-initiated-shutdown-behavior terminate --user-data "$USER_DATA"
        --metadata-options HttpTokens=required,HttpPutResponseHopLimit=1
        --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$DISK,VolumeType=gp3,DeleteOnTermination=true}"
        --tag-specifications "$TAGS" ${MARKET_ARGS[@]+"${MARKET_ARGS[@]}"})
# A zone pins the request when the region-wide request fails with InsufficientInstanceCapacity.
[[ -z $ZONE ]] || COMMON+=(--placement "AvailabilityZone=$ZONE")
echo "plan: $TYPE $MARKET in ${ZONE:-$REGION}, AMI $AMI, ${HOURS}h OS shutdown timer, ${DISK}GB deleted on termination"
if ((APPROVED == 0)); then
  # Free permission check. "DryRunOperation" means the request would have been accepted.
  aws ec2 run-instances "${COMMON[@]}" --dry-run 2>&1 | tail -1 || true
  echo 'not launched: rerun with --approved after explicit approval'
  exit 0
fi
KEY=rewardhack; KEY_FILE=.tools/$KEY.pem
if [[ ! -f $KEY_FILE ]]; then
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY" --query KeyMaterial --output text > "$KEY_FILE"; chmod 600 "$KEY_FILE"
fi
SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=$KEY --query 'SecurityGroups[0].GroupId' --output text)
if [[ $SG == None ]]; then
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name "$KEY" --description 'rewardhack SSH' --query GroupId --output text)
fi
MYIP=$(curl -fsS https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null 2>&1 || true
ID=$(aws ec2 run-instances "${COMMON[@]}" --key-name "$KEY" --security-group-ids "$SG" --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
date -u +%FT%TZ > ".tools/$ID.started"
echo "$ID $IP"
