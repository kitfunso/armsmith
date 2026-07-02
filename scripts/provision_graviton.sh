#!/usr/bin/env bash
# scripts/provision_graviton.sh
#
# Stand up a single Graviton (r8g) target for armsmith, with an auto-shutdown
# backstop (CLAUDE.md rule 8: tear down after each session). Mirrors the exact
# aws cli sequence recorded in docs/spike0-result.md "Target-side checks"
# (2026-07-02): key pair -> security group -> run-instances -> wait -> IP.
#
# Usage:
#   ARMSMITH_HOME_IP=203.0.113.7/32 bash scripts/provision_graviton.sh
#
# All other settings have defaults matching the spike0 target; override via env.
set -euo pipefail

INSTANCE_TYPE="${ARMSMITH_INSTANCE_TYPE:-r8g.4xlarge}"
REGION="${ARMSMITH_REGION:-eu-west-2}"
AZ="${ARMSMITH_AZ:-eu-west-2c}"
AMI_ID="${ARMSMITH_AMI_ID:-ami-05dcc391311f872c0}"   # Ubuntu 24.04 arm64, eu-west-2 (spike0)
KEY_NAME="${ARMSMITH_KEY_NAME:-armsmith}"
KEY_PATH="${ARMSMITH_KEY_PATH:-$HOME/.ssh/${KEY_NAME}.pem}"
SG_NAME="${ARMSMITH_SG_NAME:-armsmith-sg}"
VOLUME_SIZE_GB="${ARMSMITH_VOLUME_SIZE_GB:-64}"
AUTO_SHUTDOWN_MINUTES="${ARMSMITH_AUTO_SHUTDOWN_MINUTES:-180}"
HOME_IP="${ARMSMITH_HOME_IP:?set ARMSMITH_HOME_IP=<your-public-ip>/32 (SSH source CIDR)}"

echo "== 1. key pair =="
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "key pair $KEY_NAME already exists, reusing"
else
  aws ec2 create-key-pair \
    --key-name "$KEY_NAME" \
    --region "$REGION" \
    --query 'KeyMaterial' \
    --output text > "$KEY_PATH"
  chmod 400 "$KEY_PATH"
  echo "created key pair $KEY_NAME -> $KEY_PATH"
fi

echo "== 2. security group (SSH from ${HOME_IP} only) =="
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
  --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "armsmith Graviton target: SSH from home IP only" \
    --vpc-id "$VPC_ID" \
    --region "$REGION" \
    --query 'GroupId' --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "$HOME_IP" \
    --region "$REGION"
  echo "created security group $SG_ID"
else
  echo "security group $SG_NAME already exists ($SG_ID), reusing"
fi

echo "== 3. run-instances (auto-shutdown backstop: shutdown -h +${AUTO_SHUTDOWN_MINUTES}) =="
# Pass the RAW cloud-init script: `aws ec2 run-instances --user-data` base64-
# encodes the value for you, so pre-encoding it double-encodes and cloud-init
# never runs the shutdown backstop (CLAUDE.md rule 8 -- the only cost control).
USER_DATA=$(printf '#!/bin/bash\nshutdown -h +%s\n' "$AUTO_SHUTDOWN_MINUTES")
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --placement AvailabilityZone="$AZ" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE_GB},\"VolumeType\":\"gp3\"}}]" \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=armsmith}]" \
  --region "$REGION" \
  --query 'Instances[0].InstanceId' --output text)
echo "launched $INSTANCE_ID"

echo "== 4. wait until running =="
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

echo "== 5. public IP =="
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo
echo "instance_id=$INSTANCE_ID"
echo "public_ip=$PUBLIC_IP"
echo "ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
echo
echo "Remember: CLAUDE.md rule 8 -- tear this instance down after the session:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
