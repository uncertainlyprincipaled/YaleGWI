#!/bin/bash
set -e

# Load environment variables
source .env/aws/credentials

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=gwi-training" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "No running training instance found"
    exit 1
fi

# Get instance status
aws ec2 describe-instance-status \
    --instance-ids $INSTANCE_ID \
    --include-all-instances

# Get CloudWatch metrics
aws cloudwatch get-metric-statistics \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value=$INSTANCE_ID \
    --start-time $(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --period 300 \
    --statistics Average

# Check S3 for latest checkpoint
aws s3 ls s3://$S3_BUCKET/checkpoints/ --recursive | sort | tail -n 5

# Check training logs
aws s3 ls s3://$S3_BUCKET/logs/ --recursive | sort | tail -n 5 