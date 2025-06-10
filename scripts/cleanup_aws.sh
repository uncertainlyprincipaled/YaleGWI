#!/bin/bash
set -e

# Load environment variables
source .env/aws/credentials

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=gwi-training" "Name=instance-state-name,Values=running" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

if [ ! -z "$INSTANCE_ID" ]; then
    echo "Terminating instance: $INSTANCE_ID"
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID
fi

# Clean up old checkpoints (keep last 5)
aws s3 ls s3://$S3_BUCKET/checkpoints/ --recursive | sort -r | tail -n +6 | awk '{print $4}' | while read -r file; do
    aws s3 rm "s3://$S3_BUCKET/$file"
done

# Clean up old logs (keep last 5)
aws s3 ls s3://$S3_BUCKET/logs/ --recursive | sort -r | tail -n +6 | awk '{print $4}' | while read -r file; do
    aws s3 rm "s3://$S3_BUCKET/$file"
done

echo "Cleanup complete" 