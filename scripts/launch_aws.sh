#!/bin/bash
set -e

# Load environment variables
source .env/aws/credentials

# Launch spot instance
aws ec2 run-instances \
    --instance-type g5.2xlarge \
    --image-id ami-0dfae3c90574ae005 \
    --instance-market-options "MarketType=spot,SpotOptions={SpotInstanceType='one-time',InstanceInterruptionBehavior='stop'}" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --user-data file://scripts/bootstrap_aws.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=gwi-training}]' \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx \
    --iam-instance-profile Name=gwi-training-role

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=gwi-training" "Name=instance-state-name,Values=pending" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

echo "Launched instance: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance is running at $PUBLIC_IP"
echo "SSH command: ssh -i your-key.pem ubuntu@$PUBLIC_IP" 