#!/bin/bash
set -e

# Install system dependencies
sudo apt-get update && sudo apt-get -y install unzip jq awscli git

# Setup AWS credentials
mkdir -p ~/.aws
cat > ~/.aws/credentials << EOL
[default]
aws_access_key_id=${AWS_ACCESS_KEY_ID}
aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}
region=${AWS_REGION}
EOL

# Setup Kaggle credentials
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << EOL
{
  "username": "jdmorgan",
  "key": "${KAGGLE_API_KEY}"
}
EOL
chmod 600 ~/.kaggle/kaggle.json

# Clone repo
git clone https://github.com/uncertainlyprincipaled/gwi.git
cd gwi

# Create and activate conda environment
conda env create -f environment.yml
source activate gwi

# Sync data from S3
aws s3 sync s3://${S3_BUCKET}/raw/ /mnt/waveform-inversion --no-sign-request

# Start training
python src/core/train.py \
    --epochs 120 \
    --batch 16 \
    --amp \
    --experiment-tag "overnight-$(date +%F)"

# Stop instance when done
sudo shutdown -h now 