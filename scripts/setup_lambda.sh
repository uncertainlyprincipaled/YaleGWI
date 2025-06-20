#!/bin/bash

# Lambda Labs Setup Script for YaleGWI Project
# This script sets up the environment for training on Lambda Labs

set -e  # Exit on any error

echo "🚀 Setting up Lambda Labs environment for YaleGWI..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y git wget curl unzip

# Install CUDA and PyTorch (adjust versions as needed)
echo "🔧 Installing CUDA and PyTorch..."
# Note: Lambda Labs typically has CUDA pre-installed
# Check CUDA version
nvidia-smi
nvcc --version

# Clone repository
echo "📥 Cloning YaleGWI repository..."
if [ ! -d "YaleGWI" ]; then
    git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
fi
cd YaleGWI

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "📦 Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional packages for Lambda Labs
echo "📦 Installing Lambda Labs specific packages..."
pip install zarr dask scipy scikit-image mlflow

# Setup AWS credentials (if using S3)
echo "🔑 Setting up AWS credentials..."
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "AWS credentials found in environment variables"
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
    aws configure set default.region us-east-1
else
    echo "⚠️ AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "You can still use local data or Google Drive"
fi

# Create data directories
echo "📁 Creating data directories..."
# On Lambda Labs, /data might not exist or not be writable by the default user.
# We will create it if it doesn't exist and ensure the current user owns it.
if [ ! -d "/data" ]; then
    sudo mkdir -p /data
fi
sudo chown -R $(whoami):$(whoami) /data

mkdir -p /data/preprocessed
mkdir -p /data/raw
mkdir -p outputs
mkdir -p checkpoints

# Download preprocessed data from S3 (if available)
echo "📥 Downloading preprocessed data from S3..."
if command -v aws &> /dev/null; then
    S3_BUCKET=${S3_BUCKET:-"yale-gwi"}
    echo "Attempting to download from s3://$S3_BUCKET/preprocessed/"
    
    if aws s3 ls s3://$S3_BUCKET/preprocessed/ &> /dev/null; then
        aws s3 sync s3://$S3_BUCKET/preprocessed/ /data/preprocessed/
        echo "✅ Preprocessed data downloaded from S3"
    else
        echo "⚠️ Preprocessed data not found in S3 bucket: $S3_BUCKET"
    fi
else
    echo "⚠️ AWS CLI not available, skipping S3 download"
fi

# Test GPU availability
echo "🖥️ Testing GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

# Test Phase 1 components
echo "🧪 Testing Phase 1 components..."
python3 colab_test_setup.py

# Create training script
echo "📝 Creating training script..."
cat > train_lambda.sh << 'EOF'
#!/bin/bash

# Lambda Labs Training Script
# Usage: ./train_lambda.sh [options]

set -e

# Default parameters
EPOCHS=30
BATCH_SIZE=8
USE_AMP=true
USE_S3=false
FAMILY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-amp)
            USE_AMP=false
            shift
            ;;
        --use-s3)
            USE_S3=true
            shift
            ;;
        --family)
            FAMILY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment
source venv/bin/activate

# Build command
CMD="python src/core/train_lambda.py --epochs $EPOCHS --batch $BATCH_SIZE"

if [ "$USE_AMP" = true ]; then
    CMD="$CMD --amp"
fi

if [ "$USE_S3" = true ]; then
    CMD="$CMD --use_s3"
fi

if [ -n "$FAMILY" ]; then
    CMD="$CMD --family $FAMILY"
fi

echo "🚀 Starting training with command: $CMD"
echo "📊 Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Mixed precision: $USE_AMP"
echo "  Use S3: $USE_S3"
echo "  Family: ${FAMILY:-'all'}"

# Run training
$CMD

echo "✅ Training completed!"
EOF

chmod +x train_lambda.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > monitor_training.sh << 'EOF'
#!/bin/bash

# Training monitoring script
# Usage: ./monitor_training.sh

echo "🖥️ System Resources:"
echo "==================="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "💾 Disk Usage:"
echo "=============="
df -h /data /tmp

echo ""
echo "📁 Output Files:"
echo "================"
ls -la outputs/ 2>/dev/null || echo "No outputs directory found"
ls -la checkpoints/ 2>/dev/null || echo "No checkpoints directory found"

echo ""
echo "📊 Recent Logs:"
echo "==============="
tail -n 20 training.log 2>/dev/null || echo "No training.log found"
EOF

chmod +x monitor_training.sh

echo ""
echo "🎉 Lambda Labs setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Test setup: python colab_test_setup.py"
echo "3. Start training: ./train_lambda.sh --epochs 30 --batch 8 --amp"
echo "4. Monitor training: ./monitor_training.sh"
echo ""
echo "📊 Available training options:"
echo "  --epochs N        Number of epochs (default: 30)"
echo "  --batch N         Batch size (default: 8)"
echo "  --no-amp          Disable mixed precision"
echo "  --use-s3          Use S3 for data loading"
echo "  --family NAME     Train on specific family only"
echo ""
echo "🔧 Environment variables:"
echo "  AWS_ACCESS_KEY_ID     AWS access key for S3"
echo "  AWS_SECRET_ACCESS_KEY AWS secret key for S3"
echo "  S3_BUCKET             S3 bucket name (default: yale-gwi)"
EOF 