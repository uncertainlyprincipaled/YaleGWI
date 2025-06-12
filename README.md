# SpecProj-UNet for Seismic Waveform Inversion

## Quick Start: AWS/EC2 Training

### 1. Launch a New EC2 Instance
- Use a GPU-enabled instance (e.g., `g5.xlarge`)
- Use Ubuntu 22.04 or 20.04 AMI
- Attach or create a 200GB+ EBS volume for data

### 2. SSH into the Instance
```bash
ssh -i <your-key>.pem ubuntu@<instance-ip>
```

### 3. Clone the Repository
```bash
git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
cd YaleGWI
```

### 4. Set Up Python Environment
If `venv` is not available, install it:
```bash
sudo apt-get update && sudo apt-get install -y python3-venv
```
Then create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure AWS Credentials
Create a file `.env/aws/credentials` with:
```bash
export AWS_ACCESS_KEY_ID=<your_access_key>
export AWS_SECRET_ACCESS_KEY=<your_secret_key>
export AWS_REGION=<your_region>
export S3_BUCKET=<your_bucket_name>
```
Then load them:
```bash
source .env/aws/credentials
```

### 6. Download/Sync Data from S3
```bash
python -m src.core.setup
# Or, manually:
# aws s3 sync s3://<your-bucket>/raw/ /mnt/waveform-inversion
```

> **Troubleshooting:**
> If you get a `[Errno 13] Permission denied: '/mnt/waveform-inversion'` error, run:
> ```bash
> sudo mkdir -p /mnt/waveform-inversion
> sudo chown $USER:$USER /mnt/waveform-inversion
> ```
> Then re-run the sync command.

### 7. Run Training
```bash
python src/core/train.py --epochs 120 --batch 16 --amp --experiment-tag "run-$(date +%F)"
```

---

# SpecProj-UNet for Seismic Waveform Inversion

## Data IO Policy (IMPORTANT)
**All data loading, streaming, and batching must go through `DataManager` in `src/core/data_manager.py`.**
- Do NOT load data directly in any other file.
- This ensures memory efficiency and prevents RAM spikes in Kaggle.
- All source files must import and use DataManager for any data access.

## Overview
A deep learning solution for seismic waveform inversion using a HGNet/ConvNeXt backbone with:
- EMA (Exponential Moving Average) for stable training
- AMP (Automatic Mixed Precision) for faster training
- Distributed Data Parallel (DDP) for multi-GPU training
- Test Time Augmentation (TTA) for improved inference

## Quick Start

### Local Development Setup
1. Clone the repository:
```bash
git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
cd YaleGWI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Development Workflow
1. **All data IO must use DataManager.**
2. Develop and test your code locally in `kaggle_notebook.py`
3. Copy relevant code chunks to Kaggle notebook cells as needed
4. Test functionality in Kaggle environment
5. Once everything works, copy the entire `kaggle_notebook.py` into a single Kaggle cell for final submission

### Kaggle Setup
1. Create a new notebook in the [Waveform Inversion competition](https://www.kaggle.com/competitions/waveform-inversion)

2. **Important**: Add the required datasets to your notebook first:
   - Click on the 'Data' tab
   - Click 'Add Data'
   - Search for and add:
     1. 'Waveform Inversion' competition dataset
     2. 'openfwi-preprocessed-72x72' dataset (contains preprocessed data and pretrained models)

3. Copy code chunks from `kaggle_notebook.py` into separate cells in your Kaggle notebook for testing

4. For final submission, copy the entire contents of `kaggle_notebook.py` into a single cell

## Code Structure
- `kaggle_notebook.py`: Main development file containing all code
- `src/core/data_manager.py`: **Single source of truth for all data IO**
- `requirements.txt`: Project dependencies

## TODO
- [ ] Add proper error handling for Kaggle environment detection
- [ ] Add data validation checks
- [ ] Add model checkpointing
- [ ] Add logging functionality
- [ ] Add visualization utilities
- [ ] Add test suite

## Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```

## Features

- **Physics-Guided Architecture**: Uses spectral projectors to inject wave-equation priors
- **Mixed Precision Training**: Supports both fp16 and bfloat16 for efficient training
- **Memory Optimization**: Implements efficient attention and memory management
- **Family-Aware Training**: Stratified sampling by geological families
- **Robust Training**: Includes gradient clipping, early stopping, and learning rate scheduling
- **Comprehensive Checkpointing**: Saves full training state for easy resumption

## Project Structure

```
.
├── src/
│   ├── core/
│   │   ├── config.py      # Configuration management
│   │   ├── train.py       # Training loop implementation
│   │   ├── model.py       # Model architecture
│   │   └── losses.py      # Loss functions
│   └── utils/
│       ├── data.py        # Data loading utilities
│       └── metrics.py     # Evaluation metrics
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
└── outputs/              # Training outputs and checkpoints
```

## Usage

### Training

```python
from src.core.train import train

# Start training with default settings
train(fp16=True)  # Enable mixed precision training
```

### Inference

```python
from src.core.model import get_model
import torch

# Load model and weights
model = get_model()
model.load_state_dict(torch.load('outputs/best.pth'))

# Run inference
predictions = model(input_data)
```

## Key Components

### Model Architecture
- SpecProj-UNet: Combines UNet with spectral projectors for physics-guided learning
- EMA: Exponential Moving Average for model weights
- Mixed Precision: Automatic mixed precision training support

### Training Features
- Gradient Clipping: Prevents exploding gradients
- Early Stopping: Stops training when validation performance plateaus
- Learning Rate Scheduling: Adaptive learning rate adjustment
- Memory Management: Efficient GPU memory usage
- Comprehensive Checkpointing: Saves full training state

### Loss Functions
- L1 Loss: Basic reconstruction loss
- PDE Residual: Physics-based consistency term
- Joint Loss: Combines multiple loss components

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
We use black for code formatting and flake8 for linting:
```bash
black src/
flake8 src/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Kaggle competition "Seismic Waveform Inversion"
- Inspired by various physics-guided deep learning approaches
- Uses PyTorch for deep learning implementation

## Remote Training on AWS

### Prerequisites
1. AWS Account with appropriate permissions:
   - EC2 Full Access
   - S3 Full Access
   - IAM Role creation permissions

2. Required AWS Resources:
   - S3 Bucket (create your own)
   - IAM Role for training
   - Security Group with SSH access
   - VPC Subnet

### Instance Requirements
- Type: `g5.2xlarge` (GPU-enabled)
   - 8 vCPUs
   - 32 GB RAM
   - 1 NVIDIA A10G GPU
   - 200 GB GP3 EBS volume
- AMI: Ubuntu 20.04 (use latest compatible AMI)
- Spot Instance for cost efficiency

### Setup Instructions

1. **Configure AWS Credentials**
```bash
# Create .env/aws/credentials file
mkdir -p .env/aws
cat > .env/aws/credentials << EOL
export AWS_ACCESS_KEY_ID=<your_access_key>
export AWS_SECRET_ACCESS_KEY=<your_secret_key>
export AWS_REGION=<your_region>
export S3_BUCKET=<your_bucket_name>
EOL
```

2. **Launch Training Instance**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Launch instance
./scripts/launch_aws.sh
```

3. **Monitor Training**
```bash
# SSH into instance
ssh -i <your-key>.pem ubuntu@<instance-ip>

# Monitor training progress
tail -f /var/log/cloud-init-output.log

# Check GPU status
nvidia-smi

# Monitor disk usage
df -h
```

4. **Cleanup**
```bash
# Terminate instance and clean up old checkpoints
./scripts/cleanup_aws.sh
```

### S3 Bucket Structure

```
<your-bucket-name>/
├── raw/           # Original data (untouched)
├── checkpoints/   # Model checkpoints
└── logs/         # Training logs
```

### Checkpoint Management
- Local storage: Keeps last 3 checkpoints

### Daily Workflow
1. **Start Training Session**
   ```bash
   # SSH into instance
   ssh -i <your-key>.pem ubuntu@<instance-ip>
   
   # Activate virtual environment
   source venv/bin/activate
   
   # Download data (if needed)
   python scripts/download_data.py
   
   # Run setup
   python scripts/setup.py
   
   # Start training
   python src/core/train.py
   ```

2. **End Training Session**
   ```bash
   # Terminate instance
   aws ec2 terminate-instances --instance-ids <instance-id>
   
   # Verify instance is terminated
   aws ec2 describe-instances --filters "Name=instance-state-name,Values=running"
   ```

Note: Data on the EBS volume persists between instance launches, so you only need to download data once unless you explicitly delete it. The setup script handles environment configuration and dependencies.

### Debugging Remote Training

1. **Common Issues**:
   - Out of Memory (OOM)
     - Automatic batch size reduction
     - GPU memory fraction limiting (80%)
     - Periodic cache clearing
   
   - Spot Instance Interruption
     - Automatic checkpoint saving
     - Training resumption from last checkpoint
   
   - Data Loading Issues
     - Use `probe_s3_bucket.py` to verify access
     - Check S3 permissions

2. **Monitoring Tools**:
   - CloudWatch metrics
   - Instance system logs
   - Training logs in S3
   - GPU utilization monitoring

3. **Best Practices**:
   - Use `tmux` or `screen` for persistent sessions
   - Regular log monitoring
   - Periodic checkpoint verification
   - Disk space monitoring

### Cost Optimization
- Use spot instances for training
- Automatic instance termination after training
- Regular cleanup of old checkpoints
- Efficient data loading through DataManager

### Security Considerations
- IAM roles for instance permissions
- Secure credential management
- Private subnet deployment
- Regular security group audits



