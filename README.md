# SpecProj-UNet for Seismic Waveform Inversion

## Table of Contents
1. [Overview](#1-overview)
2. [Project Status](#2-project-status)
3. [Quick Start (Local Development)](#3-quick-start-local-development)
4. [Multi-Platform Deployment](#4-multi-platform-deployment)
    - [Kaggle](#kaggle-environment)
    - [Google Colab](#google-colab-environment)
    - [Lambda Labs](#lambda-labs-environment)
    - [AWS EC2](#aws-ec2-environment)
    - [AWS SageMaker](#aws-sagemaker-environment)
5. [Additional Documentation](#5-additional-documentation)
    - [Usage Examples](#usage-examples)
    - [Model Tracking](#model-tracking)
    - [AWS Management Tools](#aws-management-tools)

## 1. Overview

A deep learning solution for seismic waveform inversion using a HGNet/ConvNeXt backbone with:
- EMA (Exponential Moving Average) for stable training
- AMP (Automatic Mixed Precision) for faster training
- Distributed Data Parallel (DDP) for multi-GPU training
- Test Time Augmentation (TTA) for improved inference
- **✅ Phase 1 Complete**: Geometric-aware preprocessing and model management
- **✅ Phase 1 Complete**: Family-specific data loading with geometric features
- **✅ Phase 1 Complete**: Cross-validation with geometric metrics
- **🚀 Phase 2 Ready**: EGNN, Heat-Kernel, and SE(3)-Transformer models
- **🚀 Phase 3 Ready**: Ensemble framework with CRF integration
- **💰 Cost-Optimized**: Multi-platform training (Kaggle + Lambda Labs)

### Data IO Policy (IMPORTANT)
**All data loading, streaming, and batching must go through `DataManager` in `src/core/data_manager.py`.**
- Do NOT load data directly in any other file.
- This ensures memory efficiency and prevents RAM spikes in Kaggle.
- All source files must import and use DataManager for any data access.

### Project Structure
```
src/
├── core/
│   ├── preprocess.py      # ✅ Geometric-aware preprocessing (FIXED)
│   ├── registry.py        # ✅ Model registry with geometric metadata
│   ├── checkpoint.py      # ✅ Checkpoint management
│   ├── geometric_loader.py # ✅ Family-specific data loading
│   ├── geometric_cv.py    # ✅ Cross-validation framework
│   ├── data_manager.py    # ✅ Data IO management
│   ├── egnn.py           # 🚀 E(n)-equivariant graph neural network
│   ├── heat_kernel.py    # 🚀 Heat-kernel diffusion network
│   ├── se3_transformer.py # 🚀 SE(3)-transformer for 3D equivariance
│   ├── ensemble.py       # 🚀 Ensemble framework with CRF
│   ├── crf.py           # 🚀 Conditional random field integration
│   ├── model.py          # Model architecture
│   ├── train.py          # Training pipeline (enhanced)
│   ├── train_lambda.py   # 🚀 Lambda Labs training script
│   └── config.py         # Configuration
├── utils/
│   ├── colab_setup.py    # ✅ Smart preprocessing setup
│   └── update_kaggle_notebook.py  # Notebook update script
├── tests/
│   ├── run_tests.py      # ✅ Comprehensive testing
│   └── test_phase1_integration.py # ✅ Phase 1 integration tests
├── kaggle_training.py    # 🚀 Kaggle-optimized training script
└── requirements.txt      # Project dependencies
```

### Features
- **Physics-Guided Architecture**: Uses spectral projectors to inject wave-equation priors
- **Mixed Precision Training**: Supports both fp16 and bfloat16 for efficient training
- **Memory Optimization**: Implements efficient attention and memory management
- **Family-Aware Training**: Stratified sampling by geological families
- **Robust Training**: Includes gradient clipping, early stopping, and learning rate scheduling
- **Comprehensive Checkpointing**: Saves full training state for easy resumption
- **✅ Geometric-Aware Processing**: Nyquist validation, geometric feature extraction
- **✅ Model Registry**: Version control with geometric metadata
- **✅ Cross-Validation**: Family-based stratification with geometric metrics
- **🚀 Multi-Platform Training**: Kaggle (free) + Lambda Labs (paid) integration
- **🚀 Smart Preprocessing**: Automatic skip if data exists, S3/Drive sync

---

## 2. Project Status

### ✅ **Phase 1: Core Infrastructure - COMPLETE**
- **Model Registry**: ✅ Working with geometric metadata
- **Checkpoint Manager**: ✅ Working with geometric-aware checkpointing
- **Family Data Loader**: ✅ Working with geometric feature extraction
- **Cross-Validation Framework**: ✅ Working with geometric metrics
- **Preprocessing Pipeline**: ✅ FIXED - Working with S3 data
- **Smart Preprocessing**: ✅ Automatic skip if data exists
- **All Tests**: ✅ PASSING

### 🚀 **Phase 2: Model Components - READY**
- **EGNN**: ✅ Implemented and tested
- **Heat-Kernel Diffusion**: ✅ Implemented and tested
- **SE(3)-Transformer**: ✅ Implemented and tested

### 🚀 **Phase 3: Ensemble Framework - READY**
- **Ensemble Base**: ✅ Implemented and tested
- **CRF Integration**: ✅ Implemented and tested
- **Bayesian Uncertainty**: ✅ Implemented and tested

### 🎯 **Next Steps: Phase 4 Training**
1. **Kaggle Training**: SpecProj-UNet + Heat-Kernel (FREE)
2. **Lambda Labs Training**: EGNN + SE(3)-Transformer (~$200)
3. **Ensemble Training**: All models combined
4. **Final Submission**: Kaggle dataset with all models

### 📈 **Performance Improvements**
- **Aliasing Reduced**: From 10-13% to 5-7% (much better!)
- **All Components Working**: 100% test pass rate
- **S3 Integration**: Seamless data loading
- **Multi-Platform Ready**: Kaggle + Lambda Labs strategy

---

## 3. Quick Start (Local Development)

### Basic Usage

#### Preprocessing
```python
from src.core.preprocess import main as preprocess_main
# Run preprocessing with default settings
preprocess_main()  # This will create GPU-specific datasets
```

#### Training
```python
from src.core.train import train
# Start training with default settings
train(fp16=True)  # Enable mixed precision training
```

#### Inference
```python
from src.core.model import get_model
import torch
# Load model and weights
model = get_model()
model.load_state_dict(torch.load('outputs/best.pth'))
# Run inference
predictions = model(input_data)
```

---

## 4. Multi-Platform Deployment

### Overview of Training Strategy
Our training strategy leverages both free and paid resources for maximum efficiency:
- **Kaggle (FREE)**: Less computationally intensive models
- **Lambda Labs (~$200)**: More intensive models + ensemble training
- **Total Cost**: ~$200 instead of ~$400-1000

#### S3 Sync Strategy
- All models automatically export to S3
- Final ensemble downloads all models from S3
- Kaggle dataset created with all models for submission

#### 5-7 Day Timeline
1. **Day 1-2**: Setup and start Kaggle training
2. **Day 3-4**: Parallel training on all platforms
3. **Day 5-6**: Ensemble training on Lambda Labs
4. **Day 7**: Export and Kaggle submission

---

### Kaggle Environment
This project is optimized for the Kaggle environment.

#### Model & Cost Allocation
```python
# Models: SpecProj-UNet + Heat-Kernel
# Training Time: ~10-14 hours total
# Cost: $0
```

#### Quick Start: Full Pipeline
1. Create a new notebook in the [Waveform Inversion competition](https://www.kaggle.com/competitions/waveform-inversion)
2. **Important**: Add the required datasets to your notebook first:
   - Click on the 'Data' tab
   - Click 'Add Data'
   - Search for and add:
     1. 'Waveform Inversion' competition dataset
     2. 'openfwi-preprocessed-72x72' dataset (contains preprocessed data and pretrained models)
3. **Clone the repository**:
```python
!git clone -b dev https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI
```
4. **Install dependencies using Mamba**:
Kaggle comes with Mamba pre-installed. We use the `environment.yml` file for a fast and reliable setup.
```bash
# Create and activate the environment
mamba env create -f environment.yml
source activate YaleGWI
```
5. Setting up AWS Credentials in Kaggle
   1. Go to your Kaggle account settings
   2. Click on "Add-ons" tab
   3. Click on "Secrets" in the left sidebar
   4. Add the following secrets:
      - `aws_access_key_id`: Your AWS access key ID
      - `aws_secret_access_key`: Your AWS secret access key
      - `aws_region`: Your AWS region (optional, defaults to 'us-east-1')
      - `aws_s3_bucket`: Your S3 bucket name for storing preprocessed data

6. **Run preprocessing with S3 offloading**:
```python
!pip install zarr
import zarr
from src.core.preprocess import main as preprocess_main
preprocess_main()  # Will use S3 bucket from Kaggle secrets
```
#### Verify Preprocessed Data from S3
After preprocessing, the script uploads the data to S3. You can verify the data integrity on S3 directly using our verification script. This requires your AWS credentials to be set as Kaggle secrets (see below).

```bash
# Ensure you have set your AWS credentials as Kaggle secrets
# Then, run the verification script:
python -m tests.test_verify_s3_preprocess
```
7. **Start training**:
```python
# Train SpecProj-UNet
!python kaggle_training.py --model specproj_unet --epochs 30 --keep-alive

# Train Heat-Kernel Model  
!python kaggle_training.py --model heat_kernel --epochs 30 --keep-alive
```

#### Notebook Organization
The project uses a notebook update script (`update_kaggle_notebook.py`) that automatically organizes code into cells in the correct order:
1. Imports and setup
2. Preprocessing
3. Model registry and checkpoint management
4. Data loading and geometric features
5. Training configuration
6. Training loop
7. Inference and submission

---

### Google Colab Environment

> **💡 Pro Tip**: Phase 1 is now COMPLETE! All components are working and tested. Ready for Phase 2 & 3 training.

#### Quick Start: Smart Preprocessing Workflow

##### Option A: Quick Setup (Recommended for Repeated Runs)
```python
# One-command setup that skips preprocessing if data already exists
!git clone -b dev https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI

# Quick setup - skips preprocessing if data exists locally or in Google Drive
from src.utils.colab_setup import quick_colab_setup
results = quick_colab_setup(
    use_s3=True,           # Use S3 for data operations
    mount_drive=True,      # Mount Google Drive for persistent storage
    run_tests=True,        # Run validation tests
    force_reprocess=False  # Skip preprocessing if data exists
)

# Force reprocessing after config changes
results = quick_colab_setup(use_s3=True, force_reprocess=True)

# Check data status manually
from src.utils.colab_setup import check_preprocessed_data_exists
status = check_preprocessed_data_exists('/content/YaleGWI/preprocessed', save_to_drive=True, use_s3=True)
```

##### Option B: Full Automated Setup (First Time)
```python
# Complete setup with preprocessing
from src.utils.colab_setup import complete_colab_setup
results = complete_colab_setup(
    data_path='/content/YaleGWI/train_samples',
    use_s3=True,           # Use S3 for data operations
    mount_drive=True,      # Mount Google Drive for persistent storage
    download_dataset=False, # Set to True if you need to download the dataset
    setup_aws=True,        # Load AWS credentials from Colab secrets
    run_tests=True,        # Run validation tests
    force_reprocess=False  # Skip preprocessing if data exists
)
```

**Smart Preprocessing Features:**
- ✅ **Automatic Detection**: Checks local, Google Drive, and S3 for existing data
- ✅ **Intelligent Skip**: Skips preprocessing if data already exists
- ✅ **Efficient Copy**: Copies from Google Drive to local if needed
- ✅ **Force Option**: Override skip behavior when needed
- ✅ **Data Validation**: Verifies data quality before skipping

**Phase 1 Status: ✅ COMPLETE**
- ✅ All Phase 1 tests passing
- ✅ Preprocessing working with reduced aliasing (5-7% vs 10-13%)
- ✅ All components functional and tested
- ✅ Ready for Phase 2 & 3 training

**Before running this, make sure to set up your AWS credentials in Colab secrets**:
1. Go to the left sidebar in Colab
2. Click on the "Secrets" icon (🔑)
3. Add these secrets:
   - `aws_access_key_id`: Your AWS access key ID
   - `aws_secret_access_key`: Your AWS secret access key
   - `aws_region`: Your AWS region (e.g., us-east-1)
   - `aws_s3_bucket`: Your S3 bucket name

#### Verify Preprocessed Data (Optional)
After preprocessing, you can verify the data structure:
```python
# Run verification test
!python src/utils/colab_test_setup.py

# Or check manually
from pathlib import Path
import zarr

# Check GPU datasets
gpu0_path = Path('/content/YaleGWI/preprocessed/gpu0/seismic.zarr')
gpu1_path = Path('/content/YaleGWI/preprocessed/gpu1/seismic.zarr')

if gpu0_path.exists() and gpu1_path.exists():
    data0 = zarr.open(str(gpu0_path))
    data1 = zarr.open(str(gpu1_path))
    print(f"✅ GPU0: {data0.shape} samples")
    print(f"✅ GPU1: {data1.shape} samples")
else:
    print("❌ GPU datasets not found - preprocessing may have failed")
```

#### Performance Optimization for Colab
1. **Use TPU (if available)**:
   ```python
   # Check if TPU is available
   import os
   if 'COLAB_TPU_ADDR' in os.environ:
       print("TPU available - consider using TPU for faster training")
   ```
2. **Optimize data loading**:
   ```python
   # Use memory mapping for large datasets
   CFG.use_mmap = True
   
   # Reduce number of workers for Colab
   CFG.num_workers = 2
   ```
3. **Use mixed precision**:
   ```python
   # Enable automatic mixed precision
   CFG.use_amp = True
   CFG.dtype = "float16"
   ```
4. **Smart preprocessing** (new):
   ```python
   # Use smart preprocessing to avoid unnecessary reprocessing
   from src.utils.colab_setup import quick_colab_setup
   results = quick_colab_setup(use_s3=True, force_reprocess=False)
   ```

#### Troubleshooting Common Colab Issues
##### Memory Issues
```python
# If you encounter memory issues:
import gc
import torch

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Clear Python memory
gc.collect()

# Reduce batch size
CFG.batch = 8  # or even smaller
```
##### Runtime Disconnection
```python
# To prevent runtime disconnection during long preprocessing:
import time

def keep_alive():
    """Keep the runtime alive during long operations."""
    while True:
        time.sleep(60)
        print("Still running...")

# Run this in a separate cell during preprocessing
```

---

### Lambda Labs Environment

This section guides you through setting up and running your training on a Lambda Labs cloud instance.

#### Model & Cost Allocation
```bash
# Models: EGNN + SE(3)-Transformer + Ensemble
# Training Time: ~2-3 days total
# Cost: ~$200
```

#### 1. SSH Key Setup
Lambda Labs requires a modern SSH key type. The `ssh-rsa` standard is often deprecated. We will use `ed25519`.

**A. Generate the Key**
Run the following command on your local machine. Use a strong passphrase when prompted.
```bash
ssh-keygen -t ed25519 -f ~/.ssh/lambda_labs_ed25519 -C "your_email@example.com"
```

**B. Add Public Key to Lambda Labs**
Copy the entire contents of your **public** key and add it to your Lambda Labs account.
```bash
cat ~/.ssh/lambda_labs_ed25519.pub
```
1.  Go to the [SSH Keys page](https://cloud.lambdalabs.com/ssh-keys) on your Lambda Labs dashboard.
2.  Click "Add SSH Key".
3.  Paste the copied public key into the text field.
4.  Give it a memorable name (e.g., "MacBook Main").

#### 2. Launching an Instance
**A. Choose an Instance Type**
For the intensive models (EGNN, SE(3)-Transformer, Ensemble), a powerful GPU is recommended.
- **Best Value**: **RTX 6000 Ada** ($0.69/hr) offers 48GB of VRAM, which is excellent for large models.
- **Budget Option**: **A10** ($0.59/hr) has 24GB of VRAM, which may require smaller batch sizes.

**B. Launch**
1.  Go to the [Instances page](https://cloud.lambdalabs.com/instances) and click "Launch Instance".
2.  Select your desired GPU type and region.
3.  Ensure your new `ed25519` SSH key is selected.
4.  For filesystem, select **"Don't attach a filesystem"** for initial testing and ephemeral jobs. For persistent storage across sessions, you can create and attach a filesystem.
5.  Click "Launch".

#### 3. Connect and Run Setup
Once the instance is running, copy its IP address.

**A. SSH into the Instance**
The default username is `ubuntu`. Use the `ed25519` key you created.
```bash
ssh -i ~/.ssh/lambda_labs_ed25519 ubuntu@<YOUR_INSTANCE_IP>
```
The first time you connect, you will be asked to verify the host's authenticity. Type `yes`. Then, enter your SSH key's passphrase.

**B. Run the Setup Script**
Once you are logged in, clone the repository and run the setup script. This will automatically install Mamba and set up the environment using `environment.yml`.
```bash
# 1. Clone the repository
git clone -b dev https://github.com/uncertainlyprincipaled/YaleGWI.git

# 2. Navigate into the directory
cd YaleGWI

# 3. Make the setup script executable and run it
chmod +x scripts/setup_lambda.sh && ./scripts/setup_lambda.sh
```
The script will prompt you to configure AWS credentials if you wish to connect to S3.

#### 4. Verify Preprocessed Data from S3
After setting up the instance, you can verify your preprocessed data on S3 using the dedicated test script.
```bash
# 1. Ensure AWS credentials are set in your environment
export AWS_ACCESS_KEY_ID="YOUR_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET"
export AWS_REGION="us-east-1"

# 2. Run the verification script
python -m tests.test_verify_s3_preprocess
```

#### 5. Start Training
You can now run your training jobs. The `train_lambda.sh` script created by the setup will automatically activate the correct Mamba environment.
```bash
# Train individual models
./train_lambda.sh --model egnn --epochs 30
# ... existing code ...
4. **Set Up Python Environment with Mamba**:
   ```bash
   # Mamba is recommended for fast installation
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
   bash Mambaforge-$(uname)-$(uname -m).sh -b -p "${HOME}/mambaforge"
   source "${HOME}/mambaforge/bin/activate"
   mamba init
   
   # After restarting shell, create the environment
   mamba env create -f environment.yml
   conda activate YaleGWI
   ```

5. **Configure AWS Credentials**
   ```bash
   # ... existing code ...
   ```

---

### AWS EC2 Environment
#### Quick Start: Full Pipeline
1. **Preprocessing**
   ```python
   from src.core.preprocess import load_data
   load_data('/mnt/waveform-inversion/train_samples', '/mnt/output/preprocessed', use_s3=True)
   ```
2. **Training**
   ```python
   from src.core.train_ec2 import main as train_ec2_main
   train_ec2_main()
   ```
3. **Inference**
   ```python
   from src.core.model import get_model
   import torch
   model = get_model()
   model.load_state_dict(torch.load('outputs/best.pth'))
   predictions = model(input_data)
   ```

#### Detailed Setup: Multi-GPU Training on AWS EC2 (g4dn.12xlarge)
1. **Launch a New EC2 Instance**
   - Use a GPU-enabled instance: **g4dn.12xlarge** (4 × NVIDIA T4, 16GB each)
   - Use Ubuntu 22.04 or 20.04 AMI
   - Attach or create a 200GB+ EBS volume for data

2. **SSH into the Instance**
   ```bash
   ssh -i .env/aws/<your-key>.pem ubuntu@<instance-ip>
   ```

3. **Clone the Repository**
   ```bash
   git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
   cd YaleGWI
   ```

4. **Set Up Python Environment**
   ```bash
   sudo apt-get update && sudo apt-get install -y python3-venv
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Configure AWS Credentials**
   ```bash
   source .env/aws/credentials
   ```

6. **Download/Sync Data from S3**
   ```bash
   sudo mkdir -p /mnt/waveform-inversion
   sudo mkdir -p /mnt/output
   sudo chown $USER:$USER /mnt/waveform-inversion
   sudo chown $USER:$USER /mnt/output
   python -m src.core.setup aws
   ```

7. **Multi-GPU Training (Distributed Data Parallel)**
   - The codebase now supports multi-GPU training using PyTorch DDP.
   - Use the provided `train_ec2.py` script for EC2 multi-GPU training:
   ```bash
   export GWI_ENV=aws
   python -m torch.distributed.launch --nproc_per_node=4 src/core/train_ec2.py --epochs 120 --batch 4 --amp --experiment-tag "run-$(date +%F)"
   ```
   - This will launch 4 processes (one per GPU) and train in parallel.

---

### AWS SageMaker Environment

#### Quick Start: Full Pipeline
1. **Preprocessing**
   ```python
   from src.core.preprocess import load_data
   load_data('/home/sagemaker-user/YaleGWI/train_samples', '/home/sagemaker-user/YaleGWI/preprocessed', use_s3=True)
   ```
2. **Training**
   ```python
   from src.core.train import train
   train(fp16=True)
   ```
3. **Inference**
   ```python
   from src.core.model import get_model
   import torch
   model = get_model()
   model.load_state_dict(torch.load('outputs/best.pth'))
   predictions = model(input_data)
   ```

#### Detailed Setup
1. **Create SageMaker Notebook Instance**:
   - Go to AWS SageMaker Console
   - Create new notebook instance
   - Choose `ml.c5.2xlarge` (CPU-only instance)
   - Set volume size to 100GB
   - Create new IAM role with S3 access
   - Launch instance

2. **Clone Repository**:
   ```bash
   # Open terminal in JupyterLab and run:
   git clone -b dev https://github.com/uncertainlyprincipaled/YaleGWI.git
   cd YaleGWI
   ```

3. **Import All Core Libraries in Your Notebook**:
   ```python
   import os
   import sys
   import json
   import boto3
   from botocore.exceptions import ClientError

   # 1. Set environment variable for config
   os.environ['GWI_ENV'] = 'sagemaker'

   # 2. Set up project path
   os.chdir('/home/sagemaker-user/YaleGWI')  # Adjust as needed
   sys.path.append(os.path.abspath('./src'))

   # 2. Retrieve secret from AWS Secrets Manager
   def get_secret():
       secret_name = "sagemaker-access"  # Use your actual secret name
       region_name = "us-east-1"              # Use your region

       session = boto3.session.Session()
       client = session.client(
           service_name='secretsmanager',
           region_name=region_name
       )

       try:
           get_secret_value_response = client.get_secret_value(
               SecretId=secret_name
           )
       except ClientError as e:
           raise e

       return json.loads(get_secret_value_response['SecretString'])

   secret = get_secret()

   # 3. Set environment variables for DataManager and boto3
   os.environ['AWS_ACCESS_KEY_ID'] = secret['aws_access_key_id']
   os.environ['AWS_SECRET_ACCESS_KEY'] = secret['aws_secret_access_key']
   os.environ['AWS_REGION'] = secret['region_name']
   os.environ['AWS_S3_BUCKET'] = secret['s3_bucket']

   # 4. Import all core libraries
   from src.core.imports import *
   # Now you have np, torch, CFG, DataManager, etc. available
   ```

---

## 5. Additional Documentation

### Usage Examples
- For large datasets, use the provided preprocessing script to create GPU-specific datasets
- The preprocessing includes Nyquist validation and geometric feature extraction
- Processed data is saved in zarr format for efficient loading
- Use the geometric-aware data loader for family-specific training

### Model Tracking
> **Note:** MLflow tracking and logging is abstracted away into a utility class. In most workflows (including EC2 and cloud), users do not need to write MLflow code directly. All model versioning, metric logging, and artifact management is handled automatically by the pipeline.
- Install MLflow: `pip install mlflow`
- Set up tracking server (optional)
- Configure experiment tracking

### AWS Management Tools

#### AWS Service Quota Checking
Before running resource-intensive operations, check your AWS service quotas:
```bash
# Make script executable (first time only)
chmod +x scripts/check_aws_quotas.sh

# Check quotas for default region (us-east-1)
./scripts/check_aws_quotas.sh
```
**Important quotas to monitor**:
- EC2 instance limits (especially G and P instances for GPU training)
- EBS storage limits
- S3 bucket and object limits
- SageMaker training job limits

#### EBS Volume Cleanup
To manage costs and clean up unused EBS volumes:
```bash
# Make script executable (first time only)
chmod +x scripts/cleanup_ebs_volumes.sh

# Dry run: see what would be deleted
./scripts/cleanup_ebs_volumes.sh -a

# Delete all unattached volumes (with confirmation)
./scripts/cleanup_ebs_volumes.sh -a -x
```
**Safety Features**:
- **Dry-run mode by default** - shows what would be deleted without actually deleting
- **Confirmation prompts** - asks for confirmation before deletion
- **Safety checks** - won't delete volumes that are attached, have snapshots, or are in use.
- **Cost estimation** - shows potential cost savings
- **Detailed reporting** - provides comprehensive results