# SpecProj-UNet for Seismic Waveform Inversion

## Table of Contents
1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Environment Setup](#3-environment-setup)
   - [Kaggle](#kaggle)
   - [Google Colab](#google-colab)
     - [Smart Preprocessing Workflow](#smart-preprocessing-workflow)
     - [Performance Optimization](#performance-optimization-for-colab)
   - [AWS EC2](#aws-ec2)
   - [AWS SageMaker](#aws-sagemaker)
4. [Usage Examples](#4-usage-examples)
5. [Model Tracking](#5-model-tracking)
6. [AWS Management Tools](#6-aws-management-tools)

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

## 2. Quick Start

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

## 🚀 **Multi-Platform Training Strategy**

### **💰 Cost-Optimized Approach**
Our training strategy leverages both free and paid resources for maximum efficiency:

- **Kaggle (FREE)**: Less computationally intensive models
- **Lambda Labs (~$200)**: More intensive models + ensemble training
- **Total Cost**: ~$200 instead of ~$400-1000

### **📊 Model Distribution**

#### **Kaggle Environment (Free GPU)**
```python
# Models: SpecProj-UNet + Heat-Kernel
# Training Time: ~10-14 hours total
# Cost: $0

# Quick setup
!git clone -b dev https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI

# Train SpecProj-UNet
!python kaggle_training.py --model specproj_unet --epochs 30 --keep-alive

# Train Heat-Kernel Model  
!python kaggle_training.py --model heat_kernel --epochs 30 --keep-alive
```

#### **Lambda Labs Environment (Paid GPU)**
```bash
# Models: EGNN + SE(3)-Transformer + Ensemble
# Training Time: ~2-3 days total
# Cost: ~$200

# Launch 2x RTX 4090 cluster
ssh root@<instance-ip>

# Setup and train
git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
cd YaleGWI
chmod +x scripts/setup_lambda.sh
./scripts/setup_lambda.sh

# Train individual models
python src/core/train_lambda.py --model egnn --epochs 30 --gpu-id 0
python src/core/train_lambda.py --model se3_transformer --epochs 30 --gpu-id 1

# Train ensemble
python src/core/train_lambda.py --model ensemble --epochs 50 --ensemble-members 4 --all-gpus
```

### **🔄 S3 Sync Strategy**
- All models automatically export to S3
- Final ensemble downloads all models from S3
- Kaggle dataset created with all models for submission

### **📝 5-7 Day Timeline**
1. **Day 1-2**: Setup and start Kaggle training
2. **Day 3-4**: Parallel training on both platforms
3. **Day 5-6**: Ensemble training on Lambda Labs
4. **Day 7**: Export and Kaggle submission

---

## 📊 **Current Project Status**

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

## 3. Environment Setup

This project supports the following environments:
- Kaggle
- Google Colab
- AWS EC2
- AWS SageMaker
- Local development

### Kaggle {#kaggle}

#### Quick Start: Full Pipeline
1. **Preprocessing**
   ```python
   from src.core.preprocess import load_data
   load_data('/kaggle/input/waveform-inversion/train_samples', '/kaggle/working/preprocessed', use_s3=False)
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
4. **Model Tracking**
   - MLflow tracking is handled automatically via the pipeline utility class. No manual MLflow code is needed in most workflows.

#### Detailed Setup
1. Create a new notebook in the [Waveform Inversion competition](https://www.kaggle.com/competitions/waveform-inversion)
2. **Important**: Add the required datasets to your notebook first:
   - Click on the 'Data' tab
   - Click 'Add Data'
   - Search for and add:
     1. 'Waveform Inversion' competition dataset
     2. 'openfwi-preprocessed-72x72' dataset (contains preprocessed data and pretrained models)
3. Clone the repository:
```python
!git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI
```
4. Install dependencies:
```python
!pip install -r requirements.txt
```
5. Run preprocessing with S3 offloading:
```python
!pip install zarr
import zarr
from src.core.preprocess import main as preprocess_main
preprocess_main()  # Will use S3 bucket from Kaggle secrets
```
6. Start training:
```python
from src.core.train import train
train(fp16=True)
```

#### Setting up AWS Credentials in Kaggle
1. Go to your Kaggle account settings
2. Click on "Add-ons" tab
3. Click on "Secrets" in the left sidebar
4. Add the following secrets:
   - `aws_access_key_id`: Your AWS access key ID
   - `aws_secret_access_key`: Your AWS secret access key
   - `aws_region`: Your AWS region (optional, defaults to 'us-east-1')
   - `aws_s3_bucket`: Your S3 bucket name for storing preprocessed data

#### Notebook Organization
The project uses a notebook update script (`update_kaggle_notebook.py`) that automatically organizes code into cells in the correct order:
1. Imports and setup
2. Preprocessing
3. Model registry and checkpoint management
4. Data loading and geometric features
5. Training configuration
6. Training loop
7. Inference and submission

### Google Colab {#google-colab}

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

##### Option C: Manual Setup
1. **Environment Setup**
   ```python
   # Clone repository and setup environment
   !git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
   %cd YaleGWI
   !pip install -r requirements.txt
   ```
2. **Data Loading and Verification**
   ```python
   # Mount Google Drive (optional, for persistent storage)
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Verify data structure
   from src.core.preprocess import verify_data_structure
   verify_data_structure('/content/YaleGWI/train_samples')
   ```
3. **Smart Preprocessing**
   ```python
   # Check if preprocessed data already exists
   from src.utils.colab_setup import check_preprocessed_data_exists
   
   data_status = check_preprocessed_data_exists(
       output_root='/content/YaleGWI/preprocessed',
       save_to_drive=True,
       use_s3=True
   )
   
   if data_status['exists_locally']:
       print("✅ Preprocessed data found locally - skipping preprocessing")
   elif data_status['exists_in_drive']:
       print("📋 Copying preprocessed data from Google Drive...")
       from src.utils.colab_setup import copy_preprocessed_data_from_drive
       copy_preprocessed_data_from_drive(
           data_status['drive_path'], 
           '/content/YaleGWI/preprocessed'
       )
   else:
       print("🔄 Running preprocessing...")
       from src.core.preprocess import load_data
       load_data('/content/YaleGWI/train_samples', '/content/YaleGWI/preprocessed', use_s3=False)
   ```
4. **Training**
   ```python
   from src.core.train import train
   train(fp16=True)
   ```
5. **Inference**
   ```python
   from src.core.model import get_model
   import torch
   model = get_model()
   model.load_state_dict(torch.load('outputs/best.pth'))
   predictions = model(input_data)
   ```
6. **Model Tracking**
   - MLflow tracking is handled automatically via the pipeline utility class. No manual MLflow code is needed in most workflows.

#### Detailed Colab Setup Guide

##### Step 1: Environment Setup

1. **Create a new Colab notebook** and ensure you have a GPU runtime:
   - Go to Runtime → Change runtime type
   - Set Hardware accelerator to "GPU"
   - Set Runtime type to "Python 3"

2. **Clone the repository and install dependencies**:
   ```python
   # Clone the repository
   !git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
   %cd YaleGWI
   
   # Install required packages
   !pip install -r requirements.txt
   
   # Install additional Colab-specific packages
   !pip install zarr dask scipy
   
   # Verify installation
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   if torch.cuda.is_available():
       print(f"GPU name: {torch.cuda.get_device_name(0)}")
   ```

3. **Set up environment variables**:
   ```python
   import os
   import sys
   
   # Set environment for Colab
   os.environ['GWI_ENV'] = 'colab'
   os.environ['DEBUG_MODE'] = '0'  # Set to '1' for debug mode
   
   # Add src to Python path
   sys.path.append('/content/YaleGWI/src')
   
   # Verify environment setup
   from src.core.config import CFG
   print(f"Environment: {CFG.env.kind}")
   print(f"Device: {CFG.env.device}")
   ```

##### Step 2: Data Loading and Verification

1. **Mount Google Drive (optional, for persistent storage)**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Create persistent directories in Drive
   !mkdir -p /content/drive/MyDrive/YaleGWI/data
   !mkdir -p /content/drive/MyDrive/YaleGWI/outputs
   !mkdir -p /content/drive/MyDrive/YaleGWI/preprocessed
   ```

2. **Download and verify the dataset**:
   ```python
   # Download the dataset (if not already available)
   !wget -O /content/YaleGWI/train_samples.zip "YOUR_DATASET_URL"
   !unzip -q /content/YaleGWI/train_samples.zip -d /content/YaleGWI/
   
   # Or if you have the data in Drive
   !cp -r /content/drive/MyDrive/YaleGWI/data/train_samples /content/YaleGWI/
   ```

3. **Verify data structure**:
   ```python
   from src.core.preprocess import verify_data_structure
   from pathlib import Path
   
   # Verify the data structure
   train_path = Path('/content/YaleGWI/train_samples')
   if train_path.exists():
       print("✓ Training data found")
       verify_data_structure(train_path)
   else:
       print("✗ Training data not found. Please download the dataset first.")
   ```

4. **Check available families**:
   ```python
   from src.core.config import CFG
   
   print("Available geological families:")
   for family, path in CFG.paths.families.items():
       if path.exists():
           print(f"  ✓ {family}: {path}")
       else:
           print(f"  ✗ {family}: {path} (missing)")
   ```

##### Step 3: Preprocessing Pipeline

1. **Run preprocessing with memory monitoring**:
   ```python
   import psutil
   import gc
   
   # Monitor memory before preprocessing
   def print_memory_usage():
       process = psutil.Process()
       memory_info = process.memory_info()
       print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
   
   print_memory_usage()
   
   # Run preprocessing
   from src.core.preprocess import load_data
   
   try:
       processed_paths = load_data(
           input_root='/content/YaleGWI/train_samples',
           output_root='/content/YaleGWI/preprocessed',
           use_s3=False  # Set to True if you have S3 configured
       )
       print(f"✓ Preprocessing completed. {len(processed_paths)} samples processed.")
   except Exception as e:
       print(f"✗ Preprocessing failed: {e}")
       raise
   
   # Clean up memory
   gc.collect()
   print_memory_usage()
   ```

2. **Verify preprocessing results**:
   ```python
   from pathlib import Path
   import numpy as np
   
   # Check preprocessed data
   preprocessed_dir = Path('/content/YaleGWI/preprocessed')
   
   if preprocessed_dir.exists():
       print("✓ Preprocessed data directory created")
       
       # Check GPU-specific datasets
       gpu0_dir = preprocessed_dir / 'gpu0'
       gpu1_dir = preprocessed_dir / 'gpu1'
       
       if gpu0_dir.exists() and gpu1_dir.exists():
           print("✓ GPU-specific datasets created")
           
           # Load a sample to verify format
           try:
               import zarr
               sample_data = zarr.open(str(gpu0_dir / 'seismic.zarr'))
               print(f"✓ Sample data shape: {sample_data.shape}")
               print(f"✓ Sample data dtype: {sample_data.dtype}")
           except Exception as e:
               print(f"✗ Error loading sample data: {e}")
       else:
           print("✗ GPU-specific datasets not found")
   else:
       print("✗ Preprocessed data directory not found")
   ```

3. **Save to Google Drive (optional)**:
   ```python
   # Copy preprocessed data to Drive for persistence
   !cp -r /content/YaleGWI/preprocessed /content/drive/MyDrive/YaleGWI/
   print("✓ Preprocessed data saved to Google Drive")
   ```

##### Step 4: Training Configuration

1. **Configure training parameters**:
   ```python
   from src.core.config import CFG
   
   # Set training parameters for Colab
   CFG.batch = 16  # Adjust based on your GPU memory
   CFG.epochs = 10  # Start with fewer epochs for testing
   CFG.use_amp = True  # Enable mixed precision
   CFG.debug_mode = False  # Set to True for debug mode
   
   print(f"Training configuration:")
   print(f"  Batch size: {CFG.batch}")
   print(f"  Epochs: {CFG.epochs}")
   print(f"  Mixed precision: {CFG.use_amp}")
   print(f"  Debug mode: {CFG.debug_mode}")
   ```

2. **Set up AWS credentials (if using S3)**:
   ```python
   import os
   
   # Set AWS credentials (if using S3)
   os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
   os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'
   os.environ['AWS_REGION'] = 'your_region'
   os.environ['AWS_S3_BUCKET'] = 'your_bucket_name'
   ```

##### Step 5: Training and Monitoring

1. **Start training**:
   ```python
   from src.core.train import train
   
   try:
       # Start training
       train(fp16=True)
       print("✓ Training completed successfully")
   except Exception as e:
       print(f"✗ Training failed: {e}")
       raise
   ```

2. **Monitor training progress**:
   ```python
   # Check training outputs
   from pathlib import Path
   
   outputs_dir = Path('/content/YaleGWI/outputs')
   if outputs_dir.exists():
       print("Training outputs:")
       for file in outputs_dir.glob('*'):
           print(f"  {file.name}")
   ```

3. **Save results to Drive**:
   ```python
   # Copy results to Drive
   !cp -r /content/YaleGWI/outputs /content/drive/MyDrive/YaleGWI/
   print("✓ Training results saved to Google Drive")
   ```

#### Setting up AWS Credentials in Colab

##### Using Google Colab Secrets 

1. **Set up secrets in Colab**:
   - Go to the left sidebar in Colab
   - Click on the "Secrets" icon (🔑)
   - Click "Add new secret"
   - Add the following secrets:
     - `aws_access_key_id`: Your AWS access key ID
     - `aws_secret_access_key`: Your AWS secret access key
     - `aws_region`: Your AWS region (e.g., us-east-1)
     - `aws_s3_bucket`: Your S3 bucket name

2. **Load secrets in your notebook**:
   ```python
   from google.colab import userdata
   import os
   
   # Load AWS credentials from Colab secrets
   os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('aws_access_key_id')
   os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('aws_secret_access_key')
   os.environ['AWS_REGION'] = userdata.get('aws_region')
   os.environ['AWS_S3_BUCKET'] = userdata.get('aws_s3_bucket')
   
   print("✅ AWS credentials loaded from Colab secrets")
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

##### Data Download Issues
```python
# If dataset download fails, try alternative methods:
# Method 1: Direct download
!wget --no-check-certificate -O train_samples.zip "YOUR_URL"

# Method 2: Using gdown (for Google Drive links)
!pip install gdown
!gdown "YOUR_GOOGLE_DRIVE_ID"

# Method 3: Manual upload
# Upload the dataset directly to Colab using the file browser
```

#### Smart Preprocessing Workflow

The new smart preprocessing system automatically detects existing preprocessed data and skips unnecessary reprocessing, dramatically speeding up development workflows.

##### How It Works

1. **Automatic Detection**: Checks multiple locations for existing data:
   - Local directory (`/content/YaleGWI/preprocessed`)
   - Google Drive (`/content/drive/MyDrive/YaleGWI/preprocessed`)
   - S3 bucket (if configured)

2. **Intelligent Skip Logic**:
   ```python
   # Priority order for data sources:
   # 1. Local data (fastest access)
   # 2. Google Drive data (copy to local)
   # 3. S3 data (download to local)
   # 4. Reprocess from raw data (slowest)
   ```

3. **Data Validation**: Verifies data quality before skipping:
   - Checks zarr dataset integrity
   - Validates sample counts
   - Ensures GPU-specific splits exist

##### Usage Patterns

**First Time Setup** (Full preprocessing):
```python
from src.utils.colab_setup import complete_colab_setup
results = complete_colab_setup(
    use_s3=True,
    mount_drive=True,
    run_tests=True,
    force_reprocess=False  # Will preprocess if no data exists
)
```

**Subsequent Runs** (Skip preprocessing):
```python
from src.utils.colab_setup import quick_colab_setup
results = quick_colab_setup(
    use_s3=True,
    mount_drive=True,
    run_tests=True,
    force_reprocess=False  # Will skip if data exists
)
```

**After Config Changes** (Force reprocessing):
```python
from src.utils.colab_setup import quick_colab_setup
results = quick_colab_setup(
    use_s3=True,
    mount_drive=True,
    run_tests=True,
    force_reprocess=True  # Will reprocess even if data exists
)
```

##### Manual Data Management

**Check Data Status**:
```python
from src.utils.colab_setup import check_preprocessed_data_exists

status = check_preprocessed_data_exists(
    output_root='/content/YaleGWI/preprocessed',
    save_to_drive=True,
    use_s3=True
)

print(f"Local data: {status['exists_locally']}")
print(f"Drive data: {status['exists_in_drive']}")
print(f"S3 data: {status['exists_in_s3']}")
print(f"Data quality: {status['data_quality']}")
```

**Copy from Google Drive**:
```python
from src.utils.colab_setup import copy_preprocessed_data_from_drive

success = copy_preprocessed_data_from_drive(
    drive_path='/content/drive/MyDrive/YaleGWI/preprocessed',
    local_path='/content/YaleGWI/preprocessed'
)
```

##### Expected Behavior

| Scenario | First Run | Subsequent Runs | After Config Change |
|----------|-----------|-----------------|-------------------|
| **Local Data** | 🔄 Preprocess | ⏭️ Skip | 🔄 Force Reprocess |
| **Drive Data** | 🔄 Preprocess | 📋 Copy from Drive | 🔄 Force Reprocess |
| **S3 Data** | 🔄 Preprocess | 📥 Download from S3 | 🔄 Force Reprocess |
| **No Data** | 🔄 Preprocess | 🔄 Preprocess | 🔄 Preprocess |

##### Troubleshooting Smart Preprocessing

**Data Not Found When It Should Exist**:
```python
# Check data structure manually
from pathlib import Path
import zarr

preprocessed_dir = Path('/content/YaleGWI/preprocessed')
if preprocessed_dir.exists():
    gpu0_dir = preprocessed_dir / 'gpu0'
    gpu1_dir = preprocessed_dir / 'gpu1'
    
    if gpu0_dir.exists() and gpu1_dir.exists():
        try:
            data0 = zarr.open(str(gpu0_dir / 'seismic.zarr'))
            data1 = zarr.open(str(gpu1_dir / 'seismic.zarr'))
            print(f"GPU0: {len(data0)} samples")
            print(f"GPU1: {len(data1)} samples")
        except Exception as e:
            print(f"Data corrupted: {e}")
            # Force reprocessing
            results = quick_colab_setup(force_reprocess=True)
```

**Force Reprocessing**:
```python
# If you need to reprocess for any reason:
from src.utils.colab_setup import quick_colab_setup
results = quick_colab_setup(force_reprocess=True)
```

**Clear All Data**:
```python
# Remove all preprocessed data to start fresh
import shutil
from pathlib import Path

# Remove local data
local_dir = Path('/content/YaleGWI/preprocessed')
if local_dir.exists():
    shutil.rmtree(local_dir)

# Remove Drive data (optional)
drive_dir = Path('/content/drive/MyDrive/YaleGWI/preprocessed')
if drive_dir.exists():
    shutil.rmtree(drive_dir)

print("All preprocessed data cleared")
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

### AWS EC2 {#aws-ec2}

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
4. **Model Tracking**
   - MLflow tracking is abstracted via a utility class. Users do not need to interact with MLflow directly; all logging and artifact management is handled by the pipeline.

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
   - Make sure your data is preprocessed for efficient distributed loading (see `preprocess.py`).

8. **Environment-Specific Scripts**
   - Use `train_ec2.py` for EC2 multi-GPU training.
   - Use `train_kaggle.py` for Kaggle single/multi-GPU training.
   - Each script is tailored for its environment for maximum robustness and portability.

9. **Notes**
   - For very large datasets, use the provided `preprocess.py` to split large .npy files into smaller per-sample files for efficient parallel loading.
   - See `src/core/config.py` for environment and DDP configuration options.
   - For troubleshooting, see the logs in `outputs/` and the EC2 console.

### AWS SageMaker {#aws-sagemaker}

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
4. **Model Tracking**
   - MLflow tracking is abstracted via a utility class. Users do not need to interact with MLflow directly; all logging and artifact management is handled by the pipeline.

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

#### Running Preprocessing

1. **Run Preprocessing with One Command**:
   Set your input and output paths, and whether to use S3. The pipeline will process all families and upload the processed datasets to your S3 bucket automatically.
   
   **Example:**
   ```python
   from src.core.preprocess import load_data
   
   # Set your parameters
   input_root = '/path/to/input'  # e.g., '/home/sagemaker-user/input_data'
   output_root = '/home/sagemaker-user/output_data'  # or any desired output path
   use_s3 = True  # Set to True to use S3 for IO
   
   # Run the full preprocessing pipeline
   load_data(input_root, output_root, use_s3=use_s3)
   ```
   - All processed data will be uploaded to your S3 bucket if `use_s3=True`.
   - Temporary files are cleaned up automatically.
   - The function returns a list of processed file paths for further use.

#### Cost Optimization

1. **Instance Management**:
   - Stop instance when not in use
   - Use spot instances for cost savings
   - Monitor instance metrics

2. **Storage Optimization**:
   - Clean up temporary files
   - Use S3 lifecycle policies
   - Compress data when possible

3. **Performance Tips**:
   - Use appropriate instance type
   - Monitor memory usage
   - Use chunked processing

#### Troubleshooting

1. **Common Issues**:
   - Memory errors: Reduce batch size
   - S3 timeouts: Increase timeout settings
   - Permission errors: Check IAM roles

2. **Debug Mode**:
   ```python
   # Enable debug mode
   os.environ['DEBUG_MODE'] = '1'
   ```

3. **Logging**:
   - Check CloudWatch logs
   - Use debug mode for verbose output
   - Monitor instance metrics

---

## 4. Usage Examples

### Data Management
- For large datasets, use the provided preprocessing script to create GPU-specific datasets
- The preprocessing includes Nyquist validation and geometric feature extraction
- Processed data is saved in zarr format for efficient loading
- Use the geometric-aware data loader for family-specific training

### Model Management
- Use the model registry to track different model versions
- Checkpoints include geometric metadata for reproducibility
- Cross-validation uses family-based stratification
- Monitor geometric metrics during training

---

## 5. Model Tracking

> **Note:** MLflow tracking and logging is abstracted away into a utility class. In most workflows (including EC2 and cloud), users do not need to write MLflow code directly. All model versioning, metric logging, and artifact management is handled automatically by the pipeline.

### Enabling MLflow Tracking
- Install MLflow: `pip install mlflow`
- Set up tracking server (optional)
- Configure experiment tracking

---

## 6. AWS Management Tools

### AWS Service Quota Checking

Before running resource-intensive operations, check your AWS service quotas:

```python
# Check AWS quotas locally
from src.utils.check_aws_quotas import AWSQuotaChecker

# Initialize quota checker
checker = AWSQuotaChecker(region='us-east-1')  # Change region as needed

# Run comprehensive check
results = checker.run_comprehensive_check()

# Print formatted report
from src.utils.check_aws_quotas import print_quota_report
print_quota_report(results)
```

Or run from command line:
```bash
# Check quotas for specific region
python -m src.utils.check_aws_quotas --region us-east-1

# Get JSON output
python -m src.utils.check_aws_quotas --region us-east-1 --output json
```

**Or use the convenient shell script**:
```bash
# Make script executable (first time only)
chmod +x scripts/check_aws_quotas.sh

# Check quotas for default region (us-east-1)
./scripts/check_aws_quotas.sh

# Check quotas for specific region
./scripts/check_aws_quotas.sh -r us-west-2

# Get JSON output
./scripts/check_aws_quotas.sh -o json

# Show help
./scripts/check_aws_quotas.sh -h
```

**Important quotas to monitor**:
- EC2 instance limits (especially G and P instances for GPU training)
- EBS storage limits
- S3 bucket and object limits
- SageMaker training job limits

### EBS Volume Cleanup

To manage costs and clean up unused EBS volumes:

#### Using the Shell Script (Quick and Easy)
```bash
# Make script executable (first time only)
chmod +x scripts/cleanup_ebs_volumes.sh

# List all volumes
./scripts/cleanup_ebs_volumes.sh -l

# Dry run: see what would be deleted
./scripts/cleanup_ebs_volumes.sh -a

# Delete all unattached volumes (with confirmation)
./scripts/cleanup_ebs_volumes.sh -a -x

# Delete volumes older than 7 days
./scripts/cleanup_ebs_volumes.sh -d 7 -x

# Delete specific volumes
./scripts/cleanup_ebs_volumes.sh -v vol-123,vol-456 -x

# Force delete (skip confirmation)
./scripts/cleanup_ebs_volumes.sh -a -x -f
```

#### Using the Python Script (Advanced Features)
```bash
# List volumes with cost analysis
python -m src.utils.cleanup_ebs_volumes --list-only --all-unattached

# Dry run with cost estimation
python -m src.utils.cleanup_ebs_volumes --all-unattached

# Actually delete unattached volumes
python -m src.utils.cleanup_ebs_volumes --all-unattached --execute

# Delete volumes older than 3 days
python -m src.utils.cleanup_ebs_volumes --all-unattached --older-than 3 --execute

# Get JSON output for automation
python -m src.utils.cleanup_ebs_volumes --all-unattached --output json
```

#### Safety Features
- **Dry-run mode by default** - shows what would be deleted without actually deleting
- **Confirmation prompts** - asks for confirmation before deletion
- **Safety checks** - won't delete volumes that are:
  - Attached to running instances
  - Have snapshots
  - In use or transitioning states
- **Cost estimation** - shows potential cost savings
- **Detailed reporting** - provides comprehensive results