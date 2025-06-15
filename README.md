# SpecProj-UNet for Seismic Waveform Inversion

## 1. Overview of Functionality and Structure

A deep learning solution for seismic waveform inversion using a HGNet/ConvNeXt backbone with:
- EMA (Exponential Moving Average) for stable training
- AMP (Automatic Mixed Precision) for faster training
- Distributed Data Parallel (DDP) for multi-GPU training
- Test Time Augmentation (TTA) for improved inference
- Geometric-aware preprocessing and model management
- Family-specific data loading with geometric features
- Cross-validation with geometric metrics

### Data IO Policy (IMPORTANT)
**All data loading, streaming, and batching must go through `DataManager` in `src/core/data_manager.py`.**
- Do NOT load data directly in any other file.
- This ensures memory efficiency and prevents RAM spikes in Kaggle.
- All source files must import and use DataManager for any data access.

### Project Structure
```
src/
├── core/
│   ├── preprocess.py      # Geometric-aware preprocessing
│   ├── registry.py        # Model registry with geometric metadata
│   ├── checkpoint.py      # Checkpoint management
│   ├── geometric_loader.py # Family-specific data loading
│   ├── geometric_cv.py    # Cross-validation framework
│   ├── data_manager.py    # Data IO management
│   ├── model.py          # Model architecture
│   └── config.py         # Configuration
├── utils/
│   └── update_kaggle_notebook.py  # Notebook update script
└── requirements.txt      # Project dependencies
```

### Features
- **Physics-Guided Architecture**: Uses spectral projectors to inject wave-equation priors
- **Mixed Precision Training**: Supports both fp16 and bfloat16 for efficient training
- **Memory Optimization**: Implements efficient attention and memory management
- **Family-Aware Training**: Stratified sampling by geological families
- **Robust Training**: Includes gradient clipping, early stopping, and learning rate scheduling
- **Comprehensive Checkpointing**: Saves full training state for easy resumption
- **Geometric-Aware Processing**: Nyquist validation, geometric feature extraction
- **Model Registry**: Version control with geometric metadata
- **Cross-Validation**: Family-based stratification with geometric metrics

### Usage
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

## 2. Environments Availability

This project supports the following environments:
- Kaggle
- Google Colab
- AWS EC2
- AWS SageMaker
- Local development

---

## 3. Kaggle Instructions

### Setting up AWS Credentials in Kaggle
1. Go to your Kaggle account settings
2. Click on "Add-ons" tab
3. Click on "Secrets" in the left sidebar
4. Add the following secrets:
   - `aws_access_key_id`: Your AWS access key ID
   - `aws_secret_access_key`: Your AWS secret access key
   - `aws_region`: Your AWS region (optional, defaults to 'us-east-1')
   - `aws_s3_bucket`: Your S3 bucket name for storing preprocessed data

### Quick Start
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
from src.core.preprocess import main as preprocess_main
preprocess_main()  # Will use S3 bucket from Kaggle secrets
```
6. Start training:
```python
from src.core.train import train
train(fp16=True)
```

### Notebook Organization
The project uses a notebook update script (`update_kaggle_notebook.py`) that automatically organizes code into cells in the correct order:
1. Imports and setup
2. Preprocessing
3. Model registry and checkpoint management
4. Data loading and geometric features
5. Training configuration
6. Training loop
7. Inference and submission

---

## 4. Colab Instructions

### Quick Start
1. Clone the repository:
```python
!git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI
```
2. Install dependencies:
```python
!pip install -r requirements.txt
```
3. Run preprocessing with S3 offloading:
```python
# In Colab, you can run preprocessing directly
from src.core.preprocess import main as preprocess_main
preprocess_main()  # Will use S3 bucket from environment variables or config

# Or specify the bucket explicitly
preprocess_main(s3_bucket='your-bucket-name')
```

### Setting up AWS Credentials in Colab
1. Set up environment variables in Colab:
```python
import os
os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'
os.environ['AWS_REGION'] = 'your_region'  # optional
os.environ['AWS_S3_BUCKET'] = 'your_bucket_name'
```

2. Or create a credentials file:
```python
import json
import os

# Create .env/aws directory
os.makedirs('.env/aws', exist_ok=True)

# Create credentials file
credentials = {
    "aws_access_key_id": "YOUR_ACCESS_KEY_ID",
    "aws_secret_access_key": "YOUR_SECRET_ACCESS_KEY",
    "region_name": "us-east-1",
    "s3_bucket": "YOUR_BUCKET_NAME"
}

with open('.env/aws/credentials.json', 'w') as f:
    json.dump(credentials, f)
```

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

## 5. AWS EC2 Multi-GPU (g4dn.12xlarge) Instructions

### Quick Start: Multi-GPU Training on AWS EC2 (g4dn.12xlarge)

#### 1. Launch a New EC2 Instance
- Use a GPU-enabled instance: **g4dn.12xlarge** (4 × NVIDIA T4, 16GB each)
- Use Ubuntu 22.04 or 20.04 AMI
- Attach or create a 200GB+ EBS volume for data

#### 2. SSH into the Instance
```bash
ssh -i .env/aws/<your-key>.pem ubuntu@<instance-ip>
```

#### 3. Clone the Repository
```bash
git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
cd YaleGWI
```

#### 4. Set Up Python Environment
```bash
sudo apt-get update && sudo apt-get install -y python3-venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Configure AWS Credentials
```bash
source .env/aws/credentials
```

#### 6. Download/Sync Data from S3
```bash
sudo mkdir -p /mnt/waveform-inversion
sudo mkdir -p /mnt/output
sudo chown $USER:$USER /mnt/waveform-inversion
sudo chown $USER:$USER /mnt/output
python -m src.core.setup aws
```

#### 7. Multi-GPU Training (Distributed Data Parallel)
- The codebase now supports multi-GPU training using PyTorch DDP.
- Use the provided `train_ec2.py` script for EC2 multi-GPU training:

```bash
export GWI_ENV=aws
python -m torch.distributed.launch --nproc_per_node=4 src/core/train_ec2.py --epochs 120 --batch 4 --amp --experiment-tag "run-$(date +%F)"
```
- This will launch 4 processes (one per GPU) and train in parallel.
- Make sure your data is preprocessed for efficient distributed loading (see `preprocess.py`).

#### 8. Environment-Specific Scripts
- Use `train_ec2.py` for EC2 multi-GPU training.
- Use `train_kaggle.py` for Kaggle single/multi-GPU training.
- Each script is tailored for its environment for maximum robustness and portability.

#### 9. Notes
- For very large datasets, use the provided `preprocess.py` to split large .npy files into smaller per-sample files for efficient parallel loading.
- See `src/core/config.py` for environment and DDP configuration options.
- For troubleshooting, see the logs in `outputs/` and the EC2 console.

---

## 6. SageMaker Instructions

1. Launch a SageMaker instance with appropriate permissions and storage.
2. Clone the repository and install dependencies as above.
3. Use the provided setup scripts and ensure S3 access is configured.
4. Follow the same data and training steps as for EC2, but use SageMaker-specific paths if needed.

---

# (All other original content, such as features, development workflow, contributing, license, etc., remains unchanged and follows these sections.)



