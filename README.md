# SpecProj-UNet for Seismic Waveform Inversion

## 1. Overview of Functionality and Structure

A deep learning solution for seismic waveform inversion using a HGNet/ConvNeXt backbone with:
- EMA (Exponential Moving Average) for stable training
- AMP (Automatic Mixed Precision) for faster training
- Distributed Data Parallel (DDP) for multi-GPU training
- Test Time Augmentation (TTA) for improved inference

### Data IO Policy (IMPORTANT)
**All data loading, streaming, and batching must go through `DataManager` in `src/core/data_manager.py`.**
- Do NOT load data directly in any other file.
- This ensures memory efficiency and prevents RAM spikes in Kaggle.
- All source files must import and use DataManager for any data access.

### Project Structure
# TODO: This needs updating; it is incompleete 
- `kaggle_notebook.py`: Main development file containing all code
- `src/core/data_manager.py`: **Single source of truth for all data IO**
- `requirements.txt`: Project dependencies

### Features
- **Physics-Guided Architecture**: Uses spectral projectors to inject wave-equation priors
- **Mixed Precision Training**: Supports both fp16 and bfloat16 for efficient training
- **Memory Optimization**: Implements efficient attention and memory management
- **Family-Aware Training**: Stratified sampling by geological families
- **Robust Training**: Includes gradient clipping, early stopping, and learning rate scheduling
- **Comprehensive Checkpointing**: Saves full training state for easy resumption

### Usage
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

### Quick Start
1. Create a new notebook in the [Waveform Inversion competition](https://www.kaggle.com/competitions/waveform-inversion)
2. **Important**: Add the required datasets to your notebook first:
   - Click on the 'Data' tab
   - Click 'Add Data'
   - Search for and add:
     1. 'Waveform Inversion' competition dataset
     2. 'openfwi-preprocessed-72x72' dataset (contains preprocessed data and pretrained models)
3. Copy code chunks from `kaggle_notebook.py` into separate cells in your Kaggle notebook for testing
4. For final submission, copy the entire contents of `kaggle_notebook.py` into a single cell

---

## 4. Colab Instructions

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
3. Develop and test your code locally in `kaggle_notebook.py`
4. Copy relevant code chunks to Colab notebook cells as needed
5. Test functionality in Colab environment

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



