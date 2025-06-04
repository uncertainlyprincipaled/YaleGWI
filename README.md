# SpecProj-UNet for Seismic Waveform Inversion

## Overview
A deep learning solution for seismic waveform inversion using a HGNet/ConvNeXt backbone with:
- EMA (Exponential Moving Average) for stable training
- AMP (Automatic Mixed Precision) for faster training
- Distributed Data Parallel (DDP) for multi-GPU training
- Test Time Augmentation (TTA) for improved inference

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
1. Train the model:
```bash
# Single GPU
python src/core/train.py

# Multi-GPU
torchrun --nproc_per_node=N src/core/train.py
```

2. Generate predictions:
```bash
python src/core/infer.py
```

## Environment Setup

### Local Development
1. Clone the repository:
```bash
git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
cd YaleGWI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset manually or using `kagglehub` as needed.

### Configuration
The following flags can be set in `src/core/config.py` or via environment variables:

- `enable_joint`: Enable joint forward-inverse paradigm (default: False)
- `latent_h`: Height of latent space grid (default: 16)
- `latent_w`: Width of latent space grid (default: 16)
- `lambda_fwd`: Weight for forward loss term (default: 1.0)
- `lambda_inv`: Weight for inverse loss term (default: 1.0)

Example environment variable usage:
```bash
export GWI_ENABLE_JOINT=true
export GWI_LATENT_H=32
export GWI_LATENT_W=32
```

### Cloud Environments

#### Google Colab
1. Install Kaggle CLI and set up authentication:
```python
# Install Kaggle CLI if needed
!pip install kaggle

# Upload your kaggle.json (API token) if not already present
from google.colab import files
files.upload()  # Then move kaggle.json to ~/.kaggle/

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

2. Download data and set up environment:
```python
# Download the competition data
!kaggle competitions download -c waveform-inversion
!unzip -q waveform-inversion.zip -d /content/data

!git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI
```

#### Alternative: Google Drive Setup (Recommended)
1. In Kaggle:
```bash
# Install rclone
!pip install -q rclone

# Configure Google Drive (run this once)
!rclone config
# Follow the prompts to set up a new remote named "gdrive"

# Run the push script
!python3 src/utils/push_to_drive.py
```

2. In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

# Unzip once
!unzip -q /content/drive/MyDrive/<GDRIVE_FOLDER>/waveform-inversion.zip -d /content/data

# Clone the repository
!git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI

# Install dependencies
!pip install -r requirements.txt
```

#### Kaggle Notebooks
1. Create a new notebook in the [Waveform Inversion competition](https://www.kaggle.com/competitions/waveform-inversion)

2. Add the following code to your first cell:
```python
# Clone repository (ignore error if already exists)
!git clone https://github.com/uncertainlyprincipaled/YaleGWI.git || true
%cd YaleGWI

# Install dependencies
!pip install -r requirements.txt

# Import the notebook utility
from src.utils.notebook_utils import setup_kaggle_notebook

# This will:
# 1. Set up the environment
# 2. Create and populate cells from kaggle_notebook.py
# 3. Execute each cell in sequence
setup_kaggle_notebook()
```

#### AWS SageMaker
- Clone the repository to `/opt/ml/code/YaleGWI`
- Dataset should be placed in `/opt/ml/input/data`

## Development Workflow

### Automatic Notebook Updates
The project includes an automatic update system that keeps the Kaggle notebook (`kaggle_notebook.py`) in sync with changes to the source files.

1. **Manual Update**:
```bash
python src/utils/update_kaggle_notebook.py
```

2. **Automatic Updates**:
```bash
python src/utils/watch_and_update.py
```

### Development Process
1. Make changes to the source files in `src/core/`
2. Changes will be automatically reflected in `kaggle_notebook.py` if the file watcher is running
3. If the file watcher is not running, manually run the update script
4. Test your changes in the Kaggle notebook

## Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```


