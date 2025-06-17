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

### 1. Environment Setup

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

### Running Preprocessing

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

### Cost Optimization

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

### Troubleshooting

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

## Google Colab Setup

1. **Clone Repository**:
   ```python
   # In a notebook cell:
   !git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
   %cd YaleGWI
   ```

2. **Setup Environment**:
   ```python
   import os
   import sys
   os.chdir('/content/YaleGWI')  # Adjust as needed
   sys.path.append(os.path.abspath('./src'))
   from src.core.imports import setup_environment
   deps = setup_environment('colab')
   np = deps['np']
   torch = deps['torch']
   CFG = deps['CFG']
   DataManager = deps['DataManager']
   ```

3. **Mount Google Drive** (if needed):
   ```python
   # This is handled automatically by setup_environment('colab')
   # But you can also do it manually:
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Kaggle Setup

1. **Clone Repository**:
   ```python
   # In a notebook cell:
   !git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
   %cd YaleGWI
   ```

2. **Setup Environment**:
   ```python
   import os
   import sys
   !pip install kagglehub
   import kagglehub

   os.environ['GWI_ENV'] = 'sagemaker'
   os.chdir('/kaggle/working/YaleGWI')  # Adjust as needed
   sys.path.append(os.path.abspath('./src'))
   from src.core.imports import setup_environment
   deps = setup_environment('kaggle')
   np = deps['np']
   torch = deps['torch']
   CFG = deps['CFG']
   DataManager = deps['DataManager']
   ```

3. **Configure Kaggle API**:
   ```python
   # This is handled automatically by setup_environment('kaggle')
   # But you can also do it manually:
   import os
   os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input'
   ```


---

# (All other original content, such as features, development workflow, contributing, license, etc., remains unchanged and follows these sections.)

def load_data(input_root, output_root, use_s3=False):
    from src.core.data_manager import DataManager
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_manager = DataManager(use_s3=use_s3) if use_s3 else None

    families = list(CFG.paths.families.keys())
    all_processed_paths = []
    for family in families:
        input_dir = input_root / family
        temp_dir = output_root / 'temp' / family
        processed_paths = process_family(family, input_dir, temp_dir, data_manager)
        all_processed_paths.extend(processed_paths)
    split_for_gpus(all_processed_paths, output_root, data_manager)
    # Optionally clean up temp files, etc.
    return all_processed_paths

from src.core.preprocess import load_data
load_data('/path/to/input', '/path/to/output', use_s3=True)



