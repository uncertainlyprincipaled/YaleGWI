import os
import subprocess
from pathlib import Path
import shutil
import boto3
from botocore.exceptions import ClientError
import logging
import time
from typing import Optional
import kagglehub  # Optional import
import json
import sys
import torch
# from dotenv import load_dotenv

def get_project_root() -> Path:
    """Get the project root directory, handling both script and notebook environments."""
    try:
        # Try to get the path from __file__ (works in scripts)
        return Path(__file__).parent.parent.parent
    except NameError:
        # In notebook environment, use current working directory
        return Path.cwd()

def push_to_kaggle(artefact_dir: Path, message: str, dataset: str = "jdmorgan/yalegwi"):
    """Upload artefact_dir (or its contents) to S3, using credentials from .env/aws."""
    logging.info("Starting push_to_kaggle...")
    # load_dotenv(dotenv_path=Path('.env/aws'))
    bucket = os.environ.get('AWS_S3_BUCKET')
    region = os.environ.get('AWS_REGION', 'us-east-1')
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    logging.info(f"Read env: bucket={bucket}, region={region}, access_key={'set' if aws_access_key_id else 'missing'}, secret_key={'set' if aws_secret_access_key else 'missing'}")
    if not bucket or not aws_access_key_id or not aws_secret_access_key:
        logging.error("Missing AWS S3 configuration in environment variables")
        return
    try:
        logging.info("Creating boto3 S3 client...")
        s3 = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        logging.info("S3 client created successfully.")
    except Exception as e:
        logging.error(f"Failed to create S3 client: {e}")
        return
    artefact_dir = Path(artefact_dir)
    for file_path in artefact_dir.glob('*'):
        if file_path.is_file():
            s3_key = f"yalegwi/{file_path.name}"
            logging.info(f"Preparing to upload {file_path} to s3://{bucket}/{s3_key}")
            try:
                s3.upload_file(str(file_path), bucket, s3_key)
                logging.info(f"Uploaded {file_path} to s3://{bucket}/{s3_key}")
            except Exception as e:
                logging.error(f"Failed to upload {file_path} to S3: {e}")
    logging.info("push_to_kaggle completed.")

def warm_kaggle_cache():
    """Warm up the Kaggle FUSE cache by creating a temporary tar archive."""
    data_dir = Path('/kaggle/input/waveform-inversion')
    tmp_tar = Path('/kaggle/working/tmp.tar.gz')
    
    # Check if data directory exists
    if not data_dir.exists():
        print("Warning: Competition data not found at /kaggle/input/waveform-inversion")
        print("Please add the competition dataset to your notebook first:")
        print("1. Click on the 'Data' tab")
        print("2. Click 'Add Data'")
        print("3. Search for 'Waveform Inversion'")
        print("4. Click 'Add' on the competition dataset")
        return
        
    try:
        subprocess.run([
            'tar', '-I', 'pigz', '-cf', str(tmp_tar),
            str(data_dir)
        ], check=True)
        tmp_tar.unlink()  # Clean up
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to warm up cache: {e}")
        print("This is not critical - continuing with setup...")
    except Exception as e:
        print(f"Warning: Unexpected error during cache warmup: {e}")
        print("This is not critical - continuing with setup...")

def setup_environment(env: str = 'kaggle'):
    """Setup environment-specific configurations and download datasets if needed.
    
    Args:
        env: Environment to setup. One of: 'kaggle', 'colab', 'sagemaker', 'aws', 'local'
    """
    env = env.lower()
    valid_envs = ['kaggle', 'colab', 'sagemaker', 'aws', 'local']
    if env not in valid_envs:
        raise ValueError(f"Invalid environment: {env}. Valid values: {', '.join(valid_envs)}")
    
    # from src.core.config import CFG

    def setup_aws_environment():
        """Setup AWS-specific environment configurations."""
        
        # Setup S3 client
        s3 = boto3.client('s3', region_name=CFG.env.aws_region)
        
        # Setup paths
        CFG.paths.root = CFG.env.ebs_mount / 'waveform-inversion'
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        CFG.paths.out = CFG.env.ebs_mount / 'output'
        
        # Create directories
        for path in [CFG.paths.root, CFG.paths.train, CFG.paths.test, CFG.paths.out]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Sync data from S3
        try:
            subprocess.run([
                "aws", "s3", "sync",
                f"s3://{CFG.env.s3_bucket}/raw/", str(CFG.paths.root)
            ], check=True)
        except ClientError as e:
            logging.error(f"Failed to sync data from S3: {e}")
            raise

    # Allow explicit environment override
    if env:
        CFG.env.kind = env
        if CFG.env.kind == 'aws' and hasattr(CFG.env, 'set_aws_attributes'):
            CFG.env.set_aws_attributes()

    # --- Load config.json if present (for AWS) ---
    config_path = Path('outputs/config.json')
    if CFG.env.kind == 'aws' and config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        # Override CFG values with those from config.json
        for k, v in config_data.items():
            if hasattr(CFG, k):
                setattr(CFG, k, v)
            elif hasattr(CFG.env, k):
                setattr(CFG.env, k, v)
            elif hasattr(CFG.paths, k):
                setattr(CFG.paths, k, v)
        print(f"Loaded configuration from {config_path}")

    # Common path setup for all environments
    def setup_paths(base_dir: Path):
        CFG.paths.root = base_dir / 'waveform-inversion'
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        CFG.paths.families = {
            'FlatVel_A'   : CFG.paths.train/'FlatVel_A',
            'FlatVel_B'   : CFG.paths.train/'FlatVel_B',
            'CurveVel_A'  : CFG.paths.train/'CurveVel_A',
            'CurveVel_B'  : CFG.paths.train/'CurveVel_B',
            'Style_A'     : CFG.paths.train/'Style_A',
            'Style_B'     : CFG.paths.train/'Style_B',
            'FlatFault_A' : CFG.paths.train/'FlatFault_A',
            'FlatFault_B' : CFG.paths.train/'FlatFault_B',
            'CurveFault_A': CFG.paths.train/'CurveFault_A',
            'CurveFault_B': CFG.paths.train/'CurveFault_B',
        }

    if env == 'aws':
        if hasattr(CFG.env, 'set_aws_attributes'):
            CFG.env.set_aws_attributes()
        setup_aws_environment()
        print("Environment setup complete for AWS")
    elif env == 'colab':
        # Create data directory
        data_dir = Path('/content/data')
        data_dir.mkdir(exist_ok=True)
        setup_paths(data_dir)

        # Setup S3 client and sync data
        try:
            s3 = boto3.client('s3', region_name=CFG.env.aws_region)
            print("Syncing data from S3...")
            s3.sync(f's3://{CFG.env.s3_bucket}/raw/', str(CFG.paths.root))
            print("S3 data sync complete")
        except ClientError as e:
            logging.error(f"Failed to sync data from S3: {e}")
            raise
        
        print("Environment setup complete for Colab")
    elif env == 'sagemaker':
        # AWS SageMaker specific setup
        data_dir = Path('/opt/ml/input/data')  
        data_dir.mkdir(exist_ok=True)

        # Create a symbolic link to the dataset
        dataset_path = Path('/opt/ml/input/data/waveform-inversion')
        dataset_path.symlink_to(data_dir / 'waveform-inversion')

        # Download dataset
        print("Downloading dataset from Kaggle...")
        kagglehub.model_download('jamie-morgan/waveform-inversion', path=str(data_dir))
        
        setup_paths(data_dir)
        print("Paths configured for SageMaker environment")
    elif env == 'kaggle':
        # In Kaggle, warm up the FUSE cache first
        # warm_kaggle_cache()
        # Use the competition data path directly
        CFG.paths.root = Path('/kaggle/input/waveform-inversion')
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        
        # Update family paths
        CFG.paths.families = {
            'FlatVel_A'   : CFG.paths.train/'FlatVel_A',
            'FlatVel_B'   : CFG.paths.train/'FlatVel_B',
            'CurveVel_A'  : CFG.paths.train/'CurveVel_A',
            'CurveVel_B'  : CFG.paths.train/'CurveVel_B',
            'Style_A'     : CFG.paths.train/'Style_A',
            'Style_B'     : CFG.paths.train/'Style_B',
            'FlatFault_A' : CFG.paths.train/'FlatFault_A',
            'FlatFault_B' : CFG.paths.train/'FlatFault_B',
            'CurveFault_A': CFG.paths.train/'CurveFault_A',
            'CurveFault_B': CFG.paths.train/'CurveFault_B',
        }
        print("Environment setup complete for Kaggle")
    elif env == 'local':
        # For local development, use a data directory in the project root
        data_dir = get_project_root() / 'data'
        data_dir.mkdir(exist_ok=True)
        setup_paths(data_dir)
        print("Environment setup complete for local development")

def verify_openfwi_setup():
    """Verify that OpenFWI dataset and weights are properly loaded."""
    
    # Check weights file
    weights_path = Path('/mnt/waveform-inversion/openfwi_backbone.pth')
    if not weights_path.exists():
        logging.error("OpenFWI weights not found!")
        return False
        
    # Try loading weights
    try:
        
        state_dict = torch.load(weights_path, map_location='cpu')
        logging.info(f"Successfully loaded OpenFWI weights with {len(state_dict)} layers")
        
        # Verify model can use these weights
        from src.core.model import get_model
        model = get_model()
        try:
            model.backbone.load_state_dict(state_dict, strict=False)
            logging.info("Successfully loaded weights into model backbone")
            return True
        except Exception as e:
            logging.error(f"Failed to load weights into model: {e}")
            return False
    except Exception as e:
        logging.error(f"Failed to verify OpenFWI weights: {e}")
        return False

if __name__ == "__main__":
    setup_environment('kaggle') 