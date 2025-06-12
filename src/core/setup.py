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
from src.core.config import CFG
import json

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

def setup_aws_environment():
    """Setup AWS-specific environment configurations."""
    from src.core.config import CFG
    
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
        s3.sync(f's3://{CFG.env.s3_bucket}/raw/', str(CFG.paths.root))
    except ClientError as e:
        logging.error(f"Failed to sync data from S3: {e}")
        raise

def push_to_kaggle(artefact_dir: Path, message: str, dataset: str = "uncertainlyprincipaled/yalegwi"):
    """Push training artefacts to Kaggle dataset with rate limiting awareness."""
    try:
        # Check if kaggle.json exists
        kaggle_json = Path.home() / '.kaggle/kaggle.json'
        if not kaggle_json.exists():
            # Try to load from environment file
            env_kaggle_json = Path(__file__).parent.parent.parent / '.env/kaggle/credentials'
            if env_kaggle_json.exists():
                kaggle_json.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(env_kaggle_json, kaggle_json)
                kaggle_json.chmod(0o600)
            else:
                raise FileNotFoundError("Kaggle credentials not found")
        
        # Push to Kaggle with rate limiting awareness
        max_retries = 3
        retry_delay = 60  # seconds
        
        for attempt in range(max_retries):
            try:
                subprocess.run([
                    "kaggle", "datasets", "version", "-p", str(artefact_dir),
                    "-m", message, "-d", dataset, "--dir-mode", "zip"
                ], check=True)
                break
            except subprocess.CalledProcessError as e:
                if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                    logging.warning(f"Rate limit hit, waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    except Exception as e:
        logging.error(f"Failed to push to Kaggle: {e}")
        raise

def setup_environment():
    """Setup environment-specific configurations and download datasets if needed."""
    from src.core.config import CFG  # Import here to avoid circular dependency
    import json

    # Allow explicit environment override
    env_override = os.environ.get('GWI_ENV', '').lower()
    if env_override:
        CFG.env.kind = env_override

    # --- NEW: Load config.json if present (for AWS) ---
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

    if CFG.env.kind == 'aws':
        setup_aws_environment()
        print("Environment setup complete for AWS")
    elif CFG.env.kind == 'colab':
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
    elif CFG.env.kind == 'sagemaker':
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
    elif CFG.env.kind == 'kaggle':
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
    else:  # local development
        # For local development, use a data directory in the project root
        data_dir = Path(__file__).parent.parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)
        setup_paths(data_dir)
        print("Environment setup complete for local development") 