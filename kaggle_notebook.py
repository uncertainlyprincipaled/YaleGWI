# %%
# Source: header
# SpecProj-UNet for Seismic Waveform Inversion
# This notebook implements a physics-guided neural network for seismic waveform inversion
# using spectral projectors and UNet architecture.


# %%
# Source: imports

# Standard library imports
from __future__ import annotations
import os
import sys
import json
import time
import signal
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Literal, NamedTuple

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import timm
import kagglehub  # Optional import

# Local imports
from src.core.config import CFG
    


# %%
# Source: config.py
# ## Configuration and Environment Setup



# %%
# Source: config.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Literal, NamedTuple, Optional
import torch
import boto3
from botocore.exceptions import ClientError
import logging
import numpy as np
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
#  Detect runtime (Kaggle / Colab / SageMaker / AWS / local) and expose a singleton #
# --------------------------------------------------------------------------- #

class _KagglePaths:
    def __init__(self):
        self.root: Path = Path('/kaggle/input/waveform-inversion')
        self.train: Path = self.root / 'train_samples'
        self.test: Path = self.root / 'test'
        # folders visible in the screenshot
        self.families = {
            'FlatVel_A'   : self.train/'FlatVel_A',
            'FlatVel_B'   : self.train/'FlatVel_B',
            'CurveVel_A'  : self.train/'CurveVel_A',
            'CurveVel_B'  : self.train/'CurveVel_B',
            'Style_A'     : self.train/'Style_A',
            'Style_B'     : self.train/'Style_B',
            'FlatFault_A' : self.train/'FlatFault_A',
            'FlatFault_B' : self.train/'FlatFault_B',
            'CurveFault_A': self.train/'CurveFault_A',
            'CurveFault_B': self.train/'CurveFault_B',
        }
        # Add AWS-specific paths
        self.aws_root: Optional[Path] = None
        self.aws_train: Optional[Path] = None
        self.aws_test: Optional[Path] = None
        self.aws_output: Optional[Path] = None

class _Env:
    def __init__(self):
        if 'KAGGLE_URL_BASE' in os.environ:
            self.kind: Literal['kaggle','colab','sagemaker','aws','local'] = 'kaggle'
        elif 'COLAB_GPU' in os.environ:
            self.kind = 'colab'
        elif 'SM_NUM_CPUS' in os.environ:
            self.kind = 'sagemaker'
        elif 'AWS_EXECUTION_ENV' in os.environ or 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
            self.kind = 'aws'
        else:
            self.kind = 'local'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

class Config:
    """Read-only singleton accessed via `CFG`."""
    _inst = None
    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
            cls._inst.env   = _Env()
            cls._inst.paths = _KagglePaths()
            cls._inst.seed  = 42

            # Debug settings
            cls._inst.debug_mode = os.environ.get('DEBUG_MODE', '0') == '1'
            cls._inst._update_debug_settings()
            
            # Common parameters
            cls._inst.lr = 1e-4
            cls._inst.weight_decay = 1e-3
            cls._inst.lambda_pde = 0.1
            cls._inst.dtype = "float16"
            cls._inst.distributed = False
            cls._inst.memory_efficient = True

            # Model parameters
            cls._inst.backbone = "hgnetv2_b2.ssld_stage2_ft_in1k"
            cls._inst.ema_decay = 0.99
            cls._inst.pretrained = True

            # Inference weight path (default for Kaggle dataset)
            cls._inst.weight_path = "/kaggle/input/yalegwi/best.pth"
            
            # Loss weights
            cls._inst.lambda_inv = 1.0
            cls._inst.lambda_fwd = 1.0
            cls._inst.lambda_pde = 0.1

            # Enable joint training by default in Kaggle
            cls._inst.enable_joint = cls._inst.env.kind == 'kaggle'

            # Always use base dataset for now
            train = cls._inst.paths.train
            cls._inst.paths.families = {
                'FlatVel_A'   : train/'FlatVel_A',
                'FlatVel_B'   : train/'FlatVel_B',
                'CurveVel_A'  : train/'CurveVel_A',
                'CurveVel_B'  : train/'CurveVel_B',
                'Style_A'     : train/'Style_A',
                'Style_B'     : train/'Style_B',
                'FlatFault_A' : train/'FlatFault_A',
                'FlatFault_B' : train/'FlatFault_B',
                'CurveFault_A': train/'CurveFault_A',
                'CurveFault_B': train/'CurveFault_B',
            }
            cls._inst.dataset_style = 'yalegwi'

        return cls._inst

    def _update_debug_settings(self):
        """Update training parameters based on debug mode."""
        if self.debug_mode:
            logging.info("Updating settings for debug mode")
            # Reduced training parameters for debug
            self.batch = 4  # Smaller batch size
            self.epochs = 2  # Just 2 epochs
            self.num_workers = 0  # Single worker for easier debugging
            self.use_amp = False  # Disable mixed precision
            self.gradient_checkpointing = False  # Disable gradient checkpointing
            # Use only one family for testing
            self.families_to_exclude = list(self.paths.families.keys())[1:]
            self.debug_upload_interval = 1  # Upload every epoch in debug mode
        else:
            # Normal training parameters
            self.batch = 32 if self.env.kind == 'kaggle' else 32
            self.epochs = 30
            self.num_workers = 4
            self.use_amp = True
            self.gradient_checkpointing = True
            self.families_to_exclude = []
            self.debug_upload_interval = 5  # Upload every 5 epochs in normal mode

    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode and update settings accordingly."""
        self.debug_mode = enabled
        self._update_debug_settings()
        logging.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        logging.info(f"Updated settings: batch={self.batch}, epochs={self.epochs}, workers={self.num_workers}")

    def is_joint(self) -> bool:
        """Helper to check if joint forward-inverse mode is enabled."""
        return self.enable_joint

    def torch_dtype(self):
        """Helper to get the torch dtype based on config."""
        return getattr(torch, self.dtype)

CFG = Config()

def save_cfg(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert CFG to a serializable dictionary
    cfg_dict = {}
    for k, v in CFG.__dict__.items():
        if k.startswith('_'):
            continue
        if isinstance(v, _Env):
            cfg_dict[k] = {
                'kind': v.kind,
                'device': v.device,
                'world_size': v.world_size
            }
        elif isinstance(v, _KagglePaths):
            cfg_dict[k] = {
                'root': str(v.root),
                'train': str(v.train),
                'test': str(v.test),
                'families': {k: str(p) for k, p in v.families.items()}
            }
        else:
            cfg_dict[k] = v
            
    (out_dir / 'config.json').write_text(
        json.dumps(cfg_dict, indent=2)
    )

class SeismicDataset(Dataset):
    def __init__(self, seis_files, vel_files, family_type, augment=False, use_mmap=True, memory_tracker=None):
        self.family_type = family_type
        self.augment = augment
        self.index = []
        self.use_mmap = use_mmap
        self.memory_tracker = memory_tracker
        self.vel_files = vel_files
        # Build index of (file, sample_idx, source_idx) triples
        for sfile, vfile in zip(seis_files, vel_files):
            f = np.load(sfile, mmap_mode='r')
            n_samples, n_sources, *_ = f.shape  # e.g., (batch, sources, ...)
            for i in range(n_samples):
                for s in range(n_sources):
                    self.index.append((sfile, vfile, i, s))
            del f

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sfile, vfile, i, _ = self.index[idx]
        # Load all sources for this sample
        seis = np.load(sfile, mmap_mode='r')[i]  # shape: (sources, receivers, timesteps)
        vel = np.load(vfile, mmap_mode='r')[i]   # shape: (1, 70, 70)
        seis = seis.astype(np.float16)           # shape: (5, receivers, timesteps)
        vel = vel.astype(np.float16)
        # Normalize per source
        mu = seis.mean(axis=(1,2), keepdims=True)
        std = seis.std(axis=(1,2), keepdims=True) + 1e-6
        seis = (seis - mu) / std
        if self.memory_tracker:
            self.memory_tracker.update(seis.nbytes + vel.nbytes)
        return torch.from_numpy(seis), torch.from_numpy(vel) 


# %%
# Source: setup.py
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


# %%
# Source: data_manager.py
"""
DataManager is the single source of truth for all data IO in this project.
All data loading, streaming, and batching must go through DataManager.
"""
# Standard library imports
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import os
import socket
import tempfile
import shutil
import mmap
import logging

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import boto3
from botocore.exceptions import ClientError

# Local imports
# from src.core.config import CFG
# from src.core.setup import push_to_kaggle

class MemoryTracker:
    """Tracks memory usage during data loading."""
    def __init__(self):
        self.total_memory = 0
        self.max_memory = 0
        
    def update(self, memory_used: int):
        self.total_memory += memory_used
        self.max_memory = max(self.max_memory, memory_used)
        
    def get_stats(self) -> Dict[str, float]:
        return {
            'total_memory_gb': self.total_memory / 1e9,
            'max_memory_gb': self.max_memory / 1e9
        }

class S3DataLoader:
    """Handles efficient data loading from S3 with local caching."""
    def __init__(self, bucket: str, region: str):
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = bucket
        self.cache_dir = Path(tempfile.gettempdir()) / 'gwi_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, s3_key: str, local_path: Path) -> Path:
        """Download a file from S3 with caching."""
        cache_path = self.cache_dir / s3_key.replace('/', '_')
        
        if not cache_path.exists():
            try:
                self.s3.download_file(self.bucket, s3_key, str(cache_path))
            except ClientError as e:
                logging.error(f"Failed to download {s3_key} from S3: {e}")
                raise
                
        # Create a hard link to avoid copying
        if not local_path.exists():
            os.link(cache_path, local_path)
            
        return local_path

class DataManager:
    """
    DataManager is the single source of truth for all data IO in this project.
    Handles memory-efficient, sample-wise access for all dataset families.
    Uses float16 for memory efficiency.
    """
    def __init__(self, use_mmap: bool = True):
        self.use_mmap = use_mmap
        self.memory_tracker = MemoryTracker()
        
        # Initialize S3 loader if in AWS environment
        if CFG.env.kind == 'aws':
            self.s3_loader = S3DataLoader(CFG.env.s3_bucket, CFG.env.aws_region)
        else:
            self.s3_loader = None
            
        if CFG.debug_mode:
            logging.info("DataManager initialized in debug mode")
            logging.info(f"Using only family: {list(CFG.paths.families.keys())[0]}")
            logging.info(f"Batch size: {CFG.batch}")
            logging.info(f"Number of workers: {CFG.num_workers}")

    def list_family_files(self, family: str):
        """Return (seis_files, vel_files, family_type) for a given family (base dataset only)."""
        if CFG.debug_mode and family in CFG.families_to_exclude:
            logging.info(f"Skipping excluded family in debug mode: {family}")
            return None, None, None
            
        root = CFG.paths.families[family]
        if not root.exists():
            raise ValueError(f"Family directory not found: {root}")
            
        # Vel/Style: data/model subfolders (batched)
        if (root / 'data').exists() and (root / 'model').exists():
            seis_files = sorted((root/'data').glob('*.npy'))
            vel_files = sorted((root/'model').glob('*.npy'))
            if seis_files and vel_files:
                if CFG.debug_mode:
                    # In debug mode, limit to first file only
                    seis_files = seis_files[:1]
                    vel_files = vel_files[:1]
                    logging.info(f"Debug mode: Using only first file from {family}")
                family_type = 'VelStyle'
                return seis_files, vel_files, family_type
                
        # Fault: seis*.npy and vel*.npy directly in folder (not batched)
        seis_files = sorted(root.glob('seis*.npy'))
        vel_files = sorted(root.glob('vel*.npy'))
        if seis_files and vel_files:
            if CFG.debug_mode:
                # In debug mode, limit to first file only
                seis_files = seis_files[:1]
                vel_files = vel_files[:1]
                logging.info(f"Debug mode: Using only first file from {family}")
            family_type = 'Fault'
            return seis_files, vel_files, family_type
            
        raise ValueError(f"Could not find valid data structure for family {family} at {root}")

    def create_dataset(self, seis_files, vel_files, family_type, augment=False):
        """Create a dataset for the given files."""
        if family_type == 'test':
            return TestDataset(seis_files)
        return SeismicDataset(
            seis_files,
            vel_files,
            family_type,
            augment,
            use_mmap=self.use_mmap,
            memory_tracker=self.memory_tracker
        )

    def create_loader(self, seis_files: List[Path], vel_files: List[Path],
                     family_type: str, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 4,
                     distributed: bool = False) -> Optional[DataLoader]:
        """Create a DataLoader for the given files. Returns None if files are empty."""
        if not seis_files:
            logging.info("No files to load, skipping loader creation")
            return None
            
        # For test data, vel_files can be None
        if family_type != 'test' and not vel_files:
            logging.info("No velocity files to load, skipping loader creation")
            return None
            
        dataset = self.create_dataset(seis_files, vel_files, family_type)
        
        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and not distributed,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )

    def get_test_files(self) -> List[Path]:
        return sorted(CFG.paths.test.glob('*.npy'))

    def upload_best_model_and_metadata(self, artefact_dir: Path, message: str = "Update best model"):
        logging.info(f"Starting upload_best_model_and_metadata with artefact_dir={artefact_dir}, message={message}")
        if CFG.debug_mode:
            logging.info("Debug mode: Simulating model upload")
            logging.info(f"Would upload from: {artefact_dir}")
            logging.info(f"With message: {message}")
            return
        if CFG.env.kind == "kaggle":
            if not artefact_dir.exists():
                logging.error(f"Artefact directory not found: {artefact_dir}")
                return
            logging.info("Calling push_to_kaggle for model upload...")
            push_to_kaggle(artefact_dir, message)
            logging.info(f"Successfully uploaded to Kaggle dataset with message: {message}")
            return
        if CFG.env.kind == "aws":
            logging.info("AWS environment stub - not implemented.")
            return
        if CFG.env.kind == "colab":
            logging.info("Colab environment stub - not implemented.")
            return
        if CFG.env.kind == "sagemaker":
            logging.info("SageMaker environment stub - not implemented.")
            return
        logging.info("Unknown environment stub - not implemented.")
        return

class SeismicDataset(Dataset):
    """
    Memory-efficient dataset for all families.
    Handles both batched (500 samples) and single-sample files.
    Data shapes:
    - Seismic: (batch, sources=5, receivers=1000, timesteps=70)
    - Velocity: (batch, channels=1, height=70, width=70)
    Uses float16 for memory efficiency.
    """
    def __init__(self, seis_files: List[Path], vel_files: Optional[List[Path]], family_type: str, 
                 augment: bool = False, use_mmap: bool = True, memory_tracker: MemoryTracker = None):
        self.family_type = family_type
        self.augment = augment
        self.index = []
        self.use_mmap = use_mmap
        self.memory_tracker = memory_tracker
        self.vel_files = vel_files
        # Build index of (file, sample_idx) pairs
        if vel_files is None:
            for sfile in seis_files:
                if self.use_mmap:
                    f = np.load(sfile, mmap_mode='r')
                    shape = f.shape
                    n_samples = shape[0] if len(shape) == 4 else 1
                    del f
                else:
                    data = np.load(sfile)
                    shape = data.shape
                    n_samples = shape[0] if len(shape) == 4 else 1
                for i in range(n_samples):
                    self.index.append((sfile, None, i))
        else:
            for sfile, vfile in zip(seis_files, vel_files):
                if self.use_mmap:
                    f = np.load(sfile, mmap_mode='r')
                    shape = f.shape
                    n_samples = shape[0] if len(shape) == 4 else 1
                    del f
                else:
                    data = np.load(sfile)
                    shape = data.shape
                    n_samples = shape[0] if len(shape) == 4 else 1
                for i in range(n_samples):
                    self.index.append((sfile, vfile, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sfile, vfile, i = self.index[idx]
        # Load x
        if self.use_mmap:
            x = np.load(sfile, mmap_mode='r')
        else:
            x = np.load(sfile)
        # x shape: (batch_size, num_sources, time_steps, num_receivers)
        if len(x.shape) == 4:
            x = x[i]  # shape: (num_sources, time_steps, num_receivers)
        # Rearrange to (num_sources, num_receivers, time_steps)
        if x.shape[1] == 70 and x.shape[2] == 1000:
            # Already (num_sources, time_steps, num_receivers), need (num_sources, num_receivers, time_steps)
            x = x.transpose(0, 2, 1)  # (num_sources, num_receivers, time_steps)
        elif x.shape[1] == 1000 and x.shape[2] == 70:
            # Already correct
            pass
        else:
            raise ValueError(f"Unexpected seismic data shape: {x.shape}")
        # For test mode, create dummy y
        if self.vel_files is None:
            y = np.zeros((1, 70, 70), np.float16)
        else:
            if self.use_mmap:
                y = np.load(vfile, mmap_mode='r')
            else:
                y = np.load(vfile)
            if len(y.shape) == 4:
                y = y[i]
            elif len(y.shape) == 3:
                y = y[i]
            else:
                raise ValueError(f"Unexpected velocity data shape: {y.shape}")
        # Convert to float16
        x = x.astype(np.float16)
        y = y.astype(np.float16)
        # Normalize per-receiver
        mu = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-6
        x = (x - mu) / std
        # No need to add extra dimension; DataLoader will batch
        # Track memory usage
        if self.memory_tracker:
            self.memory_tracker.update(x.nbytes + y.nbytes)
        return torch.from_numpy(x), torch.from_numpy(y)

class TestDataset(Dataset):
    """Dataset for test files that returns the test data and its identifier."""
    def __init__(self, files: List[Path]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        data = np.load(file_path)
        # Convert to float16 and normalize per-receiver
        data = data.astype(np.float16)
        mu = data.mean(axis=(1,2), keepdims=True)
        std = data.std(axis=(1,2), keepdims=True) + 1e-6
        data = (data - mu) / std
        return torch.from_numpy(data), file_path.stem 


# %%
# Source: eda.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from data_manager import DataManager
# from config import CFG
from pathlib import Path
from typing import Dict, List, Tuple
# from src.core.setup import setup_environment  # Ensure correct paths
setup_environment()  # Set up paths before any EDA code

def summarize_array(arr):
    # Convert to float32 for statistics to avoid overflow
    arr_float32 = arr.astype(np.float32)
    stats = {
        'min': np.nanmin(arr_float32),
        'max': np.nanmax(arr_float32),
        'mean': np.nanmean(arr_float32),
        'std': np.nanstd(arr_float32),
        'nan_count': np.isnan(arr).sum(),
        'inf_count': np.isinf(arr).sum(),
        'shape': arr.shape
    }
    return stats

def eda_on_family(family, n_shape_samples=3):
    dm = DataManager()
    seis_files, vel_files, family_type = dm.list_family_files(family)
    seis_stats = []
    vel_stats = []
    seis_shapes = []
    vel_shapes = []
    for idx, (sfile, vfile) in enumerate(zip(seis_files, vel_files)):
        if idx >= 3:  # Only process first 3 files per family
            break
        sdata = np.load(sfile, mmap_mode='r')
        vdata = np.load(vfile, mmap_mode='r')
        # Handle both batched and per-sample files
        if sdata.ndim == 3:  # (sources, receivers, timesteps)
            seis_stats.append(summarize_array(sdata))
            vel_stats.append(summarize_array(vdata))
            seis_shapes.append(sdata.shape)
            vel_shapes.append(vdata.shape)
        elif sdata.ndim == 4:  # (batch, sources, receivers, timesteps)
            n_samples = sdata.shape[0]
            n_pick = min(10, n_samples)
            if n_samples > 1:
                pick_indices = np.linspace(0, n_samples - 1, n_pick, dtype=int)
            else:
                pick_indices = [0]
            for i in pick_indices:
                seis_stats.append(summarize_array(sdata[i]))
                vel_stats.append(summarize_array(vdata[i]))
                seis_shapes.append(sdata[i].shape)
                vel_shapes.append(vdata[i].shape)
        else:
            print(f"Unexpected shape for {sfile}: {sdata.shape}")
        # Print a few sample shapes for validation
        if idx < n_shape_samples:
            print(f"Sample {idx} - Seis shape: {sdata.shape}, Vel shape: {vdata.shape}")
    # Shape validation
    expected_seis_shapes = [(5, 72, 72), (500, 5, 72, 72)]  # Updated to match actual shapes
    expected_vel_shapes = [(1, 70, 70), (500, 1, 70, 70)]  # Updated to match actual shapes
    for shape in seis_shapes[:n_shape_samples]:
        if shape not in expected_seis_shapes:
            print(f"[!] Unexpected seis shape: {shape} in family {family}")
    for shape in vel_shapes[:n_shape_samples]:
        if shape not in expected_vel_shapes:
            print(f"[!] Unexpected vel shape: {shape} in family {family}")
    return seis_stats, vel_stats

def print_summary(stats, name):
    print(f"Summary for {name}:")
    for k in ['min', 'max', 'mean', 'std', 'nan_count', 'inf_count']:
        vals = [s[k] for s in stats]
        if k in ['nan_count', 'inf_count']:
            print(f"  {k}: min={np.min(vals)}, max={np.max(vals)}, mean={np.mean(vals)}, sum={np.sum(vals)}")
        else:
            # Use float32 for numerical stability
            vals = np.array(vals, dtype=np.float32)
            print(f"  {k}: min={vals.min()}, max={vals.max()}, mean={vals.mean()}, sum={vals.sum()}")
    print(f"  Total samples: {len(stats)}")

def plot_family_distributions(family_stats: Dict[str, Tuple[List, List]]):
    """Plot distributions of key statistics across families."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Key Statistics Across Families')
    
    # Velocity ranges
    vel_ranges = []
    family_names = []
    for family, (_, vel_stats) in family_stats.items():
        vel_mins = [s['min'] for s in vel_stats]
        vel_maxs = [s['max'] for s in vel_stats]
        vel_ranges.append((np.mean(vel_mins), np.mean(vel_maxs)))
        family_names.append(family)
    
    # Plot velocity ranges
    vel_mins, vel_maxs = zip(*vel_ranges)
    axes[0,0].bar(family_names, vel_mins, label='Min Velocity')
    axes[0,0].bar(family_names, vel_maxs, bottom=vel_mins, label='Max Velocity')
    axes[0,0].set_title('Velocity Ranges by Family')
    axes[0,0].set_ylabel('Velocity (m/s)')
    axes[0,0].legend()
    plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45)
    
    # Seismic amplitude ranges
    seis_ranges = []
    for family, (seis_stats, _) in family_stats.items():
        seis_mins = [s['min'] for s in seis_stats]
        seis_maxs = [s['max'] for s in seis_stats]
        seis_ranges.append((np.mean(seis_mins), np.mean(seis_maxs)))
    
    # Plot seismic ranges
    seis_mins, seis_maxs = zip(*seis_ranges)
    axes[0,1].bar(family_names, seis_mins, label='Min Amplitude')
    axes[0,1].bar(family_names, seis_maxs, bottom=seis_mins, label='Max Amplitude')
    axes[0,1].set_title('Seismic Amplitude Ranges by Family')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].legend()
    plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45)
    
    # Velocity standard deviations
    vel_stds = []
    for family, (_, vel_stats) in family_stats.items():
        stds = [s['std'] for s in vel_stats]
        vel_stds.append(np.mean(stds))
    
    axes[1,0].bar(family_names, vel_stds)
    axes[1,0].set_title('Average Velocity Standard Deviation by Family')
    axes[1,0].set_ylabel('Standard Deviation (m/s)')
    plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45)
    
    # Sample counts
    sample_counts = []
    for family, (seis_stats, _) in family_stats.items():
        sample_counts.append(len(seis_stats))
    
    axes[1,1].bar(family_names, sample_counts)
    axes[1,1].set_title('Number of Samples by Family')
    axes[1,1].set_ylabel('Sample Count')
    plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    return fig

def analyze_family_correlations(family_stats: Dict[str, Tuple[List, List]]):
    """Analyze correlations between seismic and velocity statistics."""
    correlations = {}
    for family, (seis_stats, vel_stats) in family_stats.items():
        # Extract statistics
        seis_means = [s['mean'] for s in seis_stats]
        seis_stds = [s['std'] for s in seis_stats]
        vel_means = [v['mean'] for v in vel_stats]
        vel_stds = [v['std'] for v in vel_stats]
        
        # Calculate correlations
        correlations[family] = {
            'mean_corr': np.corrcoef(seis_means, vel_means)[0,1],
            'std_corr': np.corrcoef(seis_stds, vel_stds)[0,1]
        }
    
    return correlations

def extract_and_plot_geometry(family_stats: Dict[str, Tuple[List, List]]):
    """
    Attempt to extract and plot receiver/source geometry for a few representative families only.
    Assumes seismic data shape is (sources, receivers, timesteps) or (sources, timesteps, receivers).
    """
    print("\nReceiver/Source Geometry per Family:")
    # Only plot for the first 2 families
    max_plots = 2
    plotted = 0
    for family, (seis_stats, _) in family_stats.items():
        if plotted >= max_plots:
            print(f"(Skipping plots for remaining families...)")
            break
        if len(seis_stats) == 0:
            print(f"{family}: No samples found.")
            continue
        shape = seis_stats[0]['shape']
        if len(shape) == 3:
            n_sources, n_receivers, n_timesteps = shape
            print(f"{family}: sources={n_sources}, receivers={n_receivers}, timesteps={n_timesteps}")
            plt.figure()
            plt.title(f"{family} - Receiver/Source Geometry")
            plt.scatter(range(n_receivers), [0]*n_receivers, label='Receivers', marker='x')
            plt.scatter([0]*n_sources, range(n_sources), label='Sources', marker='o')
            plt.xlabel('Receiver Index')
            plt.ylabel('Source Index')
            plt.legend()
            plt.show()
            plt.close()
            plotted += 1
        elif len(shape) == 2:
            n1, n2 = shape
            print(f"{family}: 2D seismic shape: {shape}")
        else:
            print(f"{family}: Unexpected seismic shape: {shape}")


def summarize_array_shapes(family_stats: Dict[str, Tuple[List, List]]):
    """
    Summarize array shapes for seismic and velocity data per family.
    """
    print("\nArray Shape Summary per Family:")
    for family, (seis_stats, vel_stats) in family_stats.items():
        seis_shapes = [tuple(s['shape']) for s in seis_stats]
        vel_shapes = [tuple(s['shape']) for s in vel_stats]
        unique_seis_shapes = set(seis_shapes)
        unique_vel_shapes = set(vel_shapes)
        print(f"{family}: Seismic shapes: {unique_seis_shapes}")
        print(f"{family}: Velocity shapes: {unique_vel_shapes}")
        if len(unique_seis_shapes) > 1 or len(unique_vel_shapes) > 1:
            print(f"  [!] Shape inconsistency detected in {family}")


def summarize_family_sizes(family_stats: Dict[str, Tuple[List, List]]):
    """
    Print a summary of family sizes and highlight imbalances, using the true number of samples per family.
    """
    print("\nFamily Size Summary:")
    from config import CFG
    base_path = CFG.paths.train
    sizes = {family: count_samples_base(base_path, family) for family in family_stats.keys()}
    for family, size in sizes.items():
        print(f"{family}: {size} samples")
    min_size = min(sizes.values())
    max_size = max(sizes.values())
    print(f"\nSmallest family: {min_size} samples, Largest family: {max_size} samples")
    print("Families with < 10 samples:")
    for family, size in sizes.items():
        if size < 10:
            print(f"  [!] {family}: {size} samples (very small)")

# --- New: Supplement/Downsample Check ---
def check_balancing_requirements(target_count=1000):
    """
    For each family, print base and OpenFWI sample counts, and how many to supplement or downsample.
    """
    print("\n=== Data Balancing Requirements ===")
    # from config import CFG
    import os
    # Detect OpenFWI path
    openfwi_path = Path('/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72')
    base_families = list(CFG.paths.families.keys())
    # If OpenFWI is not available, skip
    if not openfwi_path.exists():
        print("OpenFWI dataset not found. Skipping balancing check.")
        return
    print(f"Target samples per family: {target_count}")
    print(f"{'Family':<15} {'Base':>6} {'OpenFWI':>8} {'To Add':>8} {'To Down':>8}")
    print("-"*50)
    for family in base_families:
        # Count base samples
        base_count = count_samples_base(CFG.paths.train, family)
        # Count OpenFWI samples
        openfwi_count = count_samples_openfwi(openfwi_path, family)
        # Compute how many to add or downsample
        to_add = max(0, target_count - base_count)
        to_down = max(0, (base_count + openfwi_count) - target_count) if (base_count + openfwi_count) > target_count else 0
        print(f"{family:<15} {base_count:>6} {openfwi_count:>8} {to_add:>8} {to_down:>8}")
    print("-"*50)

def count_samples_base(base_path, family):
    fam_dir = base_path / family
    # Vel/Style: data/ subfolder with data*.npy files (batched)
    if (fam_dir / 'data').exists():
        data_dir = fam_dir / 'data'
        files = sorted(data_dir.glob('*.npy'))
        total_samples = 0
        for f in files:
            arr = np.load(f, mmap_mode='r')
            total_samples += arr.shape[0]  # Each file: (500, ...)
        return total_samples
    else:
        # Fault: seis*.npy files in family folder (not batched)
        seis_files = sorted(fam_dir.glob('seis*.npy'))
        return len(seis_files)

def count_samples_openfwi(openfwi_path, family):
    fam_dir = openfwi_path / family
    # Look for both seis*.npy and data*.npy files
    files = list(fam_dir.glob('seis*.npy')) + list(fam_dir.glob('data*.npy'))
    total_samples = 0
    for f in files:
        arr = np.load(f, mmap_mode='r')
        # If batched, count first dimension; else count as 1
        n = arr.shape[0] if arr.ndim > 2 else 1
        total_samples += n
    return total_samples

def print_samples_per_file(base_path, family):
    """
    For each .npy file in the family directory, print the filename, shape, and number of samples (first dimension if batched, else 1).
    """
    fam_dir = base_path / family
    files = list(fam_dir.glob('seis*.npy')) + list(fam_dir.glob('vel*.npy'))
    print(f"\nSamples per file for family: {family}")
    for f in files:
        arr = np.load(f, mmap_mode='r')
        shape = arr.shape
        if len(shape) > 1:
            n_samples = shape[0]
        else:
            n_samples = 1
        print(f"  {f.name}: shape = {shape}, samples in file = {n_samples}")

def main():
    # from config import CFG
    families = list(CFG.paths.families.keys())
    family_stats = {}
    
    for family in families:
        print(f"\n--- EDA for family: {family} ---")
        seis_stats, vel_stats = eda_on_family(family)
        family_stats[family] = (seis_stats, vel_stats)
        print_summary(seis_stats, f"{family} seis")
        print_summary(vel_stats, f"{family} vel")
    
    # Generate distribution plots
    fig = plot_family_distributions(family_stats)
    plt.show()
    
    # Analyze correlations
    correlations = analyze_family_correlations(family_stats)
    print("\nCorrelations between seismic and velocity statistics:")
    for family, corrs in correlations.items():
        print(f"{family}:")
        print(f"  Mean correlation: {corrs['mean_corr']:.3f}")
        print(f"  Std correlation: {corrs['std_corr']:.3f}")

    # New EDA: Geometry, shape, and size summaries
    extract_and_plot_geometry(family_stats)
    summarize_array_shapes(family_stats)
    summarize_family_sizes(family_stats)

    # --- Print explicit sample counts for each family ---
    print("\nSample counts per family (Base and OpenFWI):")
    base_path = CFG.paths.train  # This should point to 'train_samples'
    openfwi_path = Path('/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72')
    print(f"{'Family':<15} {'Base':>6} {'OpenFWI':>8}")
    print("-"*32)
    for fam in families:
        base_count = count_samples_base(base_path, fam)
        openfwi_count = count_samples_openfwi(openfwi_path, fam)
        print(f"{fam:<15} {base_count:>6} {openfwi_count:>8}")
    print("-"*32)

    # Check samples per file for large Fault/Curve families
    for fam in ['FlatFault_A', 'FlatFault_B', 'CurveFault_A', 'CurveFault_B']:
        print_samples_per_file(base_path, fam)

    # Check balancing requirements
    check_balancing_requirements()

    # Check OpenFWI file shapes
    families = ['FlatVel_A', 'FlatFault_A', 'CurveVel_A']
    openfwi_root = Path('/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72')

    for fam in families:
        fam_dir = openfwi_root / fam
        # This needs to be updated to check for both seis*.npy and data*.npy files 

        files = sorted(fam_dir.glob('seis*.npy'))[:3] + sorted(fam_dir.glob('data*.npy'))[:3]
        print(f"\nFamily: {fam}")
        for f in files:
            arr = np.load(f, mmap_mode='r')
            print(f"  {f.name}: shape = {arr.shape}")

if __name__ == "__main__":
    main() 


# %%
# Source: model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from copy import deepcopy
# from src.core.config import CFG  # Absolute import
import logging
from pathlib import Path

def get_model(backbone: str = "hgnetv2_b2.ssld_stage2_ft_in1k", pretrained: bool = False, ema_decay: float = 0.99):
    """Create and initialize the model."""
    model = SpecProjNet(backbone=backbone, pretrained=pretrained, ema_decay=ema_decay)
    
    # Move model to device
    model = model.to(CFG.env.device)
    
    # Convert all parameters to float32
    model = model.float()
    
    # Load weights if available
    if CFG.weight_path and Path(CFG.weight_path).exists():
        try:
            state_dict = torch.load(CFG.weight_path, map_location=CFG.env.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"Successfully loaded weights from {CFG.weight_path}")
        except Exception as e:
            logging.error(f"Failed to load weights: {e}")
            logging.info("Continuing with randomly initialized weights")
    
    return model

class ModelEMA(nn.Module):
    """Exponential Moving Average wrapper for model weights."""
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class ConvBnAct2d(nn.Module):
    """Convolution block with optional batch norm and activation."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=norm_layer == nn.Identity,
        )
        self.bn = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SCSEModule2d(nn.Module):
    """Squeeze-and-Excitation module with channel and spatial attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock2d(nn.Module):
    """Decoder block with skip connection and optional attention."""
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()
        # Print channel dimensions for debugging
        print(f"DecoderBlock - in_channels: {in_channels}, skip_channels: {skip_channels}, out_channels: {out_channels}")
        
        # Upsampling block
        if upsample_mode == "deconv":
            self.upsample = nn.ConvTranspose2d(
                in_channels, 
                out_channels,  # Directly output desired number of channels
                kernel_size=scale_factor, 
                stride=scale_factor
            )
            self.channel_reduction = nn.Identity()  # No need for additional channel reduction
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.channel_reduction = nn.Identity()
        
        # Skip connection processing
        if intermediate_conv:
            self.skip_conv = ConvBnAct2d(skip_channels, skip_channels, 3, padding=1)
        else:
            self.skip_conv = nn.Identity()
            
        # Attention
        if attention_type == "scse":
            self.attention = SCSEModule2d(skip_channels)
        else:
            self.attention = nn.Identity()
            
        # Final convolution
        self.conv = ConvBnAct2d(
            out_channels + skip_channels,  # Concatenated channels
            out_channels,  # Output channels
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )

    def forward(self, x, skip=None):
        # Upsample and reduce channels
        x = self.upsample(x)
        
        # Process skip connection if available
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            skip = self.skip_conv(skip)
            skip = self.attention(skip)
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)

class UnetDecoder2d(nn.Module):
    """UNet decoder with configurable channels and upsampling."""
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple = (256, 128, 64, 32),
        scale_factors: tuple = (1,2,2,2),
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = True,
        upsample_mode: str = "deconv",
    ):
        super().__init__()
        # Reverse encoder channels to match decoder order
        encoder_channels = list(reversed(encoder_channels))
        if skip_channels is None:
            skip_channels = encoder_channels[1:]  # Skip the first channel as it's the input
            
        # Print channel dimensions for debugging
        print(f"Encoder channels (reversed): {encoder_channels}")
        print(f"Skip channels: {skip_channels}")
        print(f"Decoder channels: {decoder_channels}")
        
        # Initial channel reduction from encoder to decoder
        self.initial_reduction = nn.Conv2d(encoder_channels[0], decoder_channels[0], 1)
        
        # Create channel reduction layers for each encoder feature
        self.channel_reductions = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1)
            for in_ch, out_ch in zip(encoder_channels[1:], decoder_channels[1:])
        ])
        
        # Create skip connection channel reduction layers
        self.skip_reductions = nn.ModuleList([
            nn.Conv2d(skip_ch, out_ch, 1)
            for skip_ch, out_ch in zip(skip_channels, decoder_channels[1:])  # Skip first decoder channel
        ])
            
        self.blocks = nn.ModuleList([
            DecoderBlock2d(
                in_channels=decoder_channels[i],  # Current block's input channels
                skip_channels=decoder_channels[i+1] if i < len(decoder_channels)-1 else decoder_channels[-1],  # Next block's channels
                out_channels=decoder_channels[i+1] if i < len(decoder_channels)-1 else decoder_channels[-1],  # Next block's channels
                norm_layer=norm_layer,
                attention_type=attention_type,
                intermediate_conv=intermediate_conv,
                upsample_mode=upsample_mode,
                scale_factor=scale_factor,
            )
            for i, scale_factor in enumerate(scale_factors)
        ])

    def forward(self, feats: list[torch.Tensor]):
        # Reverse features to match decoder order
        feats = list(reversed(feats))
        
        # Initial channel reduction for the deepest feature
        x = self.initial_reduction(feats[0])
        
        # Reduce channels of remaining encoder features
        reduced_feats = [reduction(feat) for reduction, feat in zip(self.channel_reductions, feats[1:])]
        
        # Reduce channels of skip connections
        reduced_skips = [reduction(feat) for reduction, feat in zip(self.skip_reductions, feats[1:])]
        
        # Process through decoder blocks
        for block, skip in zip(self.blocks, reduced_skips):
            x = block(x, skip)
            
        return x

class CustomUpSample(nn.Module):
    """Memory-efficient upsampling module that combines bilinear upsampling with convolution."""
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # Use 1x1 conv for channel reduction to save memory
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # Use 3x3 conv for spatial features
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Enable gradient checkpointing if configured
        if CFG.gradient_checkpointing:
            self.conv = torch.utils.checkpoint.checkpoint_sequential(self.conv, 1)
            self.spatial_conv = torch.utils.checkpoint.checkpoint_sequential(self.spatial_conv, 1)
        
    def forward(self, x):
        # Use mixed precision if configured
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            # Upsample first to reduce memory usage during convolution
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
            # Apply convolutions
            x = self.conv(x)
            x = self.spatial_conv(x)
        return x

class CustomSubpixelUpsample(nn.Module):
    """Memory-efficient subpixel upsampling using pixel shuffle."""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        # Use 1x1 conv for initial channel reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2) // 2, 1)
        # Use 3x3 conv for spatial features
        self.conv2 = nn.Conv2d(out_channels * (scale_factor ** 2) // 2, 
                              out_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Enable gradient checkpointing if configured
        if CFG.gradient_checkpointing:
            self.conv1 = torch.utils.checkpoint.checkpoint_sequential(self.conv1, 1)
            self.conv2 = torch.utils.checkpoint.checkpoint_sequential(self.conv2, 1)
        
    def forward(self, x):
        # Use mixed precision if configured
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            # Progressive channel reduction to save memory
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pixel_shuffle(x)
        return x

class SegmentationHead2d(nn.Module):
    """Final segmentation head with optional upsampling."""
    def __init__(
        self,
        in_channels,
        out_channels,
        target_size: tuple[int] = (70, 70),  # Target output size
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        # Use 1x1 conv for initial channel reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        # Use 3x3 conv for spatial features
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Enable gradient checkpointing if configured
        self.use_checkpoint = CFG.gradient_checkpointing
        
        if mode == "nontrainable":
            self.upsample = nn.Upsample(size=target_size, mode="bilinear", align_corners=False)
        else:
            # Using custom subpixel upsampling
            self.upsample = CustomSubpixelUpsample(out_channels, out_channels, scale_factor=2)

    def forward(self, x):
        # Use mixed precision if configured
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            # Progressive channel reduction to save memory
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self.conv1, x)
                x = torch.utils.checkpoint.checkpoint(self.conv2, x)
            else:
                x = self.conv1(x)
                x = self.conv2(x)
            x = self.upsample(x)
        return x

class SpecProjNet(nn.Module):
    """SpecProj model with HGNet backbone."""
    def __init__(self, backbone: str = "hgnetv2_b2.ssld_stage2_ft_in1k", pretrained: bool = False, ema_decay: float = 0.99):
        super().__init__()
        
        # Input projection layer to convert to 3 channels
        self.input_proj = nn.Conv2d(5, 3, kernel_size=1)
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )
            
        # Get channel dimensions from backbone
        self.encoder_channels = self.backbone.feature_info.channels()
        print("Backbone output channels:", self.encoder_channels)
        
        # Initialize decoder blocks with correct channel dimensions
        self.decoder_blocks = nn.ModuleList()
        in_channels = self.encoder_channels[-1]  # Start with last feature map channels
        
        for i in range(len(self.encoder_channels)-1):
            skip_channels = self.encoder_channels[-(i+2)]
            out_channels = self.encoder_channels[-(i+2)]
            print(f"DecoderBlock - in_channels: {in_channels}, skip_channels: {skip_channels}, out_channels: {out_channels}")
            self.decoder_blocks.append(
                DecoderBlock2d(in_channels, skip_channels, out_channels)
            )
            in_channels = out_channels
            
        # Final projection layer
        self.final_proj = nn.Conv2d(self.encoder_channels[0], 1, kernel_size=1)
        
        # Final resize layer to match target dimensions
        self.final_resize = nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False)
        
        # Initialize EMA
        self.ema_decay = ema_decay
        self.ema_model = None
        
    def forward(self, x):
        # Handle 5D input [batch, channels, sources, receivers, timesteps]
        if x.dim() == 5:
            B, C, S, R, T = x.shape
            # Reshape to combine sources and channels: [batch, sources*channels, receivers, timesteps]
            x = x.reshape(B, S*C, R, T)
            
        # Project input to 3 channels
        x = self.input_proj(x)
            
        # Get features from backbone
        features = self.backbone(x)
        
        # Decode features
        x = features[-1]  # Start with last feature map
        for i, decoder in enumerate(self.decoder_blocks):
            skip = features[-(i+2)] if i < len(features)-1 else None
            x = decoder(x, skip)
            
        # Final projection
        x = self.final_proj(x)
        
        # Resize to target dimensions
        x = self.final_resize(x)
        
        return x
        
    def update_ema(self):
        """Update EMA weights."""
        if self.ema_model is not None:
            self.ema_model.update(self)
            
    def set_ema(self):
        """Set EMA weights."""
        if self.ema_model is not None:
            self.ema_model.set(self)
            
    def get_ema_model(self):
        """Get EMA model."""
        return self.ema_model if self.ema_model is not None else self 


# %%
# Source: proj_mask.py
# ## Model Architecture - Spectral Projector



# %%
# Source: proj_mask.py
import torch, math
from torch import nn
from config import CFG
from functools import lru_cache

@lru_cache(maxsize=None)
def _freq_grids(shape: tuple, device: torch.device):
    # shape = (..., T, R) after rfftn  ⇒  last dim is R/2+1
    T, R2 = shape[-2], shape[-1]
    dt, dx = 1e-3, 10.           # hard-coded (could move to CFG)
    freqs_t = torch.fft.rfftfreq(T, dt, device=device)          # (T//2+1,)
    freqs_x = torch.fft.fftfreq(2*(R2-1), dx, device=device)    # (R,)
    ω = freqs_t.view(-1, 1)                  # (T/2+1,1)
    k = freqs_x.view(1, -1)                  # (1,R)
    return ω, k

class PhysMask(nn.Module):
    """
    Learns background slowness c and temperature τ.
    Returns up-going and down-going wavefields.
    """
    def __init__(self,
                 c_init=3000., c_min=1500., c_max=5500.):
        super().__init__()
        self.log_c   = nn.Parameter(torch.log(torch.tensor(c_init)))
        self.log_tau = nn.Parameter(torch.zeros(()))  # τ ≈ 1

        self.register_buffer('c_min', torch.tensor(c_min))
        self.register_buffer('c_max', torch.tensor(c_max))

    def forward(self, x: torch.Tensor):
        # x  (B,S,T,R)  – real
        B, S, T, R = x.shape
        Xf = torch.fft.rfftn(x, dim=(-2,-1))
        ω, k = _freq_grids(Xf.shape[-2:], Xf.device)

        c  = torch.sigmoid(self.log_c) * (self.c_max-self.c_min) + self.c_min
        τ  = torch.exp(self.log_tau).clamp(0.1, 10.)

        ratio = ω / torch.sqrt(ω**2 + (c*k)**2 + 1e-9)
        mask_up   = torch.sigmoid( ratio / τ)
        mask_down = torch.sigmoid(-ratio / τ)

        mask_up = mask_up.expand_as(Xf)
        mask_down = mask_down.expand_as(Xf)

        up   = torch.fft.irfftn(Xf*mask_up  , dim=(-2,-1), s=(T,R))
        down = torch.fft.irfftn(Xf*mask_down, dim=(-2,-1), s=(T,R))
        return up, down

class SpectralAssembler(nn.Module):
    """
    Implements Π±† using small ε-regularized Moore-Penrose inverse.
    Reconstructs wavefield from up-going and down-going components.
    
    Reference: §1 of ICLR25FWI paper for mathematical formulation.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        """
        Args:
            up: Up-going wavefield (B,S,T,R)
            down: Down-going wavefield (B,S,T,R)
        Returns:
            Reconstructed wavefield (B,S,T,R)
        """
        # Convert to frequency domain
        Up_f = torch.fft.rfftn(up, dim=(-2,-1))
        Down_f = torch.fft.rfftn(down, dim=(-2,-1))
        
        # Compute Moore-Penrose inverse with regularization
        # Π† = (Π^T Π + εI)^(-1) Π^T
        # where Π is the projection operator
        Up_f_conj = Up_f.conj()
        Down_f_conj = Down_f.conj()
        
        # Regularized inverse
        denom = (Up_f_conj * Up_f + Down_f_conj * Down_f + self.eps)
        inv = 1.0 / denom
        
        # Apply inverse to reconstruct
        recon_f = inv * (Up_f_conj * Up_f + Down_f_conj * Down_f)
        
        # Convert back to time domain
        return torch.fft.irfftn(recon_f, dim=(-2,-1), s=up.shape[-2:])

def split_and_reassemble(x: torch.Tensor, mask: PhysMask, assembler: SpectralAssembler) -> torch.Tensor:
    """
    Helper function to split wavefield and reassemble it.
    Useful for unit testing the projection operators.
    
    Args:
        x: Input wavefield (B,S,T,R)
        mask: PhysMask instance
        assembler: SpectralAssembler instance
    Returns:
        Reconstructed wavefield (B,S,T,R)
    """
    up, down = mask(x)
    return assembler(up, down) 


# %%
# Source: iunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG

class CouplingLayer(nn.Module):
    """Coupling layer for invertible transformations."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels//2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels//2, 3, padding=1)
        )
        
    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            x2 = x2 + self.net(x1)
        else:
            x2 = x2 - self.net(x1)
        return torch.cat([x1, x2], dim=1)

class IUNet(nn.Module):
    """
    Invertible U-Net for latent space translation.
    Approximately 24M parameters.
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Initial projection
        self.proj_in = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # Coupling layers
        self.couplings = nn.ModuleList([
            CouplingLayer(hidden_channels, hidden_channels*2)
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.proj_out = nn.Conv2d(hidden_channels, in_channels, 1)
        
    def forward(self, z: torch.Tensor, direction: str) -> torch.Tensor:
        """
        Args:
            z: Input tensor (B,C,H,W)
            direction: Either "p→v" or "v→p"
        Returns:
            Translated tensor (B,C,H,W)
        """
        if direction not in ["p→v", "v→p"]:
            raise ValueError("direction must be either 'p→v' or 'v→p'")
            
        x = self.proj_in(z)
        
        # Apply coupling layers
        for coupling in self.couplings:
            x = coupling(x, reverse=(direction == "v→p"))
            
        return self.proj_out(x)

def create_iunet() -> IUNet:
    """Factory function to create IU-Net with default config."""
    return IUNet(
        in_channels=1,
        hidden_channels=64,
        num_layers=4
    ) 


# %%
# Source: specproj_hybrid.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.config import CFG
from src.core.proj_mask import PhysMask, SpectralAssembler
from src.core.iunet import create_iunet
from .specproj_unet import SmallUNet
import logging

class WaveDecoder(nn.Module):
    """Simple decoder for wavefield prediction."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        return self.net(x)

class HybridSpecProj(nn.Module):
    """
    Hybrid model combining physics-guided splitting with IU-Net translation.
    Memory-efficient implementation for Kaggle.
    """
    def __init__(self, chans=32, depth=4):
        super().__init__()
        self.mask = PhysMask()
        self.assembler = SpectralAssembler()
        
        # IU-Net for latent translation (only if joint training)
        self.iunet = create_iunet() if CFG.is_joint() else None
        
        # Decoders
        self.vel_decoder = SpecProjNet(
            backbone=CFG.backbone,
            pretrained=False,
            ema_decay=CFG.ema_decay
        )
        self.wave_decoder = WaveDecoder() if CFG.is_joint() else None
        
        # Enable gradient checkpointing if configured
        if CFG.gradient_checkpointing:
            self.vel_decoder.backbone.use_checkpoint = True
            if self.iunet is not None:
                self.iunet.use_checkpoint = True
        
    def forward(self, seis: torch.Tensor, mode: str = "inverse") -> tuple:
        """
        Args:
            seis: Seismic data (B,S,T,R)
            mode: Either "inverse" or "joint"
        Returns:
            If mode == "inverse":
                (v_pred, None)
            If mode == "joint":
                (v_pred, p_pred)
        """
        # Split wavefield
        up, down = self.mask(seis)
        
        if not CFG.is_joint() or mode == "inverse":
            # Use original SpecProjNet path
            B,S,T,R = up.shape
            up = up.reshape(B*S, 1, T, R).repeat(1,5,1,1)
            down = down.reshape(B*S, 1, T, R).repeat(1,5,1,1)
            v_pred = self.vel_decoder(up)
            return v_pred, None
            
        # Joint mode with IU-Net translation
        # 1. Translate up/down to velocity space
        v_up = self.iunet(up, "p→v")
        v_down = self.iunet(down, "p→v")
        
        # 2. Decode velocity
        v_pred = self.vel_decoder(torch.cat([v_up, v_down], dim=1))
        
        # 3. Translate back to wavefield space
        p_up = self.iunet(v_up, "v→p")
        p_down = self.iunet(v_down, "v→p")
        
        # 4. Reassemble and decode wavefield
        p_recon = self.assembler(p_up, p_down)
        p_pred = self.wave_decoder(p_recon)
        
        return v_pred, p_pred

# For backwards compatibility
SpecProjUNet = HybridSpecProj 


# %%
# Source: losses.py
# ## Loss Functions



# %%
# Source: losses.py
import torch
import torch.nn.functional as F
from src.core.config import CFG
import torch.cuda.amp as amp
from typing import Dict, Tuple, Optional

def track_memory_usage():
    """Track current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0.0

def pde_residual(v_pred: torch.Tensor,
                 seis: torch.Tensor,
                 dt: float = 1e-3,
                 dx: float = 10.,
                 use_amp: bool = True) -> torch.Tensor:
    """
    Enhanced acoustic wave equation residual with mixed precision support.
    
    Args:
        v_pred: Predicted velocity field
        seis: Seismic data
        dt: Time step
        dx: Spatial step
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        PDE residual loss
    """
    with amp.autocast(enabled=use_amp):
        p = seis[:,0]  # (B,T,R)
        
        # Second-order time derivative
        d2t = (p[:,2:] - 2*p[:,1:-1] + p[:,:-2]) / (dt*dt)
        
        # Laplacian in both dimensions
        lap_x = (p[:,:,2:] - 2*p[:,:,1:-1] + p[:,:,:-2]) / (dx*dx)
        lap_y = (p[:,2:,:] - 2*p[:,1:-1,:] + p[:,:-2,:]) / (dx*dx)
        lap = lap_x + lap_y
        
        # Velocity squared
        v2 = v_pred[...,1:-1,1:-1]**2
        
        # Full residual
        res = d2t[...,1:-1] - v2*lap[:,1:-1]
        
        return res.abs()

class JointLoss(torch.nn.Module):
    """
    Enhanced joint loss with memory tracking and mixed precision support.
    """
    def __init__(self,
                 λ_inv: float = 1.0,
                 λ_fwd: float = 1.0,
                 λ_pde: float = 0.1,
                 λ_reg: float = 0.01,
                 use_amp: bool = True):
        super().__init__()
        self.λ_inv = λ_inv
        self.λ_fwd = λ_fwd
        self.λ_pde = λ_pde
        self.λ_reg = λ_reg
        self.use_amp = use_amp
        
    def forward(self,
                v_pred: torch.Tensor,
                v_true: torch.Tensor,
                p_pred: Optional[torch.Tensor] = None,
                p_true: Optional[torch.Tensor] = None,
                seis_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        with amp.autocast(enabled=self.use_amp):
            # Inverse loss (velocity)
            l_inv = F.l1_loss(v_pred, v_true)
            
            # Forward loss (wavefield)
            l_fwd = torch.tensor(0.0, device=v_pred.device)
            if p_pred is not None and p_true is not None:
                l_fwd = F.l1_loss(p_pred, p_true)
                
            # PDE residual
            l_pde = torch.tensor(0.0, device=v_pred.device)
            if seis_batch is not None:
                l_pde = pde_residual(v_pred, seis_batch, use_amp=self.use_amp).mean()
                
            # Regularization loss (optional)
            l_reg = torch.tensor(0.0, device=v_pred.device)
            if hasattr(self, 'λ_reg') and self.λ_reg > 0:
                l_reg = F.mse_loss(v_pred[:,:,1:] - v_pred[:,:,:-1], 
                                 torch.zeros_like(v_pred[:,:,1:]))
            
            # Combine losses
            total = (self.λ_inv * l_inv + 
                    self.λ_fwd * l_fwd + 
                    self.λ_pde * l_pde +
                    self.λ_reg * l_reg)
            
            # Track memory usage
            memory_used = track_memory_usage()
                    
            return total, {
                'l_inv': l_inv.item(),
                'l_fwd': l_fwd.item() if isinstance(l_fwd, torch.Tensor) else l_fwd,
                'l_pde': l_pde.item() if isinstance(l_pde, torch.Tensor) else l_pde,
                'l_reg': l_reg.item() if isinstance(l_reg, torch.Tensor) else l_reg,
                'memory_mb': memory_used
            }

def get_loss_fn() -> torch.nn.Module:
    """Get the appropriate loss function based on configuration."""
    if CFG.is_joint():
        return JointLoss(
            λ_inv=CFG.lambda_inv,
            λ_fwd=CFG.lambda_fwd,
            λ_pde=CFG.lambda_pde,
            λ_reg=CFG.lambda_reg if hasattr(CFG, 'lambda_reg') else 0.01,
            use_amp=CFG.use_amp if hasattr(CFG, 'use_amp') else True
        )
    else:
        return torch.nn.L1Loss()  # Default to L1 loss for non-joint training

# For backwards compatibility
HybridLoss = JointLoss 


# %%
# Source: train.py
"""
Training script that uses DataManager for all data IO.
"""
import torch
import random
import numpy as np
import gc  # Add this import for garbage collection
from pathlib import Path
from tqdm import tqdm
import torch.cuda.amp as amp
import logging
import sys
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import json
import boto3
from botocore.exceptions import ClientError
import signal
import time
import torch.nn as nn
import shutil

# Local imports - ensure these are at the top level
from src.core.config import CFG
from src.core.data_manager import DataManager
from src.core.model import get_model
from src.core.losses import get_loss_fn
from src.core.setup import push_to_kaggle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def set_seed(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

class SpotInstanceHandler:
    """Handles spot instance interruptions gracefully and manages S3 uploads/cleanup."""
    def __init__(self, checkpoint_dir: Path, s3_upload_interval: int):
        self.checkpoint_dir = checkpoint_dir
        self.s3_upload_interval = s3_upload_interval
        self.last_checkpoint = 0
        self.interrupted = False
        signal.signal(signal.SIGTERM, self.handle_interruption)
        signal.signal(signal.SIGINT, self.handle_interruption)

    def handle_interruption(self, signum, frame):
        logging.info(f"Received signal {signum}, preparing for interruption...")
        self.interrupted = True

    def should_upload_s3(self, epoch: int, is_last_epoch: bool) -> bool:
        return self.interrupted or (epoch % self.s3_upload_interval == 0) or is_last_epoch

    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, metrics, upload_s3=False):
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }
            ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, ckpt_path)
            local_checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if len(local_checkpoints) > 3:
                for old_ckpt in local_checkpoints[:-3]:
                    old_ckpt.unlink()
                    logging.info(f"Removed old local checkpoint: {old_ckpt}")
            meta_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_metadata.json'
            with open(meta_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'optimizer_state_dict': convert_to_serializable(optimizer.state_dict()),
                    'scheduler_state_dict': convert_to_serializable(scheduler.state_dict()),
                    'scaler_state_dict': convert_to_serializable(scaler.state_dict()),
                    'metrics': metrics,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                }, f, indent=2)
            if upload_s3 and CFG.env.kind == 'aws':
                s3 = boto3.client('s3', region_name=CFG.env.aws_region)
                s3_key = f"checkpoints/checkpoint_epoch_{epoch}.pt"
                s3.upload_file(str(ckpt_path), CFG.env.s3_bucket, s3_key)
                response = s3.list_objects_v2(Bucket=CFG.env.s3_bucket, Prefix='checkpoints/checkpoint_epoch_')
                if 'Contents' in response:
                    checkpoints = sorted(response['Contents'], key=lambda x: x['LastModified'])
                    if len(checkpoints) > 5:
                        for old_ckpt in checkpoints[:-5]:
                            s3.delete_object(Bucket=CFG.env.s3_bucket, Key=old_ckpt['Key'])
                            logging.info(f"Removed old S3 checkpoint: {old_ckpt['Key']}")
            self.last_checkpoint = time.time()
            logging.info(f"Checkpoint saved for epoch {epoch}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise

def save_checkpoint(model, optimizer, scheduler, epoch, loss, data_manager):
    logging.info(f"Starting save_checkpoint for epoch {epoch} with loss {loss}")
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    try:
        torch.save(checkpoint, out_dir / 'checkpoint.pth')
        logging.info(f"Checkpoint saved at outputs/checkpoint.pth for epoch {epoch}")
        best_path = out_dir / 'best.pth'
        is_best = False
        if not best_path.exists():
            is_best = True
        else:
            try:
                prev_checkpoint = torch.load(best_path)
                prev_loss = prev_checkpoint.get('loss', float('inf'))
                is_best = loss < prev_loss
            except Exception as e:
                logging.warning(f"Could not load previous checkpoint: {e}")
                is_best = True
        if is_best:
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved at outputs/best.pth with loss: {loss:.4f}")
            metadata = {
                'epoch': epoch,
                'loss': float(loss),
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'batch_size': CFG.batch,
                    'learning_rate': CFG.lr,
                    'weight_decay': CFG.weight_decay,
                    'backbone': CFG.backbone,
                }
            }
            with open(out_dir / 'best_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Best model metadata saved at outputs/best_metadata.json")
            print("Uploading best model and metadata to S3")
            data_manager.upload_best_model_and_metadata(
                out_dir,
                f"Update best model - loss: {loss:.4f} at epoch {epoch}"
            )
        logging.info(f"save_checkpoint completed for epoch {epoch}")
    except Exception as e:
        logging.error(f"Exception in save_checkpoint: {e}")
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

def train(dryrun: bool = False, fp16: bool = False):
    """Main training loop."""
    # Initialize data manager
    data_manager = DataManager()
    
    # Create loaders for each family
    train_loaders = []
    for family in CFG.paths.families:
        if CFG.debug_mode and family in CFG.families_to_exclude:
            continue
            
        seis_files, vel_files, family_type = data_manager.list_family_files(family)
        if seis_files is None:  # Skip excluded families
            continue
            
        print(f"Processing family: {family} ({family_type}), #seis: {len(seis_files)}, #vel: {len(vel_files)}")
        
        loader = data_manager.create_loader(
            seis_files, vel_files, family_type,
            batch_size=CFG.batch,
            shuffle=True,
            num_workers=CFG.num_workers
        )
        if loader is not None:
            train_loaders.append(loader)
            
    if not train_loaders:
        raise ValueError("No valid data loaders created!")
        
    # Initialize model and move to device
    model = get_model()
    model.train()
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.epochs,
        eta_min=CFG.lr/100
    )
    
    # Initialize loss function
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(CFG.epochs):
        total_loss = 0
        total_batches = 0
        
        for loader in train_loaders:
            for batch_idx, (x, y) in enumerate(loader):
                try:
                    # Move data to device and convert to float32
                    x = x.to(CFG.env.device).float()
                    y = y.to(CFG.env.device).float()
                    
                    # Forward pass
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    total_loss += loss.item()
                    total_batches += 1
                    
                    # Print progress
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{CFG.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        print(f"WARNING: out of memory in batch {batch_idx}. Skipping batch.")
                        continue
                    else:
                        raise e
                        
        # Update learning rate
        scheduler.step()
        
        # Print epoch statistics
        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{CFG.epochs} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % CFG.debug_upload_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, data_manager)
            
    # After all epochs:
    # Save best.pth and best_metadata.json to /kaggle/working for persistence
    kaggle_working = Path('/kaggle/working')
    kaggle_working.mkdir(parents=True, exist_ok=True)
    out_dir = Path('outputs')
    for fname in ['best.pth', 'best_metadata.json']:
        src = out_dir / fname
        dst = kaggle_working / fname
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"File {src} does not exist, skipping copy.")
    print("Uploading model to Kaggle dataset (final upload)...")
    push_to_kaggle(out_dir, "Final upload after training")
    print("Upload complete.")

    return model

if __name__ == '__main__':
    # Set debug mode to True and update settings
    CFG.set_debug_mode(True)
    train() 


# %%
# Source: infer.py
"""
Inference script that uses DataManager for all data IO.
All data IO in this file must go through DataManager (src/core/data_manager.py)
Do NOT load data directly in this file.
"""
import torch
from tqdm import tqdm
from torch.amp import autocast
from model import SpecProjNet
from data_manager import DataManager
from config import CFG
import pandas as pd
import os
from pathlib import Path
import csv

def format_submission(vel_map, oid):
    # vel_map: (70, 70) numpy array
    # oid: string (file stem)
    rows = []
    for y in range(vel_map.shape[0]):
        row = {'oid_ypos': f'{oid}_y_{y}'}
        for i, x in enumerate(range(1, 70, 2)):
            row[f'x_{x}'] = float(vel_map[y, x])
        rows.append(row)
    return rows

def infer():
    # Use weights path from config
    weights = CFG.weight_path

    # Model setup
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=False,
        ema_decay=0.99,
    ).to(CFG.env.device)
    print("Model created")
    checkpoint = torch.load(weights, map_location=CFG.env.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_ema()
    print("Weights loaded and EMA set")
    model.eval()
    print("Model set to eval")

    # Data setup
    data_manager = DataManager()
    test_files = data_manager.get_test_files()
    print("Number of test files:", len(test_files))
    # test_files = test_files[:10]  # Uncomment for debugging only
    test_dataset = data_manager.create_dataset(test_files, None, 'test')
    test_loader = data_manager.create_loader(
        test_files, None, 'test',
        batch_size=2,  # or your preferred batch size
        shuffle=False,
        num_workers=0
    )
    print("Test loader created")

    # Prepare CSV header
    x_cols = [f'x_{x}' for x in range(1, 70, 2)]
    fieldnames = ['oid_ypos'] + x_cols

    print("Starting inference loop")
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for batch_idx, (seis, _) in enumerate(tqdm(test_loader, desc="Processing test files")):
                seis = seis.to(CFG.env.device, non_blocking=True).float()
                with autocast('cuda', enabled=CFG.use_amp):
                    preds = model(seis)
                preds = preds.cpu().numpy()
                for i in range(preds.shape[0]):
                    vel_map = preds[i]
                    if vel_map.shape[0] == 1:
                        vel_map = vel_map[0]
                    oid = Path(test_files[batch_idx * test_loader.batch_size + i]).stem
                    rows = format_submission(vel_map, oid)
                    writer.writerows(rows)  # Write this batch's rows immediately
    print("Inference complete and submission.csv saved.")

if __name__ == "__main__":
    infer() 

