# %% [markdown]
# ## Configuration and Environment Setup

# %%
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Literal, NamedTuple, Optional
import torch
import boto3
from botocore.exceptions import ClientError
import logging

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
        # Environment detection
        if os.environ.get("AWS_EXECUTION_ENV") or Path("/home/ec2-user").exists():
            self.kind: Literal['kaggle','colab','sagemaker','aws','local'] = 'aws'
        elif 'KAGGLE_URL_BASE' in os.environ:
            self.kind = 'kaggle'
        elif 'COLAB_GPU' in os.environ:
            self.kind = 'colab'
        elif 'SM_NUM_CPUS' in os.environ:
            self.kind = 'sagemaker'
        else:
            self.kind = 'local'
            
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # AWS-specific settings
        if self.kind == 'aws':
            self.set_aws_attributes()

    def set_aws_attributes(self):
        self.aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.environ.get('S3_BUCKET', 'yale-gwi-data')
        self.ebs_mount = Path('/mnt')
        self.use_spot = os.environ.get('USE_SPOT', 'true').lower() == 'true'
        self.spot_interruption_probability = float(os.environ.get('SPOT_INTERRUPTION_PROB', '0.1'))
        # Load AWS credentials from environment file
        aws_creds_path = Path(__file__).parent.parent.parent / '.env/aws/credentials'
        if aws_creds_path.exists():
            with open(aws_creds_path) as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value

class Config:
    """Read-only singleton accessed via `CFG`."""
    _inst = None
    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
            cls._inst.env   = _Env()
            cls._inst.paths = _KagglePaths()
            cls._inst.seed  = 42

            # Training hyper-parameters
            cls._inst.batch   = 16
            cls._inst.lr      = 1e-4
            cls._inst.weight_decay = 1e-3
            cls._inst.epochs  = 120
            cls._inst.lambda_pde = 0.1
            cls._inst.dtype = "float16"  # Default dtype for tensors
            cls._inst.num_workers = 2
            cls._inst.distributed = False  # Whether to use distributed training
            cls._inst.s3_upload_interval = 30  # S3 upload/checkpoint interval in epochs

            # Memory optimization settings
            cls._inst.memory_efficient = True  # Enable memory efficient operations
            cls._inst.use_amp = True  # Enable automatic mixed precision
            cls._inst.gradient_checkpointing = True  # Enable gradient checkpointing

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

            # AWS-specific settings
            if cls._inst.env.kind == 'aws':
                cls._inst.paths.aws_root = cls._inst.env.ebs_mount / 'waveform-inversion'
                cls._inst.paths.aws_train = cls._inst.paths.aws_root / 'train_samples'
                cls._inst.paths.aws_test = cls._inst.paths.aws_root / 'test'
                cls._inst.paths.aws_output = cls._inst.env.ebs_mount / 'output'
                
                # Update paths for AWS environment
                cls._inst.paths.root = cls._inst.paths.aws_root
                cls._inst.paths.train = cls._inst.paths.aws_train
                cls._inst.paths.test = cls._inst.paths.aws_test
                
                # Update family paths for AWS
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

            # Optionally, exclude families with too few samples
            cls._inst.families_to_exclude = []

        return cls._inst

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