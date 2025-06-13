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
        env_kind = os.environ.get("GWI_ENV", "").lower()
        if not env_kind:
            raise RuntimeError(
                "You must specify the environment kind via the GWI_ENV environment variable "
                "(e.g., 'aws', 'kaggle', 'colab', 'sagemaker', 'local')."
            )
        self.kind = env_kind
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
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
            cls._inst.batch   = 1
            cls._inst.lr      = 1e-4
            cls._inst.weight_decay = 1e-3
            cls._inst.epochs  = 120
            cls._inst.lambda_pde = 0.1
            cls._inst.dtype = "float16"  # Default dtype for tensors
            cls._inst.num_workers = 0
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

            # Set environment-specific paths and weight_path
            env_kind = cls._inst.env.kind
            if env_kind == 'aws':
                cls._inst.paths.aws_root = cls._inst.env.ebs_mount / 'waveform-inversion'
                cls._inst.paths.aws_train = cls._inst.paths.aws_root / 'train_samples'
                cls._inst.paths.aws_test = cls._inst.paths.aws_root / 'test'
                cls._inst.paths.aws_output = cls._inst.env.ebs_mount / 'output'
                cls._inst.paths.root = cls._inst.paths.aws_root
                cls._inst.paths.train = cls._inst.paths.aws_train
                cls._inst.paths.test = cls._inst.paths.aws_test
                cls._inst.weight_path = "/mnt/waveform-inversion/best.pth"
            elif env_kind == 'kaggle':
                cls._inst.paths.root = Path('/kaggle/input/waveform-inversion')
                cls._inst.paths.train = cls._inst.paths.root / 'train_samples'
                cls._inst.paths.test = cls._inst.paths.root / 'test'
                cls._inst.weight_path = "/kaggle/input/yalegwi/best.pth"
            elif env_kind == 'colab':
                cls._inst.paths.root = Path('/content/data/waveform-inversion')
                cls._inst.paths.train = cls._inst.paths.root / 'train_samples'
                cls._inst.paths.test = cls._inst.paths.root / 'test'
                cls._inst.weight_path = "/content/data/waveform-inversion/best.pth"
            elif env_kind == 'sagemaker':
                cls._inst.paths.root = Path('/opt/ml/input/data/waveform-inversion')
                cls._inst.paths.train = cls._inst.paths.root / 'train_samples'
                cls._inst.paths.test = cls._inst.paths.root / 'test'
                cls._inst.weight_path = "/opt/ml/input/data/waveform-inversion/best.pth"
            else:  # local
                cls._inst.paths.root = Path(__file__).parent.parent.parent / 'data/waveform-inversion'
                cls._inst.paths.train = cls._inst.paths.root / 'train_samples'
                cls._inst.paths.test = cls._inst.paths.root / 'test'
                cls._inst.weight_path = str(cls._inst.paths.root / 'best.pth')

            # Set family paths for all environments
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

            # Loss weights
            cls._inst.lambda_inv = 1.0
            cls._inst.lambda_fwd = 1.0
            cls._inst.lambda_pde = 0.1

            # Enable joint training by default in Kaggle
            cls._inst.enable_joint = env_kind == 'kaggle'

            cls._inst.dataset_style = 'yalegwi'
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
        sfile, vfile, i, s = self.index[idx]
        # Load only the required source
        seis = np.load(sfile, mmap_mode='r')[i, s]  # shape: (receivers, timesteps)
        vel = np.load(vfile, mmap_mode='r')[i]      # shape: (1, 70, 70)
        # Convert to float16 and add batch/source dims
        seis = seis.astype(np.float16)[None, ...]    # shape: (1, receivers, timesteps)
        vel = vel.astype(np.float16)
        # Normalize
        mu = seis.mean(axis=(1,2), keepdims=True)
        std = seis.std(axis=(1,2), keepdims=True) + 1e-6
        seis = (seis - mu) / std
        if self.memory_tracker:
            self.memory_tracker.update(seis.nbytes + vel.nbytes)
        return torch.from_numpy(seis), torch.from_numpy(vel) 