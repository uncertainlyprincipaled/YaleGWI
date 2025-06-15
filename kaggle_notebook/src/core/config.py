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