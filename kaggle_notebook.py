"""
# Seismic Waveform Inversion - Preprocessing Pipeline

This notebook implements the preprocessing pipeline for seismic waveform inversion, including:
- Geometric-aware preprocessing with Nyquist validation
- Family-specific data loading
- Cross-validation framework
- Model registry and checkpoint management
"""

# Install dependencies
# !pip install -r requirements.txt


# %%
# Source: src/core/config.py
# ## Configuration and Environment Setup



# %%
# Source: src/core/config.py
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
        # Geological families
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
        
        # Preprocessed data paths (what the training code expects)
        self.preprocessed_paths = [
            Path('/kaggle/working/preprocessed'),      # Kaggle working directory
            Path('/content/drive/MyDrive/YaleGWI/preprocessed'),  # Google Drive
            Path('preprocessed'),                     # Local directory
            Path('/content/YaleGWI/preprocessed'),    # Colab local
        ]
        
        # Expected zarr file structure for geometric loading
        self.expected_zarr_structure = {
            'geometric_loader': {
                'description': 'Family-specific directories with seis/ zarr arrays',
                'structure': {
                    'gpu0': {
                        'FlatVel_A': 'seis/',
                        'FlatVel_B': 'seis/',
                        'CurveVel_A': 'seis/',
                        'CurveVel_B': 'seis/',
                        'Style_A': 'seis/',
                        'Style_B': 'seis/',
                        'FlatFault_A': 'seis/',
                        'FlatFault_B': 'seis/',
                        'CurveFault_A': 'seis/',
                        'CurveFault_B': 'seis/',
                    },
                    'gpu1': {
                        # Same structure for GPU1
                    }
                }
            },
            'current_preprocessing': {
                'description': 'GPU-specific directories with combined seismic.zarr files',
                'structure': {
                    'gpu0': 'seismic.zarr/ (contains seismic/ and velocity/ arrays)',
                    'gpu1': 'seismic.zarr/ (contains seismic/ and velocity/ arrays)',
                }
            }
        }

class _S3Paths:
    def __init__(self):
        self.bucket = os.environ.get('AWS_S3_BUCKET', 'yale-gwi')
        self.raw_prefix = 'raw/train_samples'
        self.preprocessed_prefix = 'preprocessed'
        # Per-family S3 paths
        self.families = {
            'FlatVel_A'   : f'{self.raw_prefix}/FlatVel_A',
            'FlatVel_B'   : f'{self.raw_prefix}/FlatVel_B',
            'CurveVel_A'  : f'{self.raw_prefix}/CurveVel_A',
            'CurveVel_B'  : f'{self.raw_prefix}/CurveVel_B',
            'Style_A'     : f'{self.raw_prefix}/Style_A',
            'Style_B'     : f'{self.raw_prefix}/Style_B',
            'FlatFault_A' : f'{self.raw_prefix}/FlatFault_A',
            'FlatFault_B' : f'{self.raw_prefix}/FlatFault_B',
            'CurveFault_A': f'{self.raw_prefix}/CurveFault_A',
            'CurveFault_B': f'{self.raw_prefix}/CurveFault_B',
        }
        # For preprocessed data
        self.preprocessed_families = {
            fam: f'{self.preprocessed_prefix}/{fam}' for fam in self.families
        }

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
            cls._inst.s3_paths = _S3Paths()
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
            cls._inst.ema_decay = 0.99 # Is this the correct value?
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
    
    def check_preprocessed_data_structure(self, data_root: Path) -> dict:
        """
        Check if the preprocessed data structure matches expected formats.
        
        Args:
            data_root: Path to preprocessed data directory
            
        Returns:
            dict: Status of data structure compatibility
        """
        if not data_root.exists():
            return {
                'exists': False,
                'geometric_compatible': False,
                'current_structure': 'not_found',
                'issues': ['Data directory does not exist']
            }
        
        # Check for current preprocessing structure (GPU-specific with seismic.zarr)
        gpu0_path = data_root / 'gpu0'
        gpu1_path = data_root / 'gpu1'
        
        current_structure = 'unknown'
        geometric_compatible = False
        issues = []
        
        if gpu0_path.exists() and gpu1_path.exists():
            # Check for current structure: gpu0/seismic.zarr/
            seismic_zarr_gpu0 = gpu0_path / 'seismic.zarr'
            seismic_zarr_gpu1 = gpu1_path / 'seismic.zarr'
            
            if seismic_zarr_gpu0.exists() and seismic_zarr_gpu1.exists():
                current_structure = 'gpu_specific_combined'
                
                # Check if it contains the expected arrays
                try:
                    import zarr
                    gpu0_data = zarr.open(str(seismic_zarr_gpu0))
                    if 'seismic' in gpu0_data and 'velocity' in gpu0_data:
                        geometric_compatible = False  # Current structure doesn't match geometric loader expectations
                        issues.append('Current structure uses combined seismic.zarr files, but geometric loader expects family-specific directories')
                    else:
                        issues.append('seismic.zarr exists but missing expected seismic/velocity arrays')
                except Exception as e:
                    issues.append(f'Error reading seismic.zarr: {e}')
            else:
                # Check for geometric loader structure: family-specific directories
                family_dirs = [d for d in gpu0_path.iterdir() if d.is_dir()]
                if family_dirs:
                    # Check if any family has the expected seis/ structure
                    for family_dir in family_dirs:
                        seis_path = family_dir / 'seis'
                        if seis_path.exists():
                            current_structure = 'family_specific_geometric'
                            geometric_compatible = True
                            break
                    else:
                        current_structure = 'family_specific_unknown'
                        issues.append('Family directories exist but missing expected seis/ zarr arrays')
                else:
                    issues.append('Neither gpu-specific nor family-specific structure found')
        else:
            issues.append('Missing gpu0/ and gpu1/ directories')
        
        return {
            'exists': True,
            'geometric_compatible': geometric_compatible,
            'current_structure': current_structure,
            'issues': issues,
            'data_root': str(data_root)
        }

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
                'families': {k: str(p) for k, p in v.families.items()},
                'preprocessed_paths': [str(p) for p in v.preprocessed_paths],
                'expected_zarr_structure': v.expected_zarr_structure
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
        # Get the file paths and indices for this sample
        sfile, vfile, i, _ = self.index[idx]
        
        # Load the seismic data for this sample using memory mapping
        # This loads all sources for the given sample index
        # Shape: (sources, receivers, timesteps)
        seis = np.load(sfile, mmap_mode='r')[i]
        
        # Load the corresponding velocity model
        # Shape: (1, 70, 70) - represents the subsurface velocity structure
        vel = np.load(vfile, mmap_mode='r')[i]
        
        # Convert both arrays to float16 to reduce memory usage
        # This is important for large datasets and GPU memory efficiency
        seis = seis.astype(np.float16)
        vel = vel.astype(np.float16)
        
        # Normalize the seismic data per source
        # This helps with training stability and convergence
        # 1. Calculate mean across receivers and timesteps for each source
        mu = seis.mean(axis=(1,2), keepdims=True)
        # 2. Calculate standard deviation with small epsilon to avoid division by zero
        std = seis.std(axis=(1,2), keepdims=True) + 1e-6
        # 3. Apply normalization
        seis = (seis - mu) / std
        
        # Update memory tracking if enabled
        if self.memory_tracker:
            self.memory_tracker.update(seis.nbytes + vel.nbytes)
            
        # Convert numpy arrays to PyTorch tensors and return
        # This is required for PyTorch's DataLoader
        return torch.from_numpy(seis), torch.from_numpy(vel)

# Temporary mapping for family data structure
FAMILY_FILE_MAP = {
    'CurveFault_A': {
        'seis_glob': 'seis*.npy', 'vel_glob': 'vel*.npy', 'seis_dir': '', 'vel_dir': '', 'downsample_factor': 1
    },
    'CurveFault_B': {
        'seis_glob': 'seis*.npy', 'vel_glob': 'vel*.npy', 'seis_dir': '', 'vel_dir': '', 'downsample_factor': 1
    },
    'CurveVel_A': {
        'seis_glob': 'data/*.npy', 'vel_glob': 'model/*.npy', 'seis_dir': 'data', 'vel_dir': 'model', 'downsample_factor': 4
    },
    'CurveVel_B': {
        'seis_glob': 'data/*.npy', 'vel_glob': 'model/*.npy', 'seis_dir': 'data', 'vel_dir': 'model', 'downsample_factor': 4
    },
    'FlatFault_A': {
        'seis_glob': 'seis*.npy', 'vel_glob': 'vel*.npy', 'seis_dir': '', 'vel_dir': '', 'downsample_factor': 1
    },
    'FlatFault_B': {
        'seis_glob': 'seis*.npy', 'vel_glob': 'vel*.npy', 'seis_dir': '', 'vel_dir': '', 'downsample_factor': 1
    },
    'FlatVel_A': {
        'seis_glob': 'data/*.npy', 'vel_glob': 'model/*.npy', 'seis_dir': 'data', 'vel_dir': 'model', 'downsample_factor': 4
    },
    'FlatVel_B': {
        'seis_glob': 'data/*.npy', 'vel_glob': 'model/*.npy', 'seis_dir': 'data', 'vel_dir': 'model', 'downsample_factor': 4
    },
    'Style_A': {
        'seis_glob': 'data/*.npy', 'vel_glob': 'model/*.npy', 'seis_dir': 'data', 'vel_dir': 'model', 'downsample_factor': 4
    },
    'Style_B': {
        'seis_glob': 'data/*.npy', 'vel_glob': 'model/*.npy', 'seis_dir': 'data', 'vel_dir': 'model', 'downsample_factor': 4
    },
} 




# %%
# Source: src/core/preprocess.py
"""
Seismic Data Preprocessing Pipeline

This module implements a comprehensive preprocessing pipeline for seismic data, focusing on:
1. Data downsampling while preserving signal integrity (Nyquist-Shannon theorem)
2. Memory-efficient processing using memory mapping and chunked operations
3. Distributed storage using Zarr and S3
4. GPU-optimized data splitting

Key Concepts:
- Nyquist-Shannon Theorem: Ensures we don't lose information during downsampling
- Memory Mapping: Allows processing large files without loading them entirely into memory
- Chunked Processing: Enables parallel processing and efficient memory usage
- Zarr Storage: Provides efficient compression and chunked storage for large datasets
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
# For Kaggle/Colab, install zarr
# !pip install zarr
import zarr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.signal import decimate
import logging
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings
import boto3
from botocore.exceptions import ClientError
import json
from src.core.config import CFG, FAMILY_FILE_MAP
import tempfile
from src.core.data_manager import DataManager
import pickle
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for preprocessing
CHUNK_TIME = 256  # After decimating by 4 - optimized for GPU memory
CHUNK_SRC_REC = 8  # Chunk size for source-receiver dimensions
NYQUIST_FREQ = 500  # Hz (half of original sampling rate) - critical for downsampling

class PreprocessingFeedback:
    """A simple class to collect feedback during preprocessing."""
    def __init__(self):
        self.nyquist_warnings = 0
        self.arrays_processed = 0

    def add_nyquist_warning(self):
        self.nyquist_warnings += 1

    def increment_arrays_processed(self):
        self.arrays_processed += 1

    @property
    def warning_percentage(self) -> float:
        if self.arrays_processed == 0:
            return 0.0
        return (self.nyquist_warnings / self.arrays_processed) * 100

class PreprocessingCache:
    """Cache for preprocessing results to avoid reprocessing test data."""
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path('/tmp/preprocessing_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, data_hash: str, dt_decimate: int, is_seismic: bool) -> str:
        """Generate cache key for preprocessing parameters."""
        key_data = f"{data_hash}_{dt_decimate}_{is_seismic}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, data_hash: str, dt_decimate: int, is_seismic: bool) -> Optional[np.ndarray]:
        """Get cached preprocessing result if available."""
        cache_key = self._get_cache_key(data_hash, dt_decimate, is_seismic)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def cache_result(self, data_hash: str, dt_decimate: int, is_seismic: bool, result: np.ndarray):
        """Cache preprocessing result."""
        cache_key = self._get_cache_key(data_hash, dt_decimate, is_seismic)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

def verify_data_structure(data_root: Path) -> bool:
    """
    Verify that the data structure is correct before preprocessing.
    
    Args:
        data_root: Root directory containing the training data
        
    Returns:
        bool: True if data structure is valid
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        logger.error(f"Data root directory does not exist: {data_root}")
        return False
    
    # Expected families
    expected_families = [
        'FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B',
        'Style_A', 'Style_B', 'FlatFault_A', 'FlatFault_B',
        'CurveFault_A', 'CurveFault_B'
    ]
    
    print("Verifying data structure...")
    print(f"Data root: {data_root}")
    print()
    
    all_valid = True
    
    for family in expected_families:
        family_dir = data_root / family
        if not family_dir.exists():
            print(f"✗ {family}: Directory not found")
            all_valid = False
            continue
        
        # Check for .npy files
        npy_files = list(family_dir.glob('*.npy'))
        if not npy_files:
            print(f"✗ {family}: No .npy files found")
            all_valid = False
            continue
        
        print(f"✓ {family}: {len(npy_files)} files found")
        
        # Check first file structure
        try:
            sample_file = npy_files[0]
            sample_data = np.load(sample_file, mmap_mode='r')
            
            if sample_data.ndim == 4:
                print(f"  - Shape: {sample_data.shape} (batch, sources, time, receivers)")
                print(f"  - Dtype: {sample_data.dtype}")
                
                # Check expected dimensions
                if sample_data.shape[2] != 2000:  # Time dimension
                    print(f"  ⚠ Warning: Expected time dimension 2000, got {sample_data.shape[2]}")
                if sample_data.shape[3] != 70:  # Receiver dimension
                    print(f"  ⚠ Warning: Expected receiver dimension 70, got {sample_data.shape[3]}")
                    
            elif sample_data.ndim == 3:
                print(f"  - Shape: {sample_data.shape} (sources, time, receivers)")
                print(f"  - Dtype: {sample_data.dtype}")
                
                # Check expected dimensions
                if sample_data.shape[1] != 2000:  # Time dimension
                    print(f"  ⚠ Warning: Expected time dimension 2000, got {sample_data.shape[1]}")
                if sample_data.shape[2] != 70:  # Receiver dimension
                    print(f"  ⚠ Warning: Expected receiver dimension 70, got {sample_data.shape[2]}")
            else:
                print(f"  ⚠ Warning: Unexpected number of dimensions: {sample_data.ndim}")
                
        except Exception as e:
            print(f"  ✗ Error loading sample file: {e}")
            all_valid = False
    
    print()
    if all_valid:
        print("✓ Data structure verification passed!")
        return True
    else:
        print("✗ Data structure verification failed!")
        return False

def validate_nyquist(data: np.ndarray, original_fs: int = 1000, dt_decimate: int = 4, feedback: Optional[PreprocessingFeedback] = None) -> bool:
    """
    Validate that the data satisfies Nyquist criterion after downsampling.
    
    Intuition:
    - Nyquist-Shannon Theorem: Sampling rate must be > 2x highest frequency
    - Energy Analysis: Check if significant energy exists above Nyquist frequency
    - Safety Margin: Warn if >1% of energy is above Nyquist (potential aliasing)
    
    Args:
        data: Input seismic data array
        original_fs: Original sampling frequency in Hz
        dt_decimate: The factor by which the data will be downsampled
        feedback: An optional feedback collector.
        
    Returns:
        bool: True if data satisfies Nyquist criterion
    """
    if data.ndim not in [3, 4]:
        logger.warning(f"Unexpected data dimension {data.ndim} in validate_nyquist. Skipping.")
        return True
    
    # Handle different data shapes more robustly
    if data.ndim == 4:
        # (batch, sources, time, receivers) or (batch, channels, time, receivers)
        if data.shape[1] == 5:  # sources
            time_axis = 2
        elif data.shape[1] == 1:  # channels
            time_axis = 2
        else:
            # Try to infer time axis - look for the longest dimension
            time_axis = np.argmax(data.shape[1:]) + 1
    else:  # 3D
        # (sources, time, receivers) or (time, receivers, sources)
        if data.shape[0] == 5:  # sources first
            time_axis = 1
        elif data.shape[2] == 5:  # sources last
            time_axis = 0
        else:
            # Try to infer time axis - look for the longest dimension
            time_axis = np.argmax(data.shape)

    # Ensure time_axis is valid
    if time_axis >= data.ndim:
        logger.warning(f"Invalid time_axis {time_axis} for data shape {data.shape}. Skipping validation.")
        return True

    try:
        # Compute FFT
        fft_data = np.fft.rfft(data, axis=time_axis)
        freqs = np.fft.rfftfreq(data.shape[time_axis], d=1/original_fs)
        
        # Check if significant energy exists above Nyquist frequency
        nyquist_mask = freqs > (original_fs / (2 * dt_decimate))
        
        # Handle case where nyquist_mask might be empty or have wrong shape
        if not np.any(nyquist_mask):
            return True
            
        # Ensure mask has correct shape for broadcasting
        mask_shape = [1] * data.ndim
        mask_shape[time_axis] = -1
        nyquist_mask = nyquist_mask.reshape(mask_shape)
        
        high_freq_energy = np.abs(fft_data * nyquist_mask).mean()
        total_energy = np.abs(fft_data).mean()
        
        # If more than 1% of energy is above Nyquist, warn
        if total_energy > 1e-9 and high_freq_energy / total_energy > 0.01:
            warnings.warn(f"Significant energy above Nyquist frequency detected: {high_freq_energy/total_energy:.2%}")
            if feedback:
                feedback.add_nyquist_warning()
            return False
        return True
        
    except Exception as e:
        logger.warning(f"Error in validate_nyquist: {e}. Skipping validation.")
        return True

def preprocess_one(arr: np.ndarray, dt_decimate: int = 4, is_seismic: bool = True, feedback: Optional[PreprocessingFeedback] = None) -> np.ndarray:
    """
    Preprocess a single seismic array with downsampling and normalization.
    
    Intuition:
    - Downsampling: Reduce data size while preserving signal integrity
    - Anti-aliasing: Prevent frequency folding during downsampling
    - Memory Efficiency: Use float16 for reduced memory footprint
    - Robust Normalization: Handle outliers using percentiles
    
    Processing Steps:
    1. Validate Nyquist criterion (if seismic)
    2. Apply anti-aliasing filter and downsample (if seismic and dt_decimate > 1)
    3. Normalize using robust statistics (in float32/64)
    4. Convert to float16 for storage
    
    Args:
        arr: Input seismic array
        dt_decimate: The factor by which to downsample the data
        is_seismic: Flag to indicate if the data is seismic or a velocity model
        feedback: An optional feedback collector.
        
    Returns:
        np.ndarray: Preprocessed array
    """
    try:
        if feedback:
            feedback.increment_arrays_processed()

        if is_seismic and dt_decimate > 1:
            if arr.ndim not in [3, 4]:
                logger.warning(f"Unexpected data dimension {arr.ndim} for seismic data. Skipping decimation.")
            else:
                # Determine time axis more robustly
                if arr.ndim == 4:
                    # (batch, sources/channels, time, receivers)
                    if arr.shape[1] == 5:  # sources
                        time_axis = 2
                    elif arr.shape[1] == 1:  # channels
                        time_axis = 2
                    else:
                        # Try to infer time axis - look for the longest dimension
                        time_axis = np.argmax(arr.shape[1:]) + 1
                else:  # 3D
                    # (sources, time, receivers) or (time, receivers, sources)
                    if arr.shape[0] == 5:  # sources first
                        time_axis = 1
                    elif arr.shape[2] == 5:  # sources last
                        time_axis = 0
                    else:
                        # Try to infer time axis - look for the longest dimension
                        time_axis = np.argmax(arr.shape)
                
                # Ensure time_axis is valid
                if time_axis >= arr.ndim:
                    logger.warning(f"Invalid time_axis {time_axis} for data shape {arr.shape}. Skipping decimation.")
                else:
                    # Validate Nyquist criterion
                    if not validate_nyquist(arr, dt_decimate=dt_decimate, feedback=feedback):
                        logger.warning("Data may violate Nyquist criterion after downsampling")
                    
                    # Decimate time axis with anti-aliasing filter
                    try:
                        # Check if the time dimension is large enough for decimation
                        time_dim_size = arr.shape[time_axis]
                        if time_dim_size < dt_decimate * 2:
                            logger.warning(f"Time dimension {time_dim_size} too small for decimation factor {dt_decimate}. Skipping decimation.")
                        else:
                            arr = decimate(arr, dt_decimate, axis=time_axis, ftype='fir')
                    except Exception as e:
                        logger.warning(f"Decimation failed: {e}. Skipping decimation.")
        elif is_seismic and dt_decimate == 1:
            logger.info("No downsampling applied (dt_decimate=1)")
        
        # Robust normalization per trace (in original precision)
        try:
            μ = np.median(arr, keepdims=True)
            σ = np.percentile(arr, 95, keepdims=True) - np.percentile(arr, 5, keepdims=True)
            
            # Avoid division by zero and handle overflow
            if np.isscalar(σ):
                if σ > 1e-6:
                    arr = (arr - μ) / σ
                else:
                    arr = arr - μ
            else:
                # Handle array case
                safe_σ = np.where(σ > 1e-6, σ, 1e-6)
                arr = (arr - μ) / safe_σ
        except Exception as e:
            logger.warning(f"Normalization failed: {e}. Using simple normalization.")
            # Fallback to simple normalization
            try:
                arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            except Exception as e2:
                logger.warning(f"Simple normalization also failed: {e2}. Skipping normalization.")

        # Convert to float16 for storage efficiency AFTER all calculations
        arr = arr.astype('float16')
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        # Return original array on error to avoid crashing the whole pipeline
        return arr
        
    return arr

def preprocess_one_cached(arr: np.ndarray, dt_decimate: int = 4, is_seismic: bool = True, 
                         feedback: Optional[PreprocessingFeedback] = None,
                         use_cache: bool = True) -> np.ndarray:
    """
    Cached version of preprocess_one for inference efficiency.
    
    Args:
        arr: Input seismic array
        dt_decimate: The factor by which to downsample the data
        is_seismic: Flag to indicate if the data is seismic or a velocity model
        feedback: An optional feedback collector
        use_cache: Whether to use caching (recommended for inference)
        
    Returns:
        np.ndarray: Preprocessed array
    """
    if not use_cache:
        return preprocess_one(arr, dt_decimate, is_seismic, feedback)
    
    # Generate hash of input data for caching
    data_hash = hashlib.md5(arr.tobytes()).hexdigest()
    
    # Check cache first
    cache = PreprocessingCache()
    cached_result = cache.get_cached_result(data_hash, dt_decimate, is_seismic)
    
    if cached_result is not None:
        logger.debug(f"Using cached preprocessing result for {data_hash[:8]}...")
        return cached_result
    
    # Process and cache result
    result = preprocess_one(arr, dt_decimate, is_seismic, feedback)
    cache.cache_result(data_hash, dt_decimate, is_seismic, result)
    
    return result

def process_family(family: str, input_path: Union[str, Path], output_dir: Path, data_manager: Optional[DataManager] = None) -> Tuple[List[str], PreprocessingFeedback]:
    """
    Process all files for a given geological family.
    
    Args:
        family: The name of the family to process.
        input_path: The local directory path or the S3 prefix for the family's data.
        output_dir: The directory to save processed files.
        data_manager: Optional DataManager for S3 operations.
        
    Returns:
        A tuple containing a list of processed file paths and a feedback object.
    """
    logger.info(f"Processing family: {family}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feedback = PreprocessingFeedback()

    # Get family-specific settings
    family_config = FAMILY_FILE_MAP.get(family, {})
    seis_glob = family_config.get('seis_glob', '*.npy')
    vel_glob = family_config.get('vel_glob', '*.npy')
    downsample_factor = family_config.get('downsample_factor', 4) # Default to 4 if not specified
    
    logger.info(f"Processing family '{family}' with downsample_factor={downsample_factor}")
    processed_paths = []

    # === S3 Processing Path ===
    if data_manager and data_manager.use_s3:
        if not isinstance(input_path, str):
            raise ValueError(f"For S3 processing, input_path must be a string prefix, but got {type(input_path)}")
            
        # 1. List files directly from S3
        family_s3_prefix = input_path
        
        family_config = FAMILY_FILE_MAP.get(family, {})
        seis_dir = family_config.get('seis_dir', '')
        vel_dir = family_config.get('vel_dir', '')

        full_seis_prefix = f"{family_s3_prefix}/{seis_dir}/" if seis_dir else f"{family_s3_prefix}/"
        full_vel_prefix = f"{family_s3_prefix}/{vel_dir}/" if vel_dir else f"{family_s3_prefix}/"
        
        seis_keys = data_manager.list_s3_files(full_seis_prefix)
        vel_keys = data_manager.list_s3_files(full_vel_prefix)
        
        logger.info(f"Found {len(seis_keys)} seismic files and {len(vel_keys)} velocity files in S3")
        
        # 2. Check if files exist in S3
        if not seis_keys or not vel_keys:
            logger.warning(f"No data files found for family {family} in S3 at prefixes: {full_seis_prefix}, {full_vel_prefix}")
            return [], feedback

        # 3. Loop and process from S3
        pbar = tqdm(zip(sorted(seis_keys), sorted(vel_keys)), total=len(seis_keys), desc=f"Processing {family} from S3")
        for seis_key, vel_key in pbar:
            # Create temporary files in the output directory instead of a temp directory
            local_seis_path = output_dir / f"temp_seis_{Path(seis_key).name}"
            local_vel_path = output_dir / f"temp_vel_{Path(vel_key).name}"
            
            try:
                data_manager.s3_download(seis_key, str(local_seis_path))
                data_manager.s3_download(vel_key, str(local_vel_path))

                seis_arr = np.load(local_seis_path, mmap_mode='r')
                vel_arr = np.load(local_vel_path, mmap_mode='r')
                
                # Apply preprocessing
                seis_arr = preprocess_one_cached(seis_arr, dt_decimate=downsample_factor, is_seismic=True, feedback=feedback)
                vel_arr = preprocess_one_cached(vel_arr, is_seismic=False, feedback=feedback)
                
                out_seis_path = output_dir / f"seis_{Path(seis_key).stem}.npy"
                out_vel_path = output_dir / f"vel_{Path(vel_key).stem}.npy"
                
                np.save(out_seis_path, seis_arr)
                np.save(out_vel_path, vel_arr)
                processed_paths.append(str(out_seis_path))
                processed_paths.append(str(out_vel_path))
                
                # Clean up temporary files
                local_seis_path.unlink(missing_ok=True)
                local_vel_path.unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Failed to process {seis_key}: {e}")
                # Clean up temporary files on error
                local_seis_path.unlink(missing_ok=True)
                local_vel_path.unlink(missing_ok=True)
                continue
    # === Local Processing Path ===
    else:
        if not isinstance(input_path, Path):
            raise ValueError(f"For local processing, input_path must be a Path object, but got {type(input_path)}")

        # 1. List files from the local directory
        input_dir = input_path
        seis_files = sorted(input_dir.glob(seis_glob))
        vel_files = sorted(input_dir.glob(vel_glob))

        # 2. Check if files exist locally
        if not seis_files or not vel_files:
            logger.warning(f"No data files found for family {family} in {input_dir}")
            return [], feedback

        # 3. Loop and process local files
        pbar = tqdm(zip(seis_files, vel_files), total=len(seis_files), desc=f"Processing {family} locally")
        for sfile, vfile in pbar:
            seis_arr = np.load(sfile, mmap_mode='r')
            vel_arr = np.load(vfile, mmap_mode='r')
            
            # Apply preprocessing
            seis_arr = preprocess_one_cached(seis_arr, dt_decimate=downsample_factor, is_seismic=True, feedback=feedback)
            vel_arr = preprocess_one_cached(vel_arr, is_seismic=False, feedback=feedback)

            out_seis_path = output_dir / f"seis_{sfile.stem}.npy"
            out_vel_path = output_dir / f"vel_{vfile.stem}.npy"
            
            np.save(out_seis_path, seis_arr)
            np.save(out_vel_path, vel_arr)
            processed_paths.append(str(out_seis_path))
            processed_paths.append(str(out_vel_path))
            
    logger.info(f"🐛 Processed {len(processed_paths)} files for family {family}")
    logger.info(f"🐛 Processed paths: {processed_paths}")
            
    return processed_paths, feedback

def create_zarr_dataset(processed_paths: List[str], output_path: Path, chunk_size: Tuple[int, ...], data_manager: Optional[DataManager] = None) -> None:
    """
    Create a zarr dataset from processed numpy files with proper shape handling.
    
    Args:
        processed_paths: List of paths to processed numpy files
        output_path: Path where to save the zarr dataset
        chunk_size: Chunk size for the zarr dataset
        data_manager: Optional DataManager for S3 operations
    """
    try:
        if not processed_paths:
            logger.info("No processed paths provided. Skipping Zarr creation.")
            return
            
        # Separate seismic and velocity files
        seismic_paths = []
        velocity_paths = []
        
        for path in processed_paths:
            filename = Path(path).name
            # Check for velocity files first (they contain 'vel' in the name)
            if 'vel' in filename:
                velocity_paths.append(path)
            # Then check for pure seismic files (contain 'seis' but not 'vel')
            elif 'seis' in filename and 'vel' not in filename:
                seismic_paths.append(path)
            else:
                logger.warning(f"Unknown file type: {path}")
        
        logger.info(f"Found {len(seismic_paths)} seismic files and {len(velocity_paths)} velocity files")
        
        # Create a single zarr group with both seismic and velocity data
        if seismic_paths and velocity_paths:
            # Process seismic data
            first_seismic = np.load(seismic_paths[0], mmap_mode='r')
            seismic_shape = first_seismic.shape
            seismic_dtype = first_seismic.dtype
            
            logger.info(f"Processing seismic data with shape: {seismic_shape}, dtype: {seismic_dtype}")
            
            # Create lazy Dask arrays for seismic data
            seismic_arrays = []
            valid_seismic = 0
            for p in seismic_paths:
                try:
                    arr = np.load(p, mmap_mode='r')
                    if arr.shape != seismic_shape:
                        logger.warning(f"Seismic shape mismatch in {p}: expected {seismic_shape}, got {arr.shape}")
                        continue
                    seismic_arrays.append(
                        da.from_delayed(dask.delayed(np.load)(p, allow_pickle=True), shape=seismic_shape, dtype=seismic_dtype)
                    )
                    valid_seismic += 1
                except Exception as e:
                    logger.warning(f"Failed to load seismic file {p}: {e}")
                    continue
            
            logger.info(f"Valid seismic arrays: {valid_seismic}/{len(seismic_paths)}")
            
            # Process velocity data
            first_velocity = np.load(velocity_paths[0], mmap_mode='r')
            velocity_shape = first_velocity.shape
            velocity_dtype = first_velocity.dtype
            
            logger.info(f"Processing velocity data with shape: {velocity_shape}, dtype: {velocity_dtype}")
            
            # Create lazy Dask arrays for velocity data
            velocity_arrays = []
            valid_velocity = 0
            for p in velocity_paths:
                try:
                    arr = np.load(p, mmap_mode='r')
                    if arr.shape != velocity_shape:
                        logger.warning(f"Velocity shape mismatch in {p}: expected {velocity_shape}, got {arr.shape}")
                        continue
                    velocity_arrays.append(
                        da.from_delayed(dask.delayed(np.load)(p, allow_pickle=True), shape=velocity_shape, dtype=velocity_dtype)
                    )
                    valid_velocity += 1
                except Exception as e:
                    logger.warning(f"Failed to load velocity file {p}: {e}")
                    continue
            
            logger.info(f"Valid velocity arrays: {valid_velocity}/{len(velocity_paths)}")
            
            if seismic_arrays and velocity_arrays:
                # Stack arrays
                seismic_stack = da.stack(seismic_arrays, axis=0)
                velocity_stack = da.stack(velocity_arrays, axis=0)
                
                logger.info(f"Seismic stack shape: {seismic_stack.shape}")
                logger.info(f"Velocity stack shape: {velocity_stack.shape}")
                
                # Save both seismic and velocity data in a single zarr file
                save_combined_zarr_data(seismic_stack, velocity_stack, output_path, data_manager)
            else:
                logger.error("No valid arrays found to save")
                return
        else:
            logger.error("Need both seismic and velocity data to create dataset")
            return
                
    except Exception as e:
        logger.error(f"Error creating/uploading zarr dataset: {str(e)}")
        raise

def save_zarr_data(stack, output_path, data_manager):
    """
    Save stacked data to zarr format with proper chunking and S3/local saving.
    
    Args:
        stack: Dask array to save
        output_path: Path to save the data
        data_manager: DataManager instance for S3 operations
    """
    # Get the actual shape after stacking
    stack_shape = stack.shape
    logger.info(f"Stack shape after stacking: {stack_shape}")
    
    # Adjust chunk size based on actual data shape and rechunk the array
    if len(stack_shape) == 5:
        # For 5D data (batch, samples, sources, time, receivers)
        # Use appropriate chunks for each dimension
        adjusted_chunk_size = (
            1,  # batch dimension - keep small for memory efficiency
            min(4, stack_shape[1]),  # samples dimension
            min(4, stack_shape[2]),  # sources dimension  
            min(64, stack_shape[3]),  # time dimension
            min(8, stack_shape[4])   # receivers dimension
        )
    elif len(stack_shape) == 4:
        # For 4D data, use smaller chunks
        adjusted_chunk_size = (1, min(4, stack_shape[1]), min(64, stack_shape[2]), min(8, stack_shape[3]))
    elif len(stack_shape) == 3:
        # For 3D data, use appropriate chunks
        adjusted_chunk_size = (1, min(64, stack_shape[0]), min(8, stack_shape[1]))
    else:
        # For other dimensions, create a default chunk size that matches the dimensions
        adjusted_chunk_size = tuple(1 for _ in range(len(stack_shape)))
        logger.warning(f"Using default chunk size {adjusted_chunk_size} for unexpected shape {stack_shape}")
        
    # Rechunk the array to the desired chunk size
    stack = stack.rechunk(adjusted_chunk_size)
    logger.info(f"Using chunk size: {adjusted_chunk_size}")
    logger.info(f"Stack shape: {stack.shape}, chunks: {stack.chunks}")

    # --- Save to Zarr ---
    # If using S3, save directly to S3. Otherwise, save locally.
    if data_manager and data_manager.use_s3:
        import s3fs
        s3_path = f"s3://{data_manager.s3_bucket}/{output_path.parent.name}/{output_path.name}"
        logger.info(f"Saving zarr dataset directly to S3: {s3_path}")
        
        # Use fsspec-based approach for zarr 3.0.8 with old s3fs compatibility
        try:
            logger.info("Saving to S3 without compression...")
            # Use direct fsspec URL instead of s3fs.S3Map for better compatibility
            stack.to_zarr(s3_path)
            logger.info("Successfully saved to S3 without compression.")
        except Exception as e:
            logger.warning(f"S3 save failed: {e}")
            # Try alternative approach for old s3fs versions
            try:
                logger.info("Trying alternative S3 save method...")
                # Compute the data first, then save
                computed_stack = stack.compute()
                # Use zarr.save with fsspec URL
                zarr.save(s3_path, computed_stack)
                logger.info("Successfully saved to S3 using alternative method.")
            except Exception as e2:
                logger.error(f"All S3 save methods failed: {e2}")
                logger.info("Falling back to local save only...")
                # Save locally as fallback
                try:
                    stack.to_zarr(output_path, component='data')
                    logger.info("Saved locally as fallback.")
                except Exception as e3:
                    logger.error(f"Local fallback also failed: {e3}")
                    # Final fallback - save as numpy arrays
                    try:
                        logger.info("Final fallback: saving as numpy arrays...")
                        computed_stack = stack.compute()
                        np.save(output_path.with_suffix('.npy'), computed_stack)
                        logger.info("Saved as numpy arrays as final fallback.")
                    except Exception as e4:
                        logger.error(f"All save methods failed: {e4}")
                        raise
    else:
        logger.info(f"Saving zarr dataset locally: {output_path}")
        
        # Save without compression
        try:
            logger.info("Saving locally without compression...")
            stack.to_zarr(
                output_path,
                component='data' # Using 'data' as component for local
            )
            logger.info("Successfully saved locally without compression.")
        except Exception as e:
            logger.warning(f"Local save failed: {e}")
            # Final fallback - compute and save
            logger.info("Attempting to save as computed arrays...")
            computed_stack = stack.compute()
            zarr.save(output_path, computed_stack)
            logger.info("Successfully saved locally as computed arrays.")

def split_for_gpus(processed_paths: List[str], output_base: Path, data_manager: Optional[DataManager] = None) -> None:
    """
    Split processed files into two datasets for the two T4 GPUs and optionally upload to S3.
    Simple family-based splitting: put half the families in each GPU dataset.
    """
    try:
        # Get all families from FAMILY_FILE_MAP
        all_families = list(FAMILY_FILE_MAP.keys())
        mid = len(all_families) // 2
        
        # Split families: first half to GPU0, second half to GPU1
        gpu0_families = all_families[:mid]
        gpu1_families = all_families[mid:]
        
        logger.info(f"GPU0 families: {gpu0_families}")
        logger.info(f"GPU1 families: {gpu1_families}")
        
        # Group processed_paths by family
        family_groups = {}
        for path in processed_paths:
            # Extract family name from path
            family = Path(path).parent.name
            family_groups.setdefault(family, []).append(path)
        
        # Assign paths to GPUs based on family
        gpu0_paths = []
        gpu1_paths = []
        
        for family in gpu0_families:
            if family in family_groups:
                gpu0_paths.extend(family_groups[family])
                
        for family in gpu1_families:
            if family in family_groups:
                gpu1_paths.extend(family_groups[family])
            
        # Create GPU-specific directories
        gpu0_dir = output_base / 'gpu0'
        gpu1_dir = output_base / 'gpu1'
        gpu0_dir.mkdir(parents=True, exist_ok=True)
        gpu1_dir.mkdir(parents=True, exist_ok=True)
        
        # Create zarr datasets for each GPU
        create_zarr_dataset(
            gpu0_paths,
            gpu0_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            data_manager
        )
        create_zarr_dataset(
            gpu1_paths,
            gpu1_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            data_manager
        )
        logger.info(f"Created GPU datasets with {len(gpu0_paths)} and {len(gpu1_paths)} samples")
    except Exception as e:
        logger.error(f"Error splitting data for GPUs: {str(e)}")
        raise

def main():
    """
    Main preprocessing pipeline.
    
    Intuition:
    - Command Line Interface: Flexible configuration
    - Family Processing: Handle different geological families
    - GPU Optimization: Split data for parallel processing
    - Cloud Integration: Optional S3 upload
    - Error Handling: Robust error reporting and logging
    """
    # Enable debug mode for testing
    os.environ['DEBUG_MODE'] = '1'
    
    # Filter out Jupyter/Colab specific arguments
    import sys
    filtered_args = [arg for arg in sys.argv if not arg.startswith('-f') and not arg.endswith('.json')]
    sys.argv = filtered_args

    parser = argparse.ArgumentParser(description="Preprocess seismic data for distributed training on T4 GPUs")
    parser.add_argument('--input_root', type=str, default=str(CFG.paths.train), help='Input train_samples root directory')
    parser.add_argument('--output_root', type=str, default='/kaggle/working/preprocessed', help='Output directory for processed files')
    parser.add_argument('--use_s3', action='store_true', help='Use S3 for data processing')
    args = parser.parse_args()

    try:
        input_root = Path(args.input_root)
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        # Initialize DataManager with S3 support if requested
        data_manager = DataManager(use_s3=args.use_s3) if args.use_s3 else None

        # Process each family
        families = list(CFG.paths.families.keys())
        all_processed_paths = []
        all_feedback = {}
        
        for family in families:
            logger.info(f"\nProcessing family: {family}")
            input_dir = input_root / family
            temp_dir = output_root / 'temp' / family
            processed_paths, feedback = process_family(family, input_dir, temp_dir, data_manager)
            all_processed_paths.extend(processed_paths)
            all_feedback[family] = feedback
            logger.info(f"Family {family}: {len(processed_paths)} samples processed")

        # Split and create zarr datasets for GPUs
        logger.info("\nCreating GPU-specific datasets...")
        split_for_gpus(all_processed_paths, output_root, data_manager)
        
        # Clean up temporary files
        temp_dir = output_root / 'temp'
        if temp_dir.exists():
            subprocess.run(['rm', '-rf', str(temp_dir)])
        
        logger.info("\nPreprocessing complete!")
        if data_manager and data_manager.use_s3:
            logger.info(f"Data uploaded to s3://{data_manager.s3_bucket}/preprocessed/")
    except Exception as e:
        logger.error(f"Error in main preprocessing: {str(e)}")
        raise

def load_data_debug(input_root, output_root, use_s3=False, debug_family='FlatVel_A'):
    """
    Debug version of load_data that processes only one family for quick S3 I/O testing.
    
    Args:
        input_root (str): Path to the root of the raw data.
        output_root (str): Path where the processed data will be saved.
        use_s3 (bool): Whether to use S3 for data I/O.
        debug_family (str): Which family to process (default: 'FlatVel_A').
        
    Returns:
        A dictionary containing feedback from the preprocessing run.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    data_manager = DataManager(use_s3=use_s3)
    
    # Process only the specified family
    families = [debug_family]
    
    logger.info(f"🐛 DEBUG MODE: Processing only family '{debug_family}'")
    logger.info(f"🐛 This will help identify S3 I/O issues quickly")
    
    all_processed_paths = []
    all_feedback = {}

    for family in families:
        logger.info(f"🐛 --- Starting debug family: {family} ---")
        family_output_dir = output_root / family

        if use_s3:
            # For S3, the input_path is a prefix string
            family_input_path = f"{input_root}/{family}"
            logger.info(f"🐛 S3 input path: {family_input_path}")
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)
        else:
            # For local, the input_path is a Path object
            family_input_path = Path(input_root) / family
            logger.info(f"🐛 Local input path: {family_input_path}")
            if not family_input_path.exists():
                logger.warning(f"🐛 Skipping family {family}: directory not found at {family_input_path}")
                continue
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)

        all_processed_paths.extend(processed_paths)
        all_feedback[family] = feedback
        
        logger.info(f"🐛 Family {family}: {len(processed_paths)} files processed")

    # Create GPU-specific datasets (simplified for debug mode)
    logger.info("🐛 --- Creating GPU-specific datasets (debug mode) ---")
    
    if all_processed_paths:
        # In debug mode, put all processed files in GPU0 for simplicity
        gpu0_dir = output_root / 'gpu0'
        gpu1_dir = output_root / 'gpu1'
        gpu0_dir.mkdir(parents=True, exist_ok=True)
        gpu1_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a minimal zarr dataset for GPU0
        create_zarr_dataset(
            all_processed_paths,
            gpu0_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            data_manager
        )
        
        # Create an empty zarr dataset for GPU1 (to maintain structure)
        logger.info("🐛 Creating empty GPU1 dataset to maintain structure")
        try:
            import zarr
            # Create empty combined zarr with both seismic and velocity arrays
            root = zarr.group(str(gpu1_dir / 'seismic.zarr'))
            root.create_dataset(
                'seismic', 
                data=np.zeros((0, 5, 500, 70), dtype='float16'),
                shape=(0, 5, 500, 70),  # Explicit shape parameter
                dtype='float16'
            )
            root.create_dataset(
                'velocity', 
                data=np.zeros((0, 1, 70, 70), dtype='float16'),
                shape=(0, 1, 70, 70),  # Explicit shape parameter
                dtype='float16'
            )
        except Exception as e:
            logger.warning(f"🐛 Could not create empty GPU1 dataset: {e}")
        
        logger.info(f"🐛 Created debug GPU datasets with {len(all_processed_paths)} samples in GPU0")
    else:
        logger.warning("🐛 No files processed - cannot create GPU datasets")
    
    # Clean up temporary family directories
    for family in families:
        family_dir = output_root / family
        if family_dir.exists():
            import shutil
            shutil.rmtree(family_dir)
            logger.info(f"🐛 Cleaned up temporary family directory: {family_dir}")
    
    logger.info("🐛 --- Debug preprocessing pipeline complete ---")
    return all_feedback

def load_data(input_root, output_root, use_s3=False):
    """
    Main function to run the complete preprocessing pipeline.
    This function discovers data families, processes them, and stores them in a
    GPU-optimized format (Zarr).
    
    Args:
        input_root (str): Path to the root of the raw data.
        output_root (str): Path where the processed data will be saved.
        use_s3 (bool): Whether to use S3 for data I/O.
        
    Returns:
        A dictionary containing feedback from the preprocessing run.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    data_manager = DataManager(use_s3=use_s3)
    families = FAMILY_FILE_MAP.keys()
    
    all_processed_paths = []
    all_feedback = {}

    for family in families:
        logger.info(f"--- Starting family: {family} ---")
        family_output_dir = output_root / family

        if use_s3:
            # For S3, the input_path is a prefix string
            family_input_path = f"{input_root}/{family}"
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)
        else:
            # For local, the input_path is a Path object
            family_input_path = Path(input_root) / family
            if not family_input_path.exists():
                logger.warning(f"Skipping family {family}: directory not found at {family_input_path}")
                continue
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)

        all_processed_paths.extend(processed_paths)
        all_feedback[family] = feedback

    # Create GPU-specific datasets
    logger.info("--- Creating GPU-specific datasets ---")
    split_for_gpus(all_processed_paths, output_root, data_manager)
    
    # Clean up temporary family directories
    for family in families:
        family_dir = output_root / family
        if family_dir.exists():
            import shutil
            shutil.rmtree(family_dir)
            logger.info(f"Cleaned up temporary family directory: {family_dir}")
    
    logger.info("--- Preprocessing pipeline complete ---")
    return all_feedback

def save_combined_zarr_data(seismic_stack, velocity_stack, output_path, data_manager):
    """
    Save both seismic and velocity data in a single zarr file with proper structure.
    
    Args:
        seismic_stack: Dask array of seismic data
        velocity_stack: Dask array of velocity data  
        output_path: Path to save the zarr file
        data_manager: DataManager instance for S3 operations
    """
    # Get the actual shapes after stacking
    seismic_shape = seismic_stack.shape
    velocity_shape = velocity_stack.shape
    logger.info(f"Seismic stack shape: {seismic_shape}")
    logger.info(f"Velocity stack shape: {velocity_shape}")
    
    # Adjust chunk size based on actual data shape and rechunk the arrays
    if len(seismic_shape) == 5:
        # For 5D data (batch, samples, sources, time, receivers)
        seismic_chunk_size = (
            1,  # batch dimension - keep small for memory efficiency
            min(4, seismic_shape[1]),  # samples dimension
            min(4, seismic_shape[2]),  # sources dimension  
            min(64, seismic_shape[3]),  # time dimension
            min(8, seismic_shape[4])   # receivers dimension
        )
    elif len(seismic_shape) == 4:
        # For 4D data, use smaller chunks
        seismic_chunk_size = (1, min(4, seismic_shape[1]), min(64, seismic_shape[2]), min(8, seismic_shape[3]))
    else:
        # For other dimensions, create a default chunk size
        seismic_chunk_size = tuple(1 for _ in range(len(seismic_shape)))
        logger.warning(f"Using default chunk size {seismic_chunk_size} for unexpected seismic shape {seismic_shape}")
    
    # Similar chunking for velocity data
    if len(velocity_shape) == 5:
        # For 5D velocity data (batch, samples, channels, height, width)
        velocity_chunk_size = (
            1,  # batch dimension - keep small for memory efficiency
            min(4, velocity_shape[1]),  # samples dimension
            min(1, velocity_shape[2]),  # channels dimension (usually 1 for velocity)
            min(8, velocity_shape[3]),  # height dimension
            min(8, velocity_shape[4])   # width dimension
        )
    elif len(velocity_shape) == 4:
        # For 4D velocity data (batch, channels, height, width)
        velocity_chunk_size = (1, min(4, velocity_shape[1]), min(8, velocity_shape[2]), min(8, velocity_shape[3]))
    elif len(velocity_shape) == 3:
        # For 3D velocity data (channels, height, width)
        velocity_chunk_size = (1, min(8, velocity_shape[1]), min(8, velocity_shape[2]))
    else:
        velocity_chunk_size = tuple(1 for _ in range(len(velocity_shape)))
        logger.warning(f"Using default chunk size {velocity_chunk_size} for unexpected velocity shape {velocity_shape}")
    
    # Rechunk the arrays
    seismic_stack = seismic_stack.rechunk(seismic_chunk_size)
    velocity_stack = velocity_stack.rechunk(velocity_chunk_size)
    
    logger.info(f"Using seismic chunk size: {seismic_chunk_size}")
    logger.info(f"Using velocity chunk size: {velocity_chunk_size}")

    # --- Save to Zarr ---
    # If using S3, save directly to S3. Otherwise, save locally.
    if data_manager and data_manager.use_s3:
        import s3fs
        s3_path = f"s3://{data_manager.s3_bucket}/{output_path.parent.name}/{output_path.name}"
        logger.info(f"Saving combined zarr dataset to S3: {s3_path}")
        
        try:
            # Use dask's to_zarr method which handles S3 better
            logger.info("Attempting S3 save with dask.to_zarr...")
            
            # Compute the data first to avoid S3 compatibility issues
            seismic_data = seismic_stack.compute()
            velocity_data = velocity_stack.compute()
            
            # Create a temporary local zarr file first
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_zarr_path = Path(tmpdir) / "temp.zarr"
                
                # Save locally first
                import zarr
                root = zarr.group(str(temp_zarr_path))
                
                # Save seismic data
                root.create_dataset(
                    'seismic', 
                    data=seismic_data,
                    chunks=seismic_chunk_size,
                    dtype='float16'
                )
                
                # Save velocity data
                root.create_dataset(
                    'velocity',
                    data=velocity_data, 
                    chunks=velocity_chunk_size,
                    dtype='float16'
                )
                
                # Now upload the entire zarr directory to S3
                import shutil
                import subprocess
                
                # Use aws CLI to sync the zarr directory to S3
                s3_uri = f"s3://{data_manager.s3_bucket}/{output_path.parent.name}/{output_path.name}"
                try:
                    subprocess.run([
                        'aws', 's3', 'sync', str(temp_zarr_path), s3_uri, '--quiet'
                    ], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback: use boto3 to upload files
                    logger.info("AWS CLI not available, using boto3 fallback...")
                    import boto3
                    s3_client = boto3.client('s3')
                    
                    # Upload each file in the zarr directory
                    for file_path in temp_zarr_path.rglob('*'):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(temp_zarr_path)
                            s3_key = f"{output_path.parent.name}/{output_path.name}/{relative_path}"
                            s3_client.upload_file(str(file_path), data_manager.s3_bucket, str(s3_key))
                
                logger.info("Successfully saved combined dataset to S3")
            
        except Exception as e:
            logger.warning(f"S3 save failed: {e}")
            # Fallback to local save
            logger.info("Falling back to local save...")
            save_combined_zarr_local(seismic_stack, velocity_stack, output_path)
            
    else:
        logger.info(f"Saving combined zarr dataset locally: {output_path}")
        save_combined_zarr_local(seismic_stack, velocity_stack, output_path)

def save_combined_zarr_local(seismic_stack, velocity_stack, output_path):
    """
    Save combined zarr dataset locally.
    """
    try:
        import zarr
        
        # Create zarr group
        root = zarr.group(str(output_path))
        
        # Compute the data first to get actual shapes
        seismic_data = seismic_stack.compute()
        velocity_data = velocity_stack.compute()
        
        # Save seismic data with explicit shape
        seismic_array = root.create_dataset(
            'seismic',
            data=seismic_data,
            chunks=seismic_stack.chunks,
            dtype='float16',
            shape=seismic_data.shape  # Explicitly provide shape
        )
        
        # Save velocity data with explicit shape
        velocity_array = root.create_dataset(
            'velocity',
            data=velocity_data,
            chunks=velocity_stack.chunks,
            dtype='float16',
            shape=velocity_data.shape  # Explicitly provide shape
        )
        
        logger.info("Successfully saved combined dataset locally")
        
    except Exception as e:
        logger.error(f"Local save failed: {e}")
        # Final fallback - save as numpy arrays
        try:
            logger.info("Final fallback: saving as numpy arrays...")
            seismic_data = seismic_stack.compute()
            velocity_data = velocity_stack.compute()
            np.save(output_path.with_suffix('.seismic.npy'), seismic_data)
            np.save(output_path.with_suffix('.velocity.npy'), velocity_data)
            logger.info("Saved as numpy arrays as final fallback")
        except Exception as e2:
            logger.error(f"All save methods failed: {e2}")
            raise

def preprocess_test_data_batch(test_files: List[Path], 
                              output_dir: Path,
                              dt_decimate: int = 4,
                              batch_size: int = 100,
                              num_workers: int = 4,
                              use_cache: bool = True) -> List[Path]:
    """
    Efficiently preprocess test data in batches for inference.
    
    Args:
        test_files: List of test file paths
        output_dir: Directory to save preprocessed files
        dt_decimate: Downsampling factor
        batch_size: Number of files to process in parallel
        num_workers: Number of parallel workers
        use_cache: Whether to use preprocessing cache
        
    Returns:
        List of preprocessed file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_files = []
    
    logger.info(f"Preprocessing {len(test_files)} test files in batches of {batch_size}")
    
    def process_file_batch(file_batch):
        """Process a batch of files."""
        batch_results = []
        for file_path in file_batch:
            try:
                # Load data
                data = np.load(file_path, mmap_mode='r')
                
                # Preprocess with caching
                processed_data = preprocess_one_cached(
                    data, dt_decimate=dt_decimate, is_seismic=True, 
                    use_cache=use_cache
                )
                
                # Save preprocessed file
                output_file = output_dir / f"preprocessed_{file_path.stem}.npy"
                np.save(output_file, processed_data)
                batch_results.append(output_file)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
                
        return batch_results
    
    # Process files in batches
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for i in range(0, len(test_files), batch_size):
            batch = test_files[i:i + batch_size]
            future = executor.submit(process_file_batch, batch)
            futures.append(future)
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                batch_results = future.result()
                processed_files.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
    
    logger.info(f"Successfully processed {len(processed_files)} test files")
    return processed_files

def create_inference_optimized_dataset(test_files: List[Path],
                                     output_dir: Path,
                                     chunk_size: Tuple[int, ...] = (1, 4, 64, 8),
                                     use_cache: bool = True) -> Path:
    """
    Create an inference-optimized zarr dataset for test data.
    
    Args:
        test_files: List of test file paths
        output_dir: Directory to save the dataset
        chunk_size: Chunk size for zarr dataset
        use_cache: Whether to use preprocessing cache
        
    Returns:
        Path to the created zarr dataset
    """
    logger.info(f"Creating inference-optimized dataset for {len(test_files)} test files")
    
    # Preprocess all test files
    preprocessed_files = preprocess_test_data_batch(
        test_files, output_dir / 'temp', use_cache=use_cache
    )
    
    # Create zarr dataset
    zarr_path = output_dir / 'test_data.zarr'
    
    try:
        # Load all preprocessed data
        all_data = []
        for file_path in tqdm(preprocessed_files, desc="Loading preprocessed data"):
            data = np.load(file_path)
            all_data.append(data)
        
        # Stack into single array
        stacked_data = np.stack(all_data, axis=0)
        logger.info(f"Stacked data shape: {stacked_data.shape}")
        
        # Save as zarr dataset
        import dask.array as da
        dask_array = da.from_array(stacked_data, chunks=chunk_size)
        dask_array.to_zarr(str(zarr_path))
        
        logger.info(f"Created inference dataset at {zarr_path}")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(output_dir / 'temp')
        
        return zarr_path
        
    except Exception as e:
        logger.error(f"Failed to create inference dataset: {e}")
        raise

def preprocess_for_inference(data: np.ndarray, 
                           dt_decimate: int = 4,
                           use_cache: bool = True) -> np.ndarray:
    """
    Inference-optimized preprocessing function that matches training preprocessing exactly.
    
    Args:
        data: Input seismic data
        dt_decimate: Downsampling factor (must match training)
        use_cache: Whether to use caching
        
    Returns:
        Preprocessed data in float16
    """
    return preprocess_one_cached(
        data, dt_decimate=dt_decimate, is_seismic=True, use_cache=use_cache
    )

if __name__ == "__main__":
    main() 


# %%
# Source: src/core/registry.py
import os
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Singleton class for managing model versions with geometric metadata.
    Ensures only one registry instance exists across the application.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, registry_dir: str = "models"):
        """
        Initialize the registry if not already initialized.
        
        Args:
            registry_dir: Directory to store model versions
        """
        if self._initialized:
            return
            
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self._initialized = True
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}}
    
    def _save_metadata(self):
        """Save current metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, 
                      model: nn.Module,
                      model_id: str,
                      family: str,
                      equivariance: List[str],
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new model version with geometric metadata.
        
        Args:
            model: PyTorch model to register
            model_id: Unique identifier for the model
            family: Geological family the model is trained on
            equivariance: List of geometric transformations the model is equivariant to
            metadata: Additional metadata to store
            
        Returns:
            str: Version ID of the registered model
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_id}_v{timestamp}"
        
        # Create model directory
        model_dir = self.registry_dir / version_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), model_dir / "model.pt")
        
        # Prepare metadata
        model_metadata = {
            "model_id": model_id,
            "version": version_id,
            "family": family,
            "equivariance": equivariance,
            "timestamp": timestamp,
            "architecture": str(model),
            "state_dict_keys": list(model.state_dict().keys()),
            **(metadata or {})
        }
        
        # Update registry
        self.metadata["models"][version_id] = model_metadata
        self._save_metadata()
        
        logger.info(f"Registered model {version_id} with {len(equivariance)} equivariance properties")
        return version_id
    
    def load_model(self, version_id: str, model_class: type) -> nn.Module:
        """
        Load a registered model version.
        
        Args:
            version_id: Version ID of the model to load
            model_class: PyTorch model class to instantiate
            
        Returns:
            nn.Module: Loaded model
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        model_dir = self.registry_dir / version_id
        state_dict = torch.load(model_dir / "model.pt")
        
        # Instantiate model and load state
        model = model_class()
        model.load_state_dict(state_dict)
        
        logger.info(f"Loaded model {version_id} with {len(self.metadata['models'][version_id]['equivariance'])} equivariance properties")
        return model
    
    def list_models(self, family: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered models, optionally filtered by family.
        
        Args:
            family: Optional family to filter by
            
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        models = self.metadata["models"]
        if family:
            return [m for m in models.values() if m["family"] == family]
        return list(models.values())
    
    def get_model_info(self, version_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model version.
        
        Args:
            version_id: Version ID of the model
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        return self.metadata["models"][version_id]
    
    def update_metadata(self, version_id: str, metadata: Dict[str, Any]):
        """
        Update metadata for a specific model version.
        
        Args:
            version_id: Version ID of the model
            metadata: New metadata to add/update
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        self.metadata["models"][version_id].update(metadata)
        self._save_metadata()
        logger.info(f"Updated metadata for model {version_id}")
    
    def delete_model(self, version_id: str):
        """
        Delete a model version from the registry.
        
        Args:
            version_id: Version ID of the model to delete
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        # Remove model files
        model_dir = self.registry_dir / version_id
        if model_dir.exists():
            for file in model_dir.glob("*"):
                file.unlink()
            model_dir.rmdir()
        
        # Remove from metadata
        del self.metadata["models"][version_id]
        self._save_metadata()
        logger.info(f"Deleted model {version_id}") 


# %%
# Source: src/core/checkpoint.py
import os
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Singleton class for managing model checkpoints with geometric metadata.
    Ensures only one checkpoint manager exists across the application.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint manager if not already initialized.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        if self._initialized:
            return
            
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        self._initialized = True
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"checkpoints": {}}
    
    def _save_metadata(self):
        """Save current metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       model_id: str,
                       family: str,
                       equivariance: List[str],
                       metrics: Dict[str, float],
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model checkpoint with geometric metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current training epoch
            model_id: Unique identifier for the model
            family: Geological family being trained
            equivariance: List of geometric transformations the model is equivariant to
            metrics: Dictionary of training metrics
            metadata: Additional metadata to store
            
        Returns:
            str: Checkpoint ID
        """
        # Generate checkpoint ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{model_id}_epoch{epoch}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path / "checkpoint.pt")
        
        # Prepare metadata
        checkpoint_metadata = {
            "checkpoint_id": checkpoint_id,
            "model_id": model_id,
            "family": family,
            "epoch": epoch,
            "equivariance": equivariance,
            "timestamp": timestamp,
            "metrics": metrics,
            **(metadata or {})
        }
        
        # Update registry
        self.metadata["checkpoints"][checkpoint_id] = checkpoint_metadata
        self._save_metadata()
        
        logger.info(f"Saved checkpoint {checkpoint_id} at epoch {epoch}")
        return checkpoint_id
    
    def load_checkpoint(self,
                       checkpoint_id: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, Dict[str, float]]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            model: PyTorch model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Tuple[int, Dict[str, float]]: (epoch, metrics)
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint {checkpoint_id} from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def list_checkpoints(self,
                        model_id: Optional[str] = None,
                        family: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints, optionally filtered by model or family.
        
        Args:
            model_id: Optional model ID to filter by
            family: Optional family to filter by
            
        Returns:
            List[Dict[str, Any]]: List of checkpoint metadata
        """
        checkpoints = self.metadata["checkpoints"]
        filtered = checkpoints.values()
        
        if model_id:
            filtered = [c for c in filtered if c["model_id"] == model_id]
        if family:
            filtered = [c for c in filtered if c["family"] == family]
            
        return sorted(filtered, key=lambda x: x["epoch"])
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint
            
        Returns:
            Dict[str, Any]: Checkpoint metadata
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        return self.metadata["checkpoints"][checkpoint_id]
    
    def update_metadata(self, checkpoint_id: str, metadata: Dict[str, Any]):
        """
        Update metadata for a specific checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint
            metadata: New metadata to add/update
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        self.metadata["checkpoints"][checkpoint_id].update(metadata)
        self._save_metadata()
        logger.info(f"Updated metadata for checkpoint {checkpoint_id}")
    
    def delete_checkpoint(self, checkpoint_id: str):
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Remove checkpoint files
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        
        # Remove from metadata
        del self.metadata["checkpoints"][checkpoint_id]
        self._save_metadata()
        logger.info(f"Deleted checkpoint {checkpoint_id}")
    
    def get_best_checkpoint(self,
                          model_id: str,
                          metric: str = "val_loss",
                          family: Optional[str] = None) -> Optional[str]:
        """
        Get the best checkpoint based on a metric.
        
        Args:
            model_id: Model ID to search for
            metric: Metric to optimize (default: val_loss)
            family: Optional family to filter by
            
        Returns:
            Optional[str]: ID of the best checkpoint, or None if no checkpoints found
        """
        checkpoints = self.list_checkpoints(model_id, family)
        if not checkpoints:
            return None
            
        # Sort by metric (lower is better for loss metrics)
        is_loss = "loss" in metric.lower()
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda x: x["metrics"].get(metric, float('inf') if is_loss else float('-inf')),
            reverse=not is_loss
        )
        
        return sorted_checkpoints[0]["checkpoint_id"] 


# %%
# Source: src/core/geometric_loader.py
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import json
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

class GeometricDataset(Dataset):
    """
    Dataset class for handling seismic data with geometric features.
    Extracts and manages geometric features for different geological families.
    """
    
    def __init__(self,
                 data_path: str,
                 family: str,
                 transform: Optional[Any] = None,
                 extract_features: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the zarr dataset
            family: Geological family name
            transform: Optional data transformations
            extract_features: Whether to extract geometric features
        """
        self.data_path = Path(data_path)
        self.family = family
        self.transform = transform
        self.extract_features = extract_features
        
        # Load zarr array
        self.data = zarr.open(self.data_path / 'seis', mode='r')
        
        # Load family metadata if exists
        metadata_path = self.data_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def extract_geometric_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract geometric features from seismic data.
        
        Args:
            data: Seismic data array
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of geometric features
        """
        features = {}
        
        # Extract structural features
        features['gradient_magnitude'] = np.gradient(data)[0]  # Time gradient
        features['gradient_direction'] = np.arctan2(np.gradient(data)[1], np.gradient(data)[0])
        
        # Extract boundary features using Canny edge detection
        features['edges'] = canny(data, sigma=2.0)
        
        # Extract spectral features
        fft_data = np.fft.rfft(data, axis=0)
        features['spectral_energy'] = np.abs(fft_data)
        features['spectral_phase'] = np.angle(fft_data)
        
        # Extract multi-scale features using Gaussian blur
        features['gaussian_1'] = gaussian_filter(data, sigma=1.0)
        features['gaussian_2'] = gaussian_filter(data, sigma=2.0)
        features['gaussian_4'] = gaussian_filter(data, sigma=4.0)
        
        return features
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample with geometric features.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing data and features
        """
        # Load seismic data
        data = self.data[idx]
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(data).float()
        
        # Extract geometric features if requested
        if self.extract_features:
            features = self.extract_geometric_features(data)
            feature_tensors = {
                k: torch.from_numpy(v).float()
                for k, v in features.items()
            }
        else:
            feature_tensors = {}
        
        # Apply transformations if any
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)
            if self.extract_features:
                feature_tensors = {
                    k: self.transform(v)
                    for k, v in feature_tensors.items()
                }
        
        return {
            'data': data_tensor,
            'features': feature_tensors,
            'family': self.family,
            'index': idx
        }

class FamilyDataLoader:
    """
    Singleton class for managing family-specific data loading with geometric features.
    Ensures consistent data loading patterns and prevents memory issues.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self,
                 data_root: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 transform: Optional[Callable] = None,
                 extract_features: bool = True):
        """
        Initialize the family data loader if not already initialized.
        
        Args:
            data_root: Root directory containing family data
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            transform: Optional transform to apply
            extract_features: Whether to extract geometric features
        """
        if self._initialized:
            return
            
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.extract_features = extract_features
        
        # Initialize datasets and loaders
        self.datasets = {}
        self.loaders = {}
        self._initialized = True
        
    def set_parameters(self, batch_size: int = None, num_workers: int = None):
        """Update parameters if needed after initialization"""
        if batch_size is not None:
            self.batch_size = batch_size
        if num_workers is not None:
            self.num_workers = num_workers
        # Reinitialize loaders with new parameters
        self._initialize_loaders()
    
    def _initialize_loaders(self):
        # Find all family directories
        family_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        
        for family_dir in family_dirs:
            family = family_dir.name
            dataset = GeometricDataset(
                family_dir,
                family,
                transform=self.transform,
                extract_features=self.extract_features
            )
            self.datasets[family] = dataset
            
            # Create data loader
            self.loaders[family] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
    
    def get_loader(self, family: str) -> DataLoader:
        """
        Get data loader for a specific family.
        
        Args:
            family: Family name
            
        Returns:
            DataLoader: Data loader for the family
        """
        if family not in self.loaders:
            raise ValueError(f"Family {family} not found")
        return self.loaders[family]
    
    def get_dataset(self, family: str) -> GeometricDataset:
        """
        Get dataset for a specific family.
        
        Args:
            family: Family name
            
        Returns:
            GeometricDataset: Dataset for the family
        """
        if family not in self.datasets:
            raise ValueError(f"Family {family} not found")
        return self.datasets[family]
    
    def get_all_loaders(self) -> Dict[str, DataLoader]:
        """
        Get all data loaders.
        
        Returns:
            Dict[str, DataLoader]: Dictionary of all data loaders
        """
        return self.loaders
    
    def get_all_datasets(self) -> Dict[str, GeometricDataset]:
        """
        Get all datasets.
        
        Returns:
            Dict[str, GeometricDataset]: Dictionary of all datasets
        """
        return self.datasets
    
    def get_family_stats(self, family: str) -> Dict[str, float]:
        """
        Get statistics for a specific family.
        
        Args:
            family: Family name
            
        Returns:
            Dict[str, float]: Dictionary of statistics
        """
        dataset = self.get_dataset(family)
        data = dataset.data
        
        stats = {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'samples': len(dataset)
        }
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all families.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of statistics for each family
        """
        return {
            family: self.get_family_stats(family)
            for family in self.datasets.keys()
        } 


# %%
# Source: src/core/geometric_cv.py
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
import json

# Optional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)

class GeometricCrossValidator:
    """
    Singleton class for implementing geometric-aware cross-validation.
    Ensures consistent validation strategy across training.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize the cross-validator if not already initialized.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed for reproducibility
        """
        if self._initialized:
            return
            
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._initialized = True
    
    def set_parameters(self, n_splits: int = None, shuffle: bool = None, random_state: int = None):
        """Update parameters if needed after initialization"""
        if n_splits is not None:
            self.n_splits = n_splits
        if shuffle is not None:
            self.shuffle = shuffle
        if random_state is not None:
            self.random_state = random_state
        # Reinitialize kfold objects with new parameters
        self.kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        self.stratified_kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
    
    def compute_geometric_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute geometric metrics between true and predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of geometric metrics
        """
        metrics = {}
        
        # Compute SSIM
        metrics['ssim'] = ssim(y_true, y_pred, data_range=y_true.max() - y_true.min())
        
        # Compute gradient magnitude similarity
        grad_true = np.gradient(y_true)[0]
        grad_pred = np.gradient(y_pred)[0]
        metrics['gradient_similarity'] = ssim(grad_true, grad_pred, data_range=grad_true.max() - grad_true.min())
        
        # Compute boundary preservation
        edges_true = canny(y_true, sigma=2.0)
        edges_pred = canny(y_pred, sigma=2.0)
        metrics['boundary_iou'] = np.logical_and(edges_true, edges_pred).sum() / np.logical_or(edges_true, edges_pred).sum()
        
        return metrics
    
    def split_by_family(self,
                       dataset: Dataset,
                       family_labels: List[str]) -> List[Tuple[Subset, Subset]]:
        """
        Split dataset by geological family.
        
        Args:
            dataset: PyTorch dataset
            family_labels: List of family labels for each sample
            
        Returns:
            List[Tuple[Subset, Subset]]: List of (train, val) splits
        """
        splits = []
        
        # Get unique families
        families = np.unique(family_labels)
        
        for family in families:
            # Get indices for this family
            family_indices = np.where(np.array(family_labels) == family)[0]
            
            # Split indices
            for train_idx, val_idx in self.kfold.split(family_indices):
                train_subset = Subset(dataset, family_indices[train_idx])
                val_subset = Subset(dataset, family_indices[val_idx])
                splits.append((train_subset, val_subset))
        
        return splits
    
    def split_by_geometry(self,
                         dataset: Dataset,
                         geometric_features: Dict[str, np.ndarray]) -> List[Tuple[Subset, Subset]]:
        """
        Split dataset by geometric features.
        
        Args:
            dataset: PyTorch dataset
            geometric_features: Dictionary of geometric features
            
        Returns:
            List[Tuple[Subset, Subset]]: List of (train, val) splits
        """
        splits = []
        
        # Combine geometric features into a single feature vector
        feature_matrix = np.column_stack([
            features.reshape(len(dataset), -1)
            for features in geometric_features.values()
        ])
        
        # Use stratified k-fold on geometric features
        for train_idx, val_idx in self.stratified_kfold.split(feature_matrix, np.zeros(len(dataset))):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            splits.append((train_subset, val_subset))
        
        return splits
    
    def log_geometric_metrics_mlflow(self, metrics: dict, prefix: str = ""):
        """Utility to log geometric metrics to MLflow if available."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}{k}", v)
        except Exception:
            pass
    
    def evaluate_fold(self,
                     model: torch.nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader,
                     device: torch.device) -> Dict[str, float]:
        """
        Evaluate a single fold.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to run evaluation on
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        model.eval()
        metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_ssim': 0.0,
            'val_ssim': 0.0,
            'train_boundary_iou': 0.0,
            'val_boundary_iou': 0.0
        }
        
        # Evaluate training set
        with torch.no_grad():
            for batch in train_loader:
                data = batch['data'].to(device)
                target = batch['target'].to(device)
                output = model(data)
                
                # Compute geometric metrics
                geom_metrics = self.compute_geometric_metrics(
                    target.cpu().numpy(),
                    output.cpu().numpy()
                )
                
                metrics['train_ssim'] += geom_metrics['ssim']
                metrics['train_boundary_iou'] += geom_metrics['boundary_iou']
        
        # Evaluate validation set
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(device)
                target = batch['target'].to(device)
                output = model(data)
                
                # Compute geometric metrics
                geom_metrics = self.compute_geometric_metrics(
                    target.cpu().numpy(),
                    output.cpu().numpy()
                )
                
                metrics['val_ssim'] += geom_metrics['ssim']
                metrics['val_boundary_iou'] += geom_metrics['boundary_iou']
        
        # Average metrics
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        metrics['train_ssim'] /= n_train
        metrics['train_boundary_iou'] /= n_train
        metrics['val_ssim'] /= n_val
        metrics['val_boundary_iou'] /= n_val
        
        # Log to MLflow
        self.log_geometric_metrics_mlflow(metrics, prefix="fold_")
        
        return metrics
    
    def cross_validate(self,
                      model: torch.nn.Module,
                      dataset: Dataset,
                      family_labels: List[str],
                      geometric_features: Dict[str, np.ndarray],
                      batch_size: int = 32,
                      device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, List[float]]:
        """
        Perform cross-validation with both family and geometric stratification.
        
        Args:
            model: PyTorch model
            dataset: PyTorch dataset
            family_labels: List of family labels
            geometric_features: Dictionary of geometric features
            batch_size: Batch size for data loaders
            device: Device to run evaluation on
            
        Returns:
            Dict[str, List[float]]: Dictionary of metrics for each fold
        """
        # Get splits
        family_splits = self.split_by_family(dataset, family_labels)
        geometry_splits = self.split_by_geometry(dataset, geometric_features)
        
        # Combine splits
        splits = []
        for (train_fam, val_fam), (train_geom, val_geom) in zip(family_splits, geometry_splits):
            # Combine train and val sets
            train_indices = list(set(train_fam.indices) & set(train_geom.indices))
            val_indices = list(set(val_fam.indices) & set(val_geom.indices))
            
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            splits.append((train_subset, val_subset))
        
        # Evaluate each fold
        fold_metrics = []
        for i, (train_subset, val_subset) in enumerate(splits):
            logger.info(f"Evaluating fold {i+1}/{len(splits)}")
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Evaluate fold
            metrics = self.evaluate_fold(model, train_loader, val_loader, device)
            # Log fold metrics to MLflow
            self.log_geometric_metrics_mlflow(metrics, prefix=f"fold{i+1}_")
            fold_metrics.append(metrics)
        
        # Aggregate results
        results = {
            'train_ssim': [m['train_ssim'] for m in fold_metrics],
            'val_ssim': [m['val_ssim'] for m in fold_metrics],
            'train_boundary_iou': [m['train_boundary_iou'] for m in fold_metrics],
            'val_boundary_iou': [m['val_boundary_iou'] for m in fold_metrics]
        }
        
        # Add mean and std
        for metric in results:
            values = results[metric]
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        
        return results
    
    def save_results(self,
                    results: Dict[str, List[float]],
                    output_path: str):
        """
        Save cross-validation results to file.
        
        Args:
            results: Dictionary of results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved cross-validation results to {output_path}") 


# %%
# Source: src/core/data_manager.py
"""
DataManager is the single source of truth for all data IO in this project.
All data loading, streaming, and batching must go through DataManager.
"""
# Standard library imports
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import os
import socket
import tempfile
import shutil
import mmap
import logging
import json

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Local imports
from src.core.config import CFG, FAMILY_FILE_MAP

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

class DataManager:
    """
    Singleton class for managing all data IO operations.
    Ensures consistent data access and prevents memory leaks in Kaggle notebook environment.
    Handles both local and S3 data operations.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, data_root: str = None, use_s3: bool = False):
        """
        Initialize the data manager if not already initialized.
        
        Args:
            data_root: Root directory for data
            use_s3: Whether to use S3 for data operations
        """
        if self._initialized:
            return
            
        self.data_root = Path(data_root) if data_root else None
        self._cache = {}
        self.use_s3 = use_s3
        self.use_mmap = True
        self.memory_tracker = MemoryTracker()
        
        if use_s3:
            self._setup_s3()
            
        self._initialized = True
    
    def s3_download(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3 to a local path.
        
        Args:
            s3_key: The key of the object in the S3 bucket.
            local_path: The local path to save the file to.
            
        Returns:
            True if download was successful, False otherwise.
        """
        if not self.use_s3:
            raise RuntimeError("S3 operations not enabled")
        
        try:
            with open(local_path, 'wb') as f:
                self.s3.download_fileobj(self.s3_bucket, s3_key, f)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.error(f"S3 object not found: {s3_key}")
            else:
                logging.error(f"Failed to download {s3_key} from S3: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during S3 download: {e}")
            return False

    def _setup_s3(self):
        """Set up S3 client and configuration."""
        try:
            # Try to load credentials from .env/aws/credentials.json
            creds_path = Path('.env/aws/credentials.json')
            if creds_path.exists():
                with open(creds_path, 'r') as f:
                    credentials = json.load(f)
            else:
                # Fall back to environment variables
                credentials = {
                    'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
                    'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    'region_name': os.environ.get('AWS_REGION', 'us-east-1')
                }
            
            self.s3 = boto3.client('s3', **credentials)
            self.s3_bucket = credentials.get('s3_bucket') or os.environ.get('AWS_S3_BUCKET')
            self.cache_dir = Path(tempfile.gettempdir()) / 'gwi_cache'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.s3_bucket:
                raise ValueError("S3 bucket not specified in credentials or environment variable AWS_S3_BUCKET")
                
        except Exception as e:
            logging.error(f"Failed to set up S3: {e}")
            self.use_s3 = False
            raise
    
    def stream_from_s3(self, s3_key: str, chunk_size: int = 1024*1024) -> np.ndarray:
        """
        Stream a file from S3 in chunks with memory mapping.
        
        Args:
            s3_key: S3 object key
            chunk_size: Size of chunks to download
            
        Returns:
            np.ndarray: Memory-mapped array of the data
        """
        if not self.use_s3:
            raise RuntimeError("S3 streaming not enabled")
            
        try:
            # Get object size
            response = self.s3.head_object(Bucket=self.s3_bucket, Key=s3_key)
            file_size = response['ContentLength']
            
            # Create memory-mapped array
            temp_file = self.cache_dir / s3_key.replace('/', '_')
            if not temp_file.exists():
                with open(temp_file, 'wb') as f:
                    for i in range(0, file_size, chunk_size):
                        end = min(i + chunk_size, file_size)
                        response = self.s3.get_object(
                            Bucket=self.s3_bucket,
                            Key=s3_key,
                            Range=f'bytes={i}-{end-1}'
                        )
                        f.write(response['Body'].read())
            
            return np.load(temp_file, mmap_mode='r')
            
        except ClientError as e:
            logging.error(f"Error streaming file {s3_key}: {e}")
            raise
    
    def upload_to_s3(self, data: np.ndarray, s3_key: str) -> bool:
        """
        Upload processed data to S3.
        
        Args:
            data: Data to upload
            s3_key: S3 object key
            
        Returns:
            bool: True if upload successful
        """
        if not self.use_s3:
            raise RuntimeError("S3 upload not enabled")
            
        try:
            # Save to temporary file
            temp_file = self.cache_dir / f"temp_{s3_key.replace('/', '_')}"
            np.save(temp_file, data)
            
            # Upload to S3
            self.s3.upload_file(
                str(temp_file),
                self.s3_bucket,
                s3_key
            )
            
            # Clean up
            temp_file.unlink()
            return True
            
        except Exception as e:
            logging.error(f"Error uploading to S3: {e}")
            return False
    
    def list_s3_files(self, prefix: str) -> List[str]:
        """
        List files in S3 bucket with given prefix.
        
        Args:
            prefix: S3 key prefix
            
        Returns:
            List[str]: List of S3 keys
        """
        if not self.use_s3:
            raise RuntimeError("S3 operations not enabled")
            
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix,
                MaxKeys=1000
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logging.error(f"Error listing S3 objects: {e}")
            return []
    
    def list_family_files(self, family: str) -> Tuple[List[Path], List[Path], str]:
        """
        Return (seis_files, vel_files, family_type) for a given family.
        Handles both local and S3 data sources.
        """
        if CFG.debug_mode and family in CFG.families_to_exclude:
            logging.info(f"Skipping excluded family in debug mode: {family}")
            return None, None, None
        
        if self.use_s3:
            # Use S3Paths config and FAMILY_FILE_MAP for all S3 prefixes
            s3_family_prefix = CFG.s3_paths.families[family]
            info = FAMILY_FILE_MAP[family]
            # Vel/Style: data/model subfolders (batched)
            seis_prefix = f"{s3_family_prefix}/{info['seis_dir']}/" if info['seis_dir'] else f"{s3_family_prefix}/"
            vel_prefix = f"{s3_family_prefix}/{info['vel_dir']}/" if info['vel_dir'] else f"{s3_family_prefix}/"
            seis_files = self.list_s3_files(seis_prefix)
            vel_files = self.list_s3_files(vel_prefix)
            if seis_files and vel_files:
                if CFG.debug_mode:
                    seis_files = seis_files[:1]
                    vel_files = vel_files[:1]
                return seis_files, vel_files, 'VelStyle' if info['seis_dir'] else 'Fault'
        else:
            # Original local file handling
            root = CFG.paths.families[family]
            info = FAMILY_FILE_MAP[family]
            if not root.exists():
                raise ValueError(f"Family directory not found: {root}")
            # Vel/Style: data/model subfolders (batched)
            if info['seis_dir'] and info['vel_dir']:
                seis_files = sorted((root/info['seis_dir']).glob('*.npy'))
                vel_files = sorted((root/info['vel_dir']).glob('*.npy'))
                if seis_files and vel_files:
                    if CFG.debug_mode:
                        seis_files = seis_files[:1]
                        vel_files = vel_files[:1]
                    return seis_files, vel_files, 'VelStyle'
            # Fault: seis*.npy and vel*.npy directly in folder
            seis_files = sorted(root.glob('seis*.npy'))
            vel_files = sorted(root.glob('vel*.npy'))
            if seis_files and vel_files:
                if CFG.debug_mode:
                    seis_files = seis_files[:1]
                    vel_files = vel_files[:1]
                return seis_files, vel_files, 'Fault'
        raise ValueError(f"Could not find valid data structure for family {family}")

    def create_dataset(self, seis_files: Union[List[Path], List[str]], 
                      vel_files: Union[List[Path], List[str]], 
                      family_type: str, 
                      augment: bool = False) -> Dataset:
        """Create a dataset for the given files."""
        if family_type == 'test':
            return TestDataset(seis_files, self)
        return SeismicDataset(
            seis_files,
            vel_files,
            family_type,
            augment,
            use_mmap=self.use_mmap,
            memory_tracker=self.memory_tracker,
            data_manager=self
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
    Handles both local and S3 data sources.
    """
    def __init__(self, seis_files: Union[List[Path], List[str]], 
                 vel_files: Optional[Union[List[Path], List[str]]], 
                 family_type: str, 
                 augment: bool = False, 
                 use_mmap: bool = True, 
                 memory_tracker: MemoryTracker = None,
                 data_manager: DataManager = None):
        self.family_type = family_type
        self.augment = augment
        self.index = []
        self.use_mmap = use_mmap
        self.memory_tracker = memory_tracker
        self.vel_files = vel_files
        self.data_manager = data_manager
        
        # Build index of (file, sample_idx) pairs
        if vel_files is None:
            for sfile in seis_files:
                if self.data_manager and self.data_manager.use_s3:
                    # For S3 files, we need to get the shape without loading
                    response = self.data_manager.s3.head_object(
                        Bucket=self.data_manager.s3_bucket,
                        Key=sfile
                    )
                    # Assuming the first dimension is batch size
                    n_samples = 500  # Default for batched files
                else:
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
                if self.data_manager and self.data_manager.use_s3:
                    n_samples = 500  # Default for batched files
                else:
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
        
        # Load seismic data
        if self.data_manager and self.data_manager.use_s3:
            x = self.data_manager.stream_from_s3(sfile)
        else:
            if self.use_mmap:
                x = np.load(sfile, mmap_mode='r')
            else:
                x = np.load(sfile)
                
        # Handle data shapes
        if len(x.shape) == 4:
            x = x[i]
        if x.shape[1] == 70 and x.shape[2] == 1000:
            x = x.transpose(0, 2, 1)
            
        # Load velocity data
        if self.vel_files is None:
            y = np.zeros((1, 70, 70), np.float16)
        else:
            if self.data_manager and self.data_manager.use_s3:
                y = self.data_manager.stream_from_s3(vfile)
            else:
                if self.use_mmap:
                    y = np.load(vfile, mmap_mode='r')
                else:
                    y = np.load(vfile)
            if len(y.shape) == 4:
                y = y[i]
            elif len(y.shape) == 3:
                y = y[i]
                
        # Convert to float16 and normalize
        x = x.astype(np.float16)
        y = y.astype(np.float16)
        mu = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-6
        x = (x - mu) / std
        
        # Track memory usage
        if self.memory_tracker:
            self.memory_tracker.update(x.nbytes + y.nbytes)
            
        return torch.from_numpy(x), torch.from_numpy(y)

class TestDataset(Dataset):
    """Dataset for test files that returns the test data and its identifier."""
    def __init__(self, files: Union[List[Path], List[str]], data_manager: DataManager = None):
        self.files = files
        self.data_manager = data_manager

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        
        if self.data_manager and self.data_manager.use_s3:
            data = self.data_manager.stream_from_s3(file_path)
        else:
            data = np.load(file_path)
            
        # Convert to float16 and normalize
        data = data.astype(np.float16)
        mu = data.mean(axis=(1,2), keepdims=True)
        std = data.std(axis=(1,2), keepdims=True) + 1e-6
        data = (data - mu) / std
        
        return torch.from_numpy(data), Path(file_path).stem 


# %%
# Source: src/core/model.py
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
# Source: src/utils/update_kaggle_notebook.py
import os
import shutil
from pathlib import Path
import re
from typing import List
import logging
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.config import CFG

# Files to include in the Kaggle notebook
FILES_TO_INCLUDE = [
    'src/core/config.py',
    'src/core/preprocess.py',
    'src/core/registry.py',
    'src/core/checkpoint.py',
    'src/core/geometric_loader.py',
    'src/core/geometric_cv.py',
    'src/core/data_manager.py',
    'src/core/model.py',
    'src/utils/update_kaggle_notebook.py',
    'requirements.txt'
]

def extract_code_blocks(content: str, source_path: str = None) -> List[tuple[str, str]]:
    """Extract code blocks from Python file."""
    blocks = []
    current_block = []
    
    for line in content.split('\n'):
        if line.startswith('# %%'):
            if current_block:
                blocks.append(('\n'.join(current_block), source_path))
                current_block = []
        else:
            current_block.append(line)
    
    if current_block:
        blocks.append(('\n'.join(current_block), source_path))
    
    return blocks

def create_notebook_block(content: str, source_path: str = None) -> str:
    """Create a notebook code block."""
    if source_path:
        return f'# %%\n# Source: {source_path}\n{content}\n\n'
    return f'# %%\n{content}\n\n'

def create_notebook():
    """Create/update the Kaggle notebook Python file."""
    # Create the main notebook file
    notebook_content = []
    
    # Add header
    notebook_content.append('''"""
# Seismic Waveform Inversion - Preprocessing Pipeline

This notebook implements the preprocessing pipeline for seismic waveform inversion, including:
- Geometric-aware preprocessing with Nyquist validation
- Family-specific data loading
- Cross-validation framework
- Model registry and checkpoint management
"""

# Install dependencies
# !pip install -r requirements.txt

''')
    
    # Process each file
    for file_path in FILES_TO_INCLUDE:
        src_path = Path(file_path)
        if src_path.exists():
            with open(src_path, 'r') as f:
                content = f.read()
                blocks = extract_code_blocks(content, str(src_path))
                for block, source in blocks:
                    notebook_content.append(create_notebook_block(block, source))
        else:
            print(f"Warning: {file_path} not found")
    
    # Write to the root kaggle_notebook.py
    with open(project_root / 'kaggle_notebook.py', 'w') as f:
        f.write('\n'.join(notebook_content))
    
    print("Notebook updated successfully!")

if __name__ == "__main__":
    create_notebook() 


# %%
# Source: requirements.txt
# Core PyTorch packages
torch
torchvision
torchaudio

# Scientific computing
numpy
pandas
matplotlib
scipy

# Utilities
tqdm
pytest
psutil

# AWS and cloud storage
boto3
botocore
awscli
s3fs

# Data processing
zarr
dask
polars

# Deep learning
timm
einops
monai
pytorch-lightning==2.0.0
torchmetrics==0.11.4
segmentation-models-pytorch

# Configuration and monitoring
omegaconf
watchdog

# Google services
kagglehub
google-auth-oauthlib
google-auth-httplib2
google-api-python-client

# Additional utilities
webdataset
plotly
packaging
mlflow 

