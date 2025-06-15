f# %%
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
# Source: src/core/preprocess.py
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.signal import decimate
import logging
from typing import Tuple, List, Optional
import warnings
from src.core.config import CFG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for preprocessing
CHUNK_TIME = 256  # After decimating by 4
CHUNK_SRC_REC = 8
DT_DECIMATE = 4  # 1 kHz → 250 Hz
NYQUIST_FREQ = 500  # Hz (half of original sampling rate)

def validate_nyquist(data: np.ndarray, original_fs: int = 1000) -> bool:
    """
    Validate that the data satisfies Nyquist criterion after downsampling.
    
    Args:
        data: Input seismic data array
        original_fs: Original sampling frequency in Hz
        
    Returns:
        bool: True if data satisfies Nyquist criterion
    """
    # Compute FFT
    fft_data = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=1/original_fs)
    
    # Check if significant energy exists above Nyquist frequency
    nyquist_mask = freqs > (original_fs / (2 * DT_DECIMATE))
    high_freq_energy = np.abs(fft_data[:, nyquist_mask]).mean()
    total_energy = np.abs(fft_data).mean()
    
    # If more than 1% of energy is above Nyquist, warn
    if high_freq_energy / total_energy > 0.01:
        warnings.warn(f"Significant energy above Nyquist frequency detected: {high_freq_energy/total_energy:.2%}")
        return False
    return True

def preprocess_one(arr: np.ndarray) -> np.ndarray:
    """
    Preprocess a single seismic array with downsampling and normalization.
    
    Args:
        arr: Input seismic array
        
    Returns:
        np.ndarray: Preprocessed array
    """
    try:
        # Validate Nyquist criterion
        if not validate_nyquist(arr):
            logger.warning("Data may violate Nyquist criterion after downsampling")
        
        # Decimate time axis with anti-aliasing filter
        arr = decimate(arr, DT_DECIMATE, axis=1, ftype='fir')
        
        # Convert to float16
        arr = arr.astype('float16')
        
        # Robust normalization per trace
        μ = np.median(arr, keepdims=True)
        σ = np.percentile(arr, 95, keepdims=True) - np.percentile(arr, 5, keepdims=True)
        arr = (arr - μ) / (σ + 1e-8)  # Add small epsilon to avoid division by zero
        
        return arr
    except Exception as e:
        logger.error(f"Error preprocessing array: {str(e)}")
        raise

def process_family(family: str, input_dir: Path, output_dir: Path) -> List[str]:
    """
    Process all files in a family and return paths to processed files.
    
    Args:
        family: Name of the geological family
        input_dir: Input directory containing raw data
        output_dir: Output directory for processed data
        
    Returns:
        List[str]: Paths to processed files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_dir.glob('*.npy'))
    processed_paths = []
    
    for f in tqdm(files, desc=f"Processing {family}"):
        try:
            arr = np.load(f, mmap_mode='r')
            if arr.ndim == 4:
                n_samples = arr.shape[0]
                for i in range(n_samples):
                    sample = arr[i]
                    processed = preprocess_one(sample)
                    out_path = output_dir / f"sample_{len(processed_paths):06d}.npy"
                    np.save(out_path, processed)
                    processed_paths.append(str(out_path))
            elif arr.ndim == 3:
                processed = preprocess_one(arr)
                out_path = output_dir / f"sample_{len(processed_paths):06d}.npy"
                np.save(out_path, processed)
                processed_paths.append(str(out_path))
            else:
                logger.warning(f"Unexpected shape {arr.shape} in {f}")
        except Exception as e:
            logger.error(f"Error processing file {f}: {str(e)}")
            continue
    
    return processed_paths

def create_zarr_dataset(processed_paths: List[str], output_path: Path, chunk_size: Tuple[int, ...]) -> None:
    """
    Create a zarr dataset from processed files.
    
    Args:
        processed_paths: List of paths to processed files
        output_path: Path to save zarr dataset
        chunk_size: Chunk size for zarr array
    """
    try:
        # Create lazy Dask arrays
        lazy_arrays = []
        for path in processed_paths:
            x = da.from_delayed(
                dask.delayed(np.load)(path),
                shape=(32, 256, 64),  # Example dims after decimation
                dtype='float16'
            )
            lazy_arrays.append(x)
        
        # Stack arrays
        stack = da.stack(lazy_arrays, axis=0)
        
        # Save to zarr with compression
        stack.to_zarr(
            output_path,
            component='seis',
            compressor=zarr.Blosc(cname='zstd', clevel=3),
            chunks=chunk_size
        )
    except Exception as e:
        logger.error(f"Error creating zarr dataset: {str(e)}")
        raise

def split_for_gpus(processed_paths: List[str], output_base: Path) -> None:
    """
    Split processed files into two datasets for the two T4 GPUs.
    
    Args:
        processed_paths: List of paths to processed files
        output_base: Base directory for output
    """
    try:
        n_samples = len(processed_paths)
        mid_point = n_samples // 2
        
        # Create GPU-specific directories
        gpu0_dir = output_base / 'gpu0'
        gpu1_dir = output_base / 'gpu1'
        gpu0_dir.mkdir(parents=True, exist_ok=True)
        gpu1_dir.mkdir(parents=True, exist_ok=True)
        
        # Split paths
        gpu0_paths = processed_paths[:mid_point]
        gpu1_paths = processed_paths[mid_point:]
        
        # Create zarr datasets for each GPU
        create_zarr_dataset(
            gpu0_paths,
            gpu0_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC)
        )
        create_zarr_dataset(
            gpu1_paths,
            gpu1_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC)
        )
        
        logger.info(f"Created GPU datasets with {len(gpu0_paths)} and {len(gpu1_paths)} samples")
    except Exception as e:
        logger.error(f"Error splitting data for GPUs: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Preprocess seismic data for distributed training on T4 GPUs")
    parser.add_argument('--input_root', type=str, default=str(CFG.paths.train), help='Input train_samples root directory')
    parser.add_argument('--output_root', type=str, default='/kaggle/working/preprocessed', help='Output directory for processed files')
    args = parser.parse_args()

    try:
        input_root = Path(args.input_root)
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        # Process each family
        families = list(CFG.paths.families.keys())
        all_processed_paths = []
        
        for family in families:
            logger.info(f"\nProcessing family: {family}")
            input_dir = input_root / family
            temp_dir = output_root / 'temp' / family
            processed_paths = process_family(family, input_dir, temp_dir)
            all_processed_paths.extend(processed_paths)
            logger.info(f"Family {family}: {len(processed_paths)} samples processed")

        # Split and create zarr datasets for GPUs
        logger.info("\nCreating GPU-specific datasets...")
        split_for_gpus(all_processed_paths, output_root)
        
        # Clean up temporary files
        temp_dir = output_root / 'temp'
        if temp_dir.exists():
            subprocess.run(['rm', '-rf', str(temp_dir)])
        
        logger.info("\nPreprocessing complete!")
        logger.info(f"GPU 0 dataset: {output_root}/gpu0/seismic.zarr")
        logger.info(f"GPU 1 dataset: {output_root}/gpu1/seismic.zarr")
    except Exception as e:
        logger.error(f"Error in main preprocessing: {str(e)}")
        raise

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
    Registry for managing model versions with geometric metadata.
    Tracks model versions, preserves equivariance properties, and handles initialization.
    """
    
    def __init__(self, registry_dir: str = "models"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store model registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
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
    Manager for saving and loading model checkpoints with geometric metadata.
    Handles checkpoint versioning, geometric properties, and coordinate transformations.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        
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
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
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
    Data loader for handling family-specific data loading with geometric features.
    Manages data loading for different geological families with proper batching.
    """
    
    def __init__(self,
                 data_root: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 transform: Optional[Any] = None,
                 extract_features: bool = True):
        """
        Initialize the data loader.
        
        Args:
            data_root: Root directory containing family datasets
            batch_size: Batch size for loading
            num_workers: Number of worker processes
            transform: Optional data transformations
            extract_features: Whether to extract geometric features
        """
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.extract_features = extract_features
        
        # Initialize datasets for each family
        self.datasets = {}
        self.loaders = {}
        
        # Find all family directories
        family_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        
        for family_dir in family_dirs:
            family = family_dir.name
            dataset = GeometricDataset(
                family_dir,
                family,
                transform=transform,
                extract_features=extract_features
            )
            self.datasets[family] = dataset
            
            # Create data loader
            self.loaders[family] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
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
import json

logger = logging.getLogger(__name__)

class GeometricCrossValidator:
    """
    Cross-validation framework with geometric awareness.
    Implements stratified sampling based on geological families and geometric features.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize the cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
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
# Source: src/core/train.py
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

