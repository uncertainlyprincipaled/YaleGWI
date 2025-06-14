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
from src.core.config import CFG

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

    def list_family_files(self, family: str):
        """Return (seis_files, vel_files, family_type) for a given family (base dataset only)."""
        root = CFG.paths.families[family]
        if not root.exists():
            raise ValueError(f"Family directory not found: {root}")
        # Vel/Style: data/model subfolders (batched)
        if (root / 'data').exists() and (root / 'model').exists():
            seis_files = sorted((root/'data').glob('*.npy'))
            vel_files = sorted((root/'model').glob('*.npy'))
            if seis_files and vel_files:
                family_type = 'VelStyle'
                return seis_files, vel_files, family_type
        # Fault: seis*.npy and vel*.npy directly in folder (not batched)
        seis_files = sorted(root.glob('seis*.npy'))
        vel_files = sorted(root.glob('vel*.npy'))
        if seis_files and vel_files:
            family_type = 'Fault'
            return seis_files, vel_files, family_type
        raise ValueError(f"Could not find valid data structure for family {family} at {root}")

    def create_dataset(self, seis_files, vel_files, family_type, augment=False):
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
                     distributed: bool = False) -> DataLoader:
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
        if len(x.shape) == 4:
            x = x[i]
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
        # Convert to float16
        x = x.astype(np.float16)
        y = y.astype(np.float16)
        # Normalize per-receiver
        mu = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-6
        x = (x - mu) / std
        # Ensure correct shape: (sources, receivers, timesteps)
        if len(x.shape) == 3:
            x = x[None]
        # Track memory usage
        if self.memory_tracker:
            self.memory_tracker.update(x.nbytes + y.nbytes)
        return torch.from_numpy(x), torch.from_numpy(y) 