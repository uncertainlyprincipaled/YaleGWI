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
            
        # Handle S3 data if in AWS environment
        if CFG.env.kind == 'aws':
            try:
                # List objects in S3
                s3_objects = self.s3_loader.s3.list_objects_v2(
                    Bucket=CFG.env.s3_bucket,
                    Prefix=f"raw/{family}/"
                )
                
                # Filter and sort files
                seis_files = []
                vel_files = []
                for obj in s3_objects.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.npy'):
                        if 'seis' in key:
                            seis_files.append(key)
                        elif 'vel' in key:
                            vel_files.append(key)
                            
                # Download files to local cache
                local_seis_files = []
                local_vel_files = []
                
                for s3_key in sorted(seis_files):
                    local_path = root / Path(s3_key).name
                    local_seis_files.append(self.s3_loader.download_file(s3_key, local_path))
                    
                for s3_key in sorted(vel_files):
                    local_path = root / Path(s3_key).name
                    local_vel_files.append(self.s3_loader.download_file(s3_key, local_path))
                    
                if local_seis_files and local_vel_files:
                    return local_seis_files, local_vel_files, 'Fault'
                    
            except ClientError as e:
                logging.error(f"Failed to list S3 objects: {e}")
                raise
                
        # Fallback to local filesystem
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
    """Memory-efficient dataset for seismic data using memory mapping."""
    def __init__(self, seis_files: List[Path], vel_files: List[Path], family_type: str, 
                 augment: bool = False, use_mmap: bool = True, memory_tracker: MemoryTracker = None):
        self.seis_files = seis_files
        self.vel_files = vel_files
        self.family_type = family_type
        self.augment = augment
        self.use_mmap = use_mmap
        self.memory_tracker = memory_tracker or MemoryTracker()
        
        # Pre-load file sizes for memory tracking
        self.file_sizes = {}
        for f in seis_files + vel_files:
            self.file_sizes[f] = f.stat().st_size
            
    def __len__(self):
        return len(self.seis_files)
        
    def __getitem__(self, idx):
        seis_file = self.seis_files[idx]
        vel_file = self.vel_files[idx]
        
        # Track memory usage
        if self.memory_tracker:
            self.memory_tracker.update(self.file_sizes[seis_file] + self.file_sizes[vel_file])
            
        # Load data with memory mapping
        if self.use_mmap:
            seis = np.load(seis_file, mmap_mode='r')
            vel = np.load(vel_file, mmap_mode='r')
            
            # Convert to torch tensors with memory efficiency
            seis_tensor = torch.from_numpy(seis).float()
            vel_tensor = torch.from_numpy(vel).float()
            
            # Clear memory mapping
            del seis, vel
            
            # Reduce vel to match model output
            vel_tensor = vel_tensor.mean(dim=1, keepdim=True)
            
        else:
            seis_tensor = torch.from_numpy(np.load(seis_file)).float()
            vel_tensor = torch.from_numpy(np.load(vel_file)).float()
            
            # Reduce vel to match model output
            vel_tensor = vel_tensor.mean(dim=1, keepdim=True)
            
        return seis_tensor, vel_tensor 