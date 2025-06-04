"""
DataManager is the single source of truth for all data IO in this project.
All data loading, streaming, and batching must go through DataManager.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from config import CFG
from torch.utils.data import DistributedSampler

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
    DataManager is the single source of truth for all data IO in this project.
    Handles memory-efficient, sample-wise access for all dataset families.
    Uses float16 for memory efficiency.
    """
    def __init__(self, use_mmap: bool = True):
        self.use_mmap = use_mmap
        self.memory_tracker = MemoryTracker()

    def list_family_files(self, family: str) -> Tuple[List[Path], List[Path], str]:
        """Return (seis_files, vel_files, family_type) for a given family."""
        root = CFG.paths.families[family]
        if (root / 'data').exists():  # Vel/Style
            seis_files = sorted((root/'data').glob('*.npy'))
            vel_files = sorted((root/'model').glob('*.npy'))
            family_type = 'VelStyle'
        else:  # Fault
            seis_files = sorted(root.glob('seis*_*_*.npy'))
            vel_files = sorted(root.glob('vel*_*_*.npy'))
            family_type = 'Fault'
        return seis_files, vel_files, family_type

    def create_dataset(self, seis_files: List[Path], vel_files: List[Path], 
                      family_type: str, augment: bool = False) -> Dataset:
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
    For Vel/Style: sample-wise mmap access from large files.
    For Fault: one sample per file.
    Uses float16 for memory efficiency.
    """
    def __init__(self, seis_files: List[Path], vel_files: List[Path], family_type: str, 
                 augment: bool = False, use_mmap: bool = True, memory_tracker: MemoryTracker = None):
        self.family_type = family_type
        self.augment = augment
        self.index = []
        self.use_mmap = use_mmap
        self.memory_tracker = memory_tracker
        if family_type == 'VelStyle':
            for sfile, vfile in zip(seis_files, vel_files):
                # Each file contains 500 samples
                for i in range(500):
                    self.index.append((sfile, vfile, i))
        elif family_type == 'Fault':
            for sfile, vfile in zip(seis_files, vel_files):
                self.index.append((sfile, vfile, None))
        else:
            raise ValueError(f"Unknown family_type: {family_type}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sfile, vfile, i = self.index[idx]
        if self.family_type == 'VelStyle':
            x = np.load(sfile, mmap_mode='r' if self.use_mmap else None)[i]
            y = np.load(vfile, mmap_mode='r' if self.use_mmap else None)[i]
        else:  # Fault
            x = np.load(sfile)
            y = np.load(vfile)
            
        # Convert to float16 for memory efficiency
        x = x.astype(np.float16)
        y = y.astype(np.float16)
            
        # Normalize per-receiver
        mu = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-6
        x = (x - mu) / std
        
        # Track memory usage
        if self.memory_tracker:
            self.memory_tracker.update(x.nbytes + y.nbytes)
            
        return torch.from_numpy(x), torch.from_numpy(y) 