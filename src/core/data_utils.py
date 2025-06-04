# All data IO in this file must go through DataManager (src/core/data_manager.py)
# Do NOT load data directly in this file.
#
# ## Data Loading and Preprocessing

# %%
from config import CFG
import numpy as np, torch, os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import torch.nn.functional as F
from data_manager import DataManager

def list_family_files(family: str) -> Tuple[List[Path], Optional[List[Path]]]:
    """
    Returns sorted lists (seis_files, vel_files).  If velocity files do not
    exist (test set), second element is None.
    """
    data_manager = DataManager()
    seis_files, vel_files, _ = data_manager.list_family_files(family)
    return seis_files, vel_files

def _preprocess(x: np.ndarray) -> np.ndarray:
    """Preprocess seismic data to float16 and normalize."""
    # Convert to float16
    x = x.astype(np.float16)
    
    # Normalize per-receiver
    mu = x.mean(axis=(1,2), keepdims=True)
    std = x.std(axis=(1,2), keepdims=True) + 1e-6
    x = (x - mu)/std
    
    return x

class SeismicDataset(Dataset):
    """
    Memory-efficient dataset that uses memory mapping for large files.
    Each sample:
      x : (S, T, R) float16  seismic cube
      y : (1, H, W) float16  velocity map  (None for test)
    """
    def __init__(self,
                 seis_files: List[Path],
                 vel_files: Optional[List[Path]] = None,
                 augment: bool = False,
                 use_mmap: bool = True):
        self.seis_files = seis_files
        self.vel_files = vel_files
        self.augment = augment
        self.use_mmap = use_mmap
        assert vel_files is None or len(seis_files) == len(vel_files)

    def __len__(self): 
        return len(self.seis_files)

    def __getitem__(self, idx):
        # Load seismic data
        data_manager = DataManager()
        x = data_manager.create_dataset([self.seis_files[idx]], 
                                      [self.vel_files[idx]] if self.vel_files else None,
                                      'Fault' if self.vel_files else 'Test',
                                      augment=self.augment)[0][0]
        
        if self.vel_files is None:
            y = np.zeros((1,70,70), np.float16)  # dummy
        else:
            y = data_manager.create_dataset([self.seis_files[idx]], 
                                          [self.vel_files[idx]],
                                          'Fault',
                                          augment=self.augment)[0][1]
            
        return torch.from_numpy(x), torch.from_numpy(y)

def make_loader(seis: List[Path],
                vel: Optional[List[Path]],
                batch: int,
                shuffle: bool,
                use_mmap: bool = True,
                num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader with memory-efficient settings.
    Args:
        num_workers: Number of worker processes. If 0, persistent_workers is disabled.
    """
    data_manager = DataManager()
    return data_manager.create_loader(
        seis, vel,
        'Fault' if vel else 'Test',
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers
    ) 