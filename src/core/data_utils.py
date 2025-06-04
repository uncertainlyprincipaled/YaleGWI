# %% [markdown]
# ## Data Loading and Preprocessing

# %%
from config import CFG
import numpy as np, torch, os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import torch.nn.functional as F

def list_family_files(family: str) -> Tuple[List[Path], Optional[List[Path]]]:
    """
    Returns sorted lists (seis_files, vel_files).  If velocity files do not
    exist (test set), second element is None.
    """
    root = CFG.paths.families[family]
    if (root / 'data').exists():                 # FlatVel / CurveVel / Style
        seis = sorted((root/'data').glob('*.npy'))
        vel  = sorted((root/'model').glob('*.npy')) if (root/'model').exists() else None
    else:                                        # *Fault* families
        seis = sorted(root.glob('seis*_*_*.npy'))
        vel  = sorted(root.glob('vel*_*_*.npy')) if (root/'vel2_1_0.npy').exists() else None
    return seis, vel

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

    def _load(self, f: Path) -> np.ndarray:
        """Load data with memory mapping if enabled."""
        if self.use_mmap:
            return np.load(f, mmap_mode='r')
        return np.load(f)

    def __getitem__(self, idx):
        # Load seismic data
        x = self._load(self.seis_files[idx])
        x = _preprocess(x)
        
        if self.vel_files is None:
            y = np.zeros((1,70,70), np.float16)  # dummy
        else:
            y = self._load(self.vel_files[idx]).astype(np.float16)
            
        # Apply temporal flip augmentation if enabled
        if self.augment and np.random.random() < 0.5:
            x = x[::-1, :, ::-1]  # Flip time and receiver dimensions
            y = y[..., ::-1]      # Flip width dimension of velocity map
            
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
    return DataLoader(
        SeismicDataset(seis, vel, augment=shuffle, use_mmap=use_mmap),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 1 else None
    ) 