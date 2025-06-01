# %% [markdown]
# ## Data Loading and Preprocessing

# %%
from config import CFG
import numpy as np, torch, os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Tuple, Optional

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

class SeismicDataset(Dataset):
    """
    Each sample:
      x : (S, T, R) float32  seismic cube
      y : (1, H, W) float32  velocity map  (None for test)
    """
    def __init__(self,
                 seis_files: List[Path],
                 vel_files: Optional[List[Path]] = None,
                 augment: bool=False):
        self.seis_files = seis_files
        self.vel_files  = vel_files
        self.augment    = augment
        assert vel_files is None or len(seis_files) == len(vel_files)

    def __len__(self): return len(self.seis_files)

    def _load(self, f: Path):  # mem-mapped for RAM-saving
        return np.load(f, mmap_mode='r')

    def __getitem__(self, idx):
        x = self._load(self.seis_files[idx]).astype(np.float32)
        # normalise per-receiver
        mu  = x.mean(axis=(1,2), keepdims=True)
        std = x.std (axis=(1,2), keepdims=True) + 1e-6
        x   = (x - mu)/std
        if self.vel_files is None:
            y = np.zeros((1,70,70), np.float32)  # dummy
        else:
            y = self._load(self.vel_files[idx]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

def make_loader(seis: List[Path],
                vel : Optional[List[Path]],
                batch: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(
        SeismicDataset(seis, vel),
        batch_size     = batch,
        shuffle        = shuffle,
        num_workers    = min(4, os.cpu_count()),
        pin_memory     = True,
        persistent_workers = True,
    ) 