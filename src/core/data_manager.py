from pathlib import Path
from typing import List, Optional, Tuple, Iterator, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import polars as pl
from config import CFG
import os

class DataManager:
    """
    Unified interface for data operations. Handles:
    1. Memory-efficient data loading
    2. Streaming for large datasets
    3. Caching and memory mapping
    4. Environment-specific optimizations
    """
    def __init__(self, use_mmap: bool = True, cache_size: int = 1000):
        self.use_mmap = use_mmap
        self.cache_size = cache_size
        self._file_cache: Dict[Path, np.ndarray] = {}
        
    def _load_file(self, path: Path) -> np.ndarray:
        """Load a file with caching and memory mapping."""
        if path in self._file_cache:
            return self._file_cache[path]
            
        if self.use_mmap:
            data = np.load(path, mmap_mode='r')
        else:
            data = np.load(path)
            
        # Cache the file if it's small enough
        if data.nbytes < self.cache_size * 1024 * 1024:  # Convert MB to bytes
            self._file_cache[path] = data
            
        return data
        
    def list_family_files(self, family: str) -> Tuple[List[Path], Optional[List[Path]]]:
        """Get seismic and velocity files for a family."""
        root = CFG.paths.families[family]
        if (root / 'data').exists():  # FlatVel / CurveVel / Style
            seis = sorted((root/'data').glob('*.npy'))
            vel = sorted((root/'model').glob('*.npy')) if (root/'model').exists() else None
        else:  # *Fault* families
            seis = sorted(root.glob('seis*_*_*.npy'))
            vel = sorted(root.glob('vel*_*_*.npy')) if (root/'vel2_1_0.npy').exists() else None
        return seis, vel
        
    def stream_sequences(self, batch_size: int = 16, subset: str = "train") -> Iterator[torch.Tensor]:
        """Stream sequences using Polars for memory efficiency."""
        data_path = CFG.paths.root / f"{subset}.csv"
        scan = (pl.scan_csv(data_path)
                .with_row_index("row_id")
                .group_by("sequence_id")
                .agg(pl.all()))
        
        for df in scan.collect(streaming=True).iter_slices(n_rows=batch_size):
            yield torch.from_numpy(df.to_numpy())
            
    def create_dataset(self, 
                      seis_files: List[Path],
                      vel_files: Optional[List[Path]] = None,
                      augment: bool = False) -> Dataset:
        """Create a memory-efficient dataset."""
        return SeismicDataset(
            seis_files=seis_files,
            vel_files=vel_files,
            augment=augment,
            use_mmap=self.use_mmap
        )
        
    def create_loader(self,
                     seis_files: List[Path],
                     vel_files: Optional[List[Path]] = None,
                     batch_size: int = 32,
                     shuffle: bool = True) -> DataLoader:
        """Create a DataLoader with optimized settings."""
        dataset = self.create_dataset(seis_files, vel_files)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2 if os.cpu_count() > 1 else None
        )
        
    def get_test_files(self) -> List[Path]:
        """Get test files."""
        return sorted(CFG.paths.test.glob('*.npy'))
        
    def clear_cache(self):
        """Clear the file cache."""
        self._file_cache.clear()

class SeismicDataset(Dataset):
    """
    Memory-efficient dataset that uses memory mapping for large files.
    Each sample:
      x : (S, T, R) float32  seismic cube
      y : (1, H, W) float32  velocity map  (None for test)
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
        
        # Pre-load file sizes for memory mapping
        self.seis_sizes = [f.stat().st_size for f in seis_files]
        if vel_files:
            self.vel_sizes = [f.stat().st_size for f in vel_files]

    def __len__(self): 
        return len(self.seis_files)

    def _load(self, f: Path, size: int) -> np.ndarray:
        if self.use_mmap:
            return np.load(f, mmap_mode='r')
        else:
            return np.load(f)

    def __getitem__(self, idx):
        # Load seismic data with memory mapping
        x = self._load(self.seis_files[idx], self.seis_sizes[idx]).astype(np.float32)
        
        # Normalize per-receiver
        mu = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-6
        x = (x - mu)/std
        
        if self.vel_files is None:
            y = np.zeros((1,70,70), np.float32)  # dummy
        else:
            y = self._load(self.vel_files[idx], self.vel_sizes[idx]).astype(np.float32)
            
        return torch.from_numpy(x), torch.from_numpy(y) 