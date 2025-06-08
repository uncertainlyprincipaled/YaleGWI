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
        if not root.exists():
            raise ValueError(f"Family directory not found: {root}")
            
        # First try YaleGWI-style structure (data/model subdirectories)
        if (root / 'data').exists() and (root / 'model').exists():
            seis_files = sorted((root/'data').glob('*.npy'))
            vel_files = sorted((root/'model').glob('*.npy'))
            if seis_files and vel_files:
                family_type = 'VelStyle'
                return seis_files, vel_files, family_type
                
        # Then try OpenFWI-style structure (flat directory with seis/vel files)
        seis_files = sorted(root.glob('seis*.npy'))
        vel_files = sorted(root.glob('vel*.npy'))
        if seis_files and vel_files:
            # Pair by filename (replace 'seis' with 'vel')
            paired_seis = []
            paired_vel = []
            for sfile in seis_files:
                vfile = root / sfile.name.replace('seis', 'vel')
                if vfile.exists():
                    paired_seis.append(sfile)
                    paired_vel.append(vfile)
                else:
                    print(f"[Warning] No matching velocity file for {sfile}")
            if paired_seis and paired_vel:
                family_type = 'PerSample'
                return paired_seis, paired_vel, family_type
                
        # If we get here, we couldn't find a valid data structure
        raise ValueError(f"Could not find valid data structure for family {family} at {root}. "
                       f"Directory exists: {root.exists()}, "
                       f"Contains data/model: {(root/'data').exists() and (root/'model').exists()}, "
                       f"Contains seis/vel files: {bool(seis_files) and bool(vel_files)}")

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

    def get_balanced_family_files(self, target_count=1000):
        """
        For each family, return (seis_files, vel_files, family_type) with up to target_count samples.
        Supplement Fault families with OpenFWI if needed, and downsample large families if needed.
        """
        from config import CFG
        import numpy as np
        from pathlib import Path
        openfwi_path = Path('/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72')
        families = list(CFG.paths.families.keys())
        balanced = {}
        for family in families:
            # Get base files
            base_seis, base_vel, family_type = self.list_family_files(family)
            base_count = len(base_seis)
            # Get OpenFWI files if available
            openfwi_seis, openfwi_vel = [], []
            if openfwi_path.exists() and (openfwi_path / family).exists():
                openfwi_family = openfwi_path / family
                openfwi_seis = sorted(openfwi_family.glob('seis*.npy'))
                openfwi_vel = [openfwi_family / f.name.replace('seis', 'vel') for f in openfwi_seis if (openfwi_family / f.name.replace('seis', 'vel')).exists()]
            # Combine
            all_seis = base_seis + openfwi_seis
            all_vel = base_vel + openfwi_vel
            # Pair up to min length
            min_len = min(len(all_seis), len(all_vel))
            all_seis = all_seis[:min_len]
            all_vel = all_vel[:min_len]
            # Shuffle
            idx = np.arange(len(all_seis))
            np.random.shuffle(idx)
            all_seis = [all_seis[i] for i in idx]
            all_vel = [all_vel[i] for i in idx]
            # Subsample or pad
            if len(all_seis) >= target_count:
                final_seis = all_seis[:target_count]
                final_vel = all_vel[:target_count]
            else:
                final_seis = all_seis
                final_vel = all_vel
            balanced[family] = (final_seis, final_vel, family_type)
        return balanced

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
        
        # Add source dimension if not present
        if len(x.shape) == 3:  # (T,R) -> (1,T,R)
            x = x[None]
        
        # Track memory usage
        if self.memory_tracker:
            self.memory_tracker.update(x.nbytes + y.nbytes)
            
        return torch.from_numpy(x), torch.from_numpy(y) 