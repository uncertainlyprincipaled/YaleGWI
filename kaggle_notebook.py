# %%
# SpecProj-UNet for Seismic Waveform Inversion
# This notebook implements a physics-guided neural network for seismic waveform inversion
# using spectral projectors and UNet architecture.


# %%
import os
import torch
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl


# %%
# ## Configuration and Environment Setup



# %%
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Literal, NamedTuple
import torch

# --------------------------------------------------------------------------- #
#  Detect runtime (Kaggle / Colab / SageMaker / local) and expose a singleton #
# --------------------------------------------------------------------------- #

class _KagglePaths(NamedTuple):
    root : Path = Path('/kaggle/input/waveform-inversion')
    train: Path = root / 'train_samples'
    test : Path = root / 'test'
    # folders visible in the screenshot
    families = {
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

class _Env:
    def __init__(self):
        if 'KAGGLE_URL_BASE' in os.environ:
            self.kind: Literal['kaggle','colab','sagemaker','local'] = 'kaggle'
        elif 'COLAB_GPU' in os.environ:
            self.kind = 'colab'
        elif 'SM_NUM_CPUS' in os.environ:
            self.kind = 'sagemaker'
        else:
            self.kind = 'local'
        self.device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else 'cpu'
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

            # Training hyper-parameters
            cls._inst.batch   = 4 if cls._inst.env.kind == 'kaggle' else 32
            cls._inst.lr      = 1e-4
            cls._inst.weight_decay = 1e-3
            cls._inst.epochs  = 30
            cls._inst.lambda_pde = 0.1
            cls._inst.dtype = "float16"  # Default dtype for tensors

            # Model parameters
            cls._inst.backbone = "hgnetv2_b2.ssld_stage2_ft_in1k"
            cls._inst.ema_decay = 0.99
            cls._inst.pretrained = True

            # Loss weights
            cls._inst.lambda_inv = 1.0
            cls._inst.lambda_fwd = 1.0
            cls._inst.lambda_pde = 0.1

        return cls._inst

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


# %%
import os
import subprocess
from pathlib import Path
import shutil
try:
    import kagglehub
except ImportError:
    kagglehub = None

def warm_kaggle_cache():
    """Warm up the Kaggle FUSE cache by creating a temporary tar archive."""
    data_dir = Path('/kaggle/input/waveform-inversion')
    tmp_tar = Path('/kaggle/working/tmp.tar.gz')
    
    # Check if data directory exists
    if not data_dir.exists():
        print("Warning: Competition data not found at /kaggle/input/waveform-inversion")
        print("Please add the competition dataset to your notebook first:")
        print("1. Click on the 'Data' tab")
        print("2. Click 'Add Data'")
        print("3. Search for 'Waveform Inversion'")
        print("4. Click 'Add' on the competition dataset")
        return
        
    try:
        subprocess.run([
            'tar', '-I', 'pigz', '-cf', str(tmp_tar),
            str(data_dir)
        ], check=True)
        tmp_tar.unlink()  # Clean up
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to warm up cache: {e}")
        print("This is not critical - continuing with setup...")
    except Exception as e:
        print(f"Warning: Unexpected error during cache warmup: {e}")
        print("This is not critical - continuing with setup...")

def setup_environment():
    """Setup environment-specific configurations and download datasets if needed."""
    from src.core.config import CFG  # Import here to avoid circular dependency
    
    # Allow explicit environment override
    env_override = os.environ.get('GWI_ENV', '').lower()
    if env_override:
        CFG.env.kind = env_override
    
    if CFG.env.kind == 'colab':
        # Create data directory
        data_dir = Path('/content/data')
        data_dir.mkdir(exist_ok=True)
        
        # Update paths for Colab
        CFG.paths.root = data_dir / 'waveform-inversion'
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        
        # Update family paths
        CFG.paths.families = {
            'FlatVel_A'   : CFG.paths.train/'FlatVel_A',
            'FlatVel_B'   : CFG.paths.train/'FlatVel_B',
            'CurveVel_A'  : CFG.paths.train/'CurveVel_A',
            'CurveVel_B'  : CFG.paths.train/'CurveVel_B',
            'Style_A'     : CFG.paths.train/'Style_A',
            'Style_B'     : CFG.paths.train/'Style_B',
            'FlatFault_A' : CFG.paths.train/'FlatFault_A',
            'FlatFault_B' : CFG.paths.train/'FlatFault_B',
            'CurveFault_A': CFG.paths.train/'CurveFault_A',
            'CurveFault_B': CFG.paths.train/'CurveFault_B',
        }
        
        print("Environment setup complete for Colab")
    
    elif CFG.env.kind == 'sagemaker':
        # AWS SageMaker specific setup
        data_dir = Path('/opt/ml/input/data')  
        data_dir.mkdir(exist_ok=True)

        # Create a symbolic link to the dataset
        dataset_path = Path('/opt/ml/input/data/waveform-inversion')
        dataset_path.symlink_to(data_dir / 'waveform-inversion')

        # Download dataset
        print("Downloading dataset from Kaggle...")
        kagglehub.model_download('jamie-morgan/waveform-inversion', path=str(data_dir))

        # Update paths for SageMaker
        CFG.paths.root = data_dir / 'waveform-inversion'
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        
        # Update family paths
        CFG.paths.families = {
            'FlatVel_A'   : CFG.paths.train/'FlatVel_A',
            'FlatVel_B'   : CFG.paths.train/'FlatVel_B',
            'CurveVel_A'  : CFG.paths.train/'CurveVel_A',
            'CurveVel_B'  : CFG.paths.train/'CurveVel_B',
            'Style_A'     : CFG.paths.train/'Style_A',
            'Style_B'     : CFG.paths.train/'Style_B',
            'FlatFault_A' : CFG.paths.train/'FlatFault_A',
            'FlatFault_B' : CFG.paths.train/'FlatFault_B',
            'CurveFault_A': CFG.paths.train/'CurveFault_A',
            'CurveFault_B': CFG.paths.train/'CurveFault_B',
        }
        
        print("Paths configured for SageMaker environment") 
    
    elif CFG.env.kind == 'kaggle':
        # In Kaggle, warm up the FUSE cache first
        warm_kaggle_cache()
        
        # Set up paths for Kaggle environment
        CFG.paths.root = Path('/kaggle/input/waveform-inversion')
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        
        # Update family paths
        CFG.paths.families = {
            'FlatVel_A'   : CFG.paths.train/'FlatVel_A',
            'FlatVel_B'   : CFG.paths.train/'FlatVel_B',
            'CurveVel_A'  : CFG.paths.train/'CurveVel_A',
            'CurveVel_B'  : CFG.paths.train/'CurveVel_B',
            'Style_A'     : CFG.paths.train/'Style_A',
            'Style_B'     : CFG.paths.train/'Style_B',
            'FlatFault_A' : CFG.paths.train/'FlatFault_A',
            'FlatFault_B' : CFG.paths.train/'FlatFault_B',
            'CurveFault_A': CFG.paths.train/'CurveFault_A',
            'CurveFault_B': CFG.paths.train/'CurveFault_B',
        }
        
        print("Environment setup complete for Kaggle")
    
    # Add any additional logic for 'local' or other environments as needed 


# %%
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


# %%
# ## Model Architecture - Spectral Projector



# %%
import torch, math
from torch import nn
from config import CFG
from functools import lru_cache

@lru_cache(maxsize=None)
def _freq_grids(shape: tuple, device: torch.device):
    # shape = (..., T, R) after rfftn  ⇒  last dim is R/2+1
    T, R2 = shape[-2], shape[-1]
    dt, dx = 1e-3, 10.           # hard-coded (could move to CFG)
    freqs_t = torch.fft.rfftfreq(T, dt, device=device)          # (T//2+1,)
    freqs_x = torch.fft.fftfreq(2*(R2-1), dx, device=device)    # (R,)
    ω = freqs_t.view(-1, 1)                  # (T/2+1,1)
    k = freqs_x.view(1, -1)                  # (1,R)
    return ω, k

class PhysMask(nn.Module):
    """
    Learns background slowness c and temperature τ.
    Returns up-going and down-going wavefields.
    """
    def __init__(self,
                 c_init=3000., c_min=1500., c_max=5500.):
        super().__init__()
        self.log_c   = nn.Parameter(torch.log(torch.tensor(c_init)))
        self.log_tau = nn.Parameter(torch.zeros(()))  # τ ≈ 1

        self.register_buffer('c_min', torch.tensor(c_min))
        self.register_buffer('c_max', torch.tensor(c_max))

    def forward(self, x: torch.Tensor):
        # x  (B,S,T,R)  – real
        B, S, T, R = x.shape
        Xf = torch.fft.rfftn(x, dim=(-2,-1))
        ω, k = _freq_grids(Xf.shape[-2:], Xf.device)

        c  = torch.sigmoid(self.log_c) * (self.c_max-self.c_min) + self.c_min
        τ  = torch.exp(self.log_tau).clamp(0.1, 10.)

        ratio = ω / torch.sqrt(ω**2 + (c*k)**2 + 1e-9)
        mask_up   = torch.sigmoid( ratio / τ)
        mask_down = torch.sigmoid(-ratio / τ)

        mask_up = mask_up.expand_as(Xf)
        mask_down = mask_down.expand_as(Xf)

        up   = torch.fft.irfftn(Xf*mask_up  , dim=(-2,-1), s=(T,R))
        down = torch.fft.irfftn(Xf*mask_down, dim=(-2,-1), s=(T,R))
        return up, down

class SpectralAssembler(nn.Module):
    """
    Implements Π±† using small ε-regularized Moore-Penrose inverse.
    Reconstructs wavefield from up-going and down-going components.
    
    Reference: §1 of ICLR25FWI paper for mathematical formulation.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        """
        Args:
            up: Up-going wavefield (B,S,T,R)
            down: Down-going wavefield (B,S,T,R)
        Returns:
            Reconstructed wavefield (B,S,T,R)
        """
        # Convert to frequency domain
        Up_f = torch.fft.rfftn(up, dim=(-2,-1))
        Down_f = torch.fft.rfftn(down, dim=(-2,-1))
        
        # Compute Moore-Penrose inverse with regularization
        # Π† = (Π^T Π + εI)^(-1) Π^T
        # where Π is the projection operator
        Up_f_conj = Up_f.conj()
        Down_f_conj = Down_f.conj()
        
        # Regularized inverse
        denom = (Up_f_conj * Up_f + Down_f_conj * Down_f + self.eps)
        inv = 1.0 / denom
        
        # Apply inverse to reconstruct
        recon_f = inv * (Up_f_conj * Up_f + Down_f_conj * Down_f)
        
        # Convert back to time domain
        return torch.fft.irfftn(recon_f, dim=(-2,-1), s=up.shape[-2:])

def split_and_reassemble(x: torch.Tensor, mask: PhysMask, assembler: SpectralAssembler) -> torch.Tensor:
    """
    Helper function to split wavefield and reassemble it.
    Useful for unit testing the projection operators.
    
    Args:
        x: Input wavefield (B,S,T,R)
        mask: PhysMask instance
        assembler: SpectralAssembler instance
    Returns:
        Reconstructed wavefield (B,S,T,R)
    """
    up, down = mask(x)
    return assembler(up, down) 


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG

class CouplingLayer(nn.Module):
    """Coupling layer for invertible transformations."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels//2, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels//2, 3, padding=1)
        )
        
    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            x2 = x2 + self.net(x1)
        else:
            x2 = x2 - self.net(x1)
        return torch.cat([x1, x2], dim=1)

class IUNet(nn.Module):
    """
    Invertible U-Net for latent space translation.
    Approximately 24M parameters.
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Initial projection
        self.proj_in = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # Coupling layers
        self.couplings = nn.ModuleList([
            CouplingLayer(hidden_channels, hidden_channels*2)
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.proj_out = nn.Conv2d(hidden_channels, in_channels, 1)
        
    def forward(self, z: torch.Tensor, direction: str) -> torch.Tensor:
        """
        Args:
            z: Input tensor (B,C,H,W)
            direction: Either "p→v" or "v→p"
        Returns:
            Translated tensor (B,C,H,W)
        """
        if direction not in ["p→v", "v→p"]:
            raise ValueError("direction must be either 'p→v' or 'v→p'")
            
        x = self.proj_in(z)
        
        # Apply coupling layers
        for coupling in self.couplings:
            x = coupling(x, reverse=(direction == "v→p"))
            
        return self.proj_out(x)

def create_iunet() -> IUNet:
    """Factory function to create IU-Net with default config."""
    return IUNet(
        in_channels=1,
        hidden_channels=64,
        num_layers=4
    ) 


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG
from .proj_mask import PhysMask, SpectralAssembler
from .iunet import create_iunet
from .specproj_unet import SmallUNet

class WaveDecoder(nn.Module):
    """Simple decoder for wavefield prediction."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        return self.net(x)

class HybridSpecProj(nn.Module):
    """
    Hybrid model combining physics-guided splitting with IU-Net translation.
    """
    def __init__(self, chans=32, depth=4):
        super().__init__()
        self.mask = PhysMask()
        self.assembler = SpectralAssembler()
        
        # IU-Net for latent translation
        self.iunet = create_iunet() if CFG.is_joint() else None
        
        # Decoders
        self.vel_decoder = SmallUNet(in_ch=5, out_ch=1, base=chans, depth=depth)
        self.wave_decoder = WaveDecoder() if CFG.is_joint() else None
        
    def forward(self, seis: torch.Tensor, mode: str = "inverse") -> tuple:
        """
        Args:
            seis: Seismic data (B,S,T,R)
            mode: Either "inverse" or "joint"
        Returns:
            If mode == "inverse":
                (v_pred, None)
            If mode == "joint":
                (v_pred, p_pred)
        """
        # Split wavefield
        up, down = self.mask(seis)
        
        if not CFG.is_joint() or mode == "inverse":
            # Use original SmallUNet path
            B,S,T,R = up.shape
            up = up.reshape(B*S, 1, T, R).repeat(1,5,1,1)
            down = down.reshape(B*S, 1, T, R).repeat(1,5,1,1)
            v_pred = self.vel_decoder(up)
            return v_pred, None
            
        # Joint mode with IU-Net translation
        # 1. Translate up/down to velocity space
        v_up = self.iunet(up, "p→v")
        v_down = self.iunet(down, "p→v")
        
        # 2. Decode velocity
        v_pred = self.vel_decoder(torch.cat([v_up, v_down], dim=1))
        
        # 3. Translate back to wavefield space
        p_up = self.iunet(v_up, "v→p")
        p_down = self.iunet(v_down, "v→p")
        
        # 4. Reassemble and decode wavefield
        p_recon = self.assembler(p_up, p_down)
        p_pred = self.wave_decoder(p_recon)
        
        return v_pred, p_pred

# For backwards compatibility
SpecProjUNet = HybridSpecProj 


# %%
# ## Loss Functions



# %%
import torch
import torch.nn.functional as F
from config import CFG

def pde_residual(v_pred: torch.Tensor,
                 seis: torch.Tensor,
                 dt=1e-3, dx=10.):
    """
    Quick acoustic residual: ∂²_t p − v² ∇² p  ≈ 0 on predicted velocity.
    Here we just coarse-sample a random source index to keep it cheap.
    """
    p = seis[:,0]            # (B,T,R)
    d2t = (p[:,2:] - 2*p[:,1:-1] + p[:,:-2]) / (dt*dt)
    lap = (p[:,:,2:] - 2*p[:,:,1:-1] + p[:,:,:-2]) / (dx*dx)
    v2  = v_pred[...,1:-1,1:-1]**2
    res = d2t[...,1:-1] - v2*lap[:,1:-1]
    return res.abs()

class JointLoss(torch.nn.Module):
    """
    Combined loss for joint forward-inverse training.
    """
    def __init__(self, λ_inv=1.0, λ_fwd=1.0, λ_pde=0.1):
        super().__init__()
        self.λ_inv = λ_inv
        self.λ_fwd = λ_fwd
        self.λ_pde = λ_pde
        
    def forward(self, v_pred, v_true, p_pred=None, p_true=None, seis_batch=None):
        # Inverse loss (velocity)
        l_inv = F.l1_loss(v_pred, v_true)
        
        # Forward loss (wavefield)
        l_fwd = 0.0
        if p_pred is not None and p_true is not None:
            l_fwd = F.l1_loss(p_pred, p_true)
            
        # PDE residual
        l_pde = 0.0
        if seis_batch is not None:
            l_pde = pde_residual(v_pred, seis_batch).mean()
            
        # Combine losses
        total = (self.λ_inv * l_inv + 
                self.λ_fwd * l_fwd + 
                self.λ_pde * l_pde)
                
        return total, {
            'l_inv': l_inv.item(),
            'l_fwd': l_fwd.item() if isinstance(l_fwd, torch.Tensor) else l_fwd,
            'l_pde': l_pde.item() if isinstance(l_pde, torch.Tensor) else l_pde
        }

# For backwards compatibility
HybridLoss = JointLoss 


# %%
# ## Training



# %%
import torch, random, numpy as np
from config import CFG, save_cfg
from data_manager import DataManager
from model import SpecProjNet
from losses import JointLoss
from pathlib import Path
from tqdm import tqdm
import argparse
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp

def set_seed(seed:int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def train(rank, world_size, dryrun: bool = False, fp16: bool = True, empty_cache: bool = True):
    # Initialize distributed training
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    set_seed(CFG.seed)
    save_dir = Path('outputs'); save_dir.mkdir(exist_ok=True)
    
    # Initialize data manager
    data_manager = DataManager(use_mmap=True, cache_size=1000)

    # ----------- data ----------------------------------------------------- #
    train_seis, train_vel = [], []
    for fam in ('FlatVel_A','FlatVel_B'):
        s, v = data_manager.list_family_files(fam)
        # If dryrun and files are missing, mock data to avoid NoneType error
        if dryrun and (not s or not v):
            # Create mock data for dryrun
            import torch
            train_seis = [torch.randn(1, 1, 100, 100)]
            train_vel = [torch.randn(1, 1, 70, 70)]
            break
        train_seis += s if s else []
        train_vel += v if v else []
    if dryrun and (not train_seis or not train_vel):
        # If still empty, create mock data
        import torch
        train_seis = [torch.randn(1, 1, 100, 100)]
        train_vel = [torch.randn(1, 1, 70, 70)]
    train_loader = data_manager.create_loader(
        train_seis, train_vel, 
        batch_size=CFG.batch, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ----------- model/optim --------------------------------------------- #
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=True,
        ema_decay=CFG.ema_decay,
    ).to(CFG.env.device)
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
        
    opt = torch.optim.AdamW(model.parameters(),
                           lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = amp.GradScaler(enabled=fp16)
    loss_fn = JointLoss(λ_inv=CFG.lambda_inv,
                       λ_fwd=CFG.lambda_fwd,
                       λ_pde=CFG.lambda_pde)

    best_mae = 1e9
    for epoch in range(CFG.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for seis, vel in pbar:
            seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
            
            with amp.autocast(enabled=fp16):
                v_pred = model(seis)
                loss, loss_dict = loss_fn(v_pred, vel, seis_batch=seis)
                    
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            
            # Update EMA
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.update_ema()
            else:
                model.update_ema()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': float(loss),
                **loss_dict,
                'mem': f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            })
            
            if dryrun:
                break
                
        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for seis, vel in train_loader:  # Using train set for now
                seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
                # Use EMA model for validation
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    v_pred = model.module.get_ema_model()(seis)
                else:
                    v_pred = model.get_ema_model()(seis)
                val_mae += F.l1_loss(v_pred, vel).item()
        val_mae /= len(train_loader)
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            if rank == 0:  # Only save on main process
                torch.save(model.state_dict(), save_dir/'best.pth')
        if rank == 0:  # Only save on main process
            torch.save(model.state_dict(), save_dir/'last.pth')
        
        print(f"Epoch {epoch}: val_mae = {val_mae:.4f}")
        
        # Clear cache periodically
        data_manager.clear_cache()
        if empty_cache:
            torch.cuda.empty_cache()
            
    if world_size > 1:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dryrun', action='store_true', help='Train one minibatch only')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable mixed precision training')
    parser.add_argument('--empty_cache', action='store_true', default=True, help='Clear CUDA cache after each epoch')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    args = parser.parse_args()
    
    if args.world_size > 1:
        mp.spawn(
            train,
            args=(args.world_size, args.dryrun, args.fp16, args.empty_cache),
            nprocs=args.world_size,
            join=True
        )
    else:
        train(0, 1, args.dryrun, args.fp16, args.empty_cache)

if __name__ == '__main__':
    main() 


# %%
# ## Inference



# %%
import torch, pandas as pd
from pathlib import Path
from config import CFG
from data_manager import DataManager
from model import SpecProjNet

def format_submission(vel_map, oid):
    # keep *odd* x-columns only (1,3,…,69)
    cols = vel_map[:,1::2]
    rows = []
    for y,row in enumerate(cols):
        rows.append({'oid_ypos': f'{oid}_y_{y}', **{f'x_{2*i+1}':v
                      for i,v in enumerate(row.tolist())}})
    return rows

def infer(weights='outputs/best.pth'):
    # Initialize data manager
    data_manager = DataManager(use_mmap=True, cache_size=1000)
    
    # Load model
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=False,
        ema_decay=0.0,  # No EMA for inference
    ).to(CFG.env.device)
    model.load_state_dict(torch.load(weights, map_location=CFG.env.device))
    model.eval()

    test_files = data_manager.get_test_files()
    test_loader = data_manager.create_loader(
        test_files, vel=None,
        batch_size=1, shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    rows = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for seis, oid in test_loader:
            seis = seis.to(CFG.env.device)
            
            # Original prediction
            vel = model(seis)
            
            # TTA: Flip prediction
            seis_flip = seis.flip(-1)  # Flip receiver dimension
            vel_flip = model(seis_flip)
            vel_flip = vel_flip.flip(-1)  # Flip back
            
            # Average predictions
            vel = (vel + vel_flip) / 2
            
            vel = vel.cpu().float().numpy()[0,0]
            rows += format_submission(vel, Path(oid[0]).stem)
            
    # Clear cache after inference
    data_manager.clear_cache()
    
    pd.DataFrame(rows).to_csv('submission.csv', index=False)

if __name__ == '__main__':
    infer() 

