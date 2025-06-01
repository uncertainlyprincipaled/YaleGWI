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


# %%
# ## Configuration and Environment Setup



# %%
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Literal, NamedTuple

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
        return cls._inst

    @classmethod
    def initialize(cls):
        """Initialize the configuration and setup environment."""
        from src.core.setup import setup_environment
        setup_environment()
        return cls()

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
import kagglehub

def setup_environment():
    """Setup environment-specific configurations and download datasets if needed."""
    from config import CFG  # Import here to avoid circular dependency
    
    if CFG.env.kind == 'colab':
        # Install required packages
        try:
            import kagglehub
        except ImportError:
            subprocess.run(['pip', 'install', 'kagglehub'], check=True)
            import kagglehub

        # Clone repository if not already present
        repo_dir = Path('/content/YaleGWI')
        if not repo_dir.exists():
            print("Cloning repository from GitHub...")
            subprocess.run(['git', 'clone', 'https://github.com/your-username/YaleGWI.git', str(repo_dir)], check=True)
            os.chdir(repo_dir)
        
        # Create data directory
        data_dir = Path('/content/data')
        data_dir.mkdir(exist_ok=True)
        
        # Download dataset
        print("Downloading dataset from Kaggle...")
        kagglehub.model_download('jamie-morgan/waveform-inversion', path=str(data_dir))
        
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
        # Clone repository if not already present
        repo_dir = Path('/opt/ml/code/YaleGWI')
        if not repo_dir.exists():
            print("Cloning repository from GitHub...")
            subprocess.run(['git', 'clone', 'https://github.com/your-username/YaleGWI.git', str(repo_dir)], check=True)
            os.chdir(repo_dir)
            
        # AWS SageMaker specific setup
        data_dir = Path('/opt/ml/input/data')
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
        
        print("Environment setup complete for SageMaker") 


# %%
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


# %%
# ## Model Architecture - UNet



# %%
import torch, torch.nn as nn, torch.nn.functional as F
from proj_mask import PhysMask
from config import CFG

# ---- Minimal residual-UNet building blocks ------------------------------- #

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in , ch_out, 3, padding=1)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.bn1, self.bn2 = nn.BatchNorm2d(ch_out), nn.BatchNorm2d(ch_out)
        self.skip = (nn.Identity() if ch_in==ch_out
                     else nn.Conv2d(ch_in, ch_out, 1))
    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x))

class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.res = ResBlock(ch_in, ch_out)
    def forward(self,x): return self.res(self.mp(x))

class Up(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up  = nn.ConvTranspose2d(ch_in, ch_out, 2, stride=2)
        self.res = ResBlock(ch_in, ch_out)
    def forward(self,x, skip):
        x = torch.cat([self.up(x), skip], 1)
        return self.res(x)

# ---- SpecProj-UNet ------------------------------------------------------- #

class SmallUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=32, depth=4):
        super().__init__()
        self.inc  = ResBlock(in_ch, base)
        self.down = nn.ModuleList([Down(base*2**i, base*2**(i+1))
                                   for i in range(depth)])
        self.up   = nn.ModuleList([Up(base*2**(i+1), base*2**i)
                                   for i in reversed(range(depth))])
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self,x):
        skips=[self.inc(x)]
        for d in self.down: skips.append( d(skips[-1]) )
        x = skips.pop()
        for u in self.up:   x = u(x, skips.pop())
        return self.outc(x)

class SpecProjUNet(nn.Module):
    def __init__(self, chans=32, depth=4):
        super().__init__()
        self.mask = PhysMask()
        self.unet_up   = SmallUNet(in_ch=5, out_ch=1, base=chans, depth=depth)
        self.unet_down = SmallUNet(in_ch=5, out_ch=1, base=chans, depth=depth)
        self.fuse = nn.Conv2d(2, 1, 1)

    def forward(self, x):            # x (B,S,T,R)
        up, down = self.mask(x)      # still (B,S,T,R)
        # merge source dim into batch for UNet (expect 5 channels)
        B,S,T,R = up.shape
        up   = up  .reshape(B*S, 1, T, R).repeat(1,5,1,1)
        down = down.reshape(B*S, 1, T, R).repeat(1,5,1,1)
        vu  = self.unet_up (up )
        vd  = self.unet_down(down)
        fused = self.fuse(torch.cat([vu, vd], 1))
        return fused.view(B, S, 1, *fused.shape[-2:]).mean(1)  # (B,1,H,W) 


# %%
# ## Loss Functions



# %%
import torch, torch.nn.functional as F
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

class HybridLoss(torch.nn.Module):
    def __init__(self, λ_pde=CFG.lambda_pde):
        super().__init__()
        self.λ = λ_pde
    def forward(self, v_pred, v_true, seis_batch):
        l1  = F.l1_loss(v_pred, v_true)
        pde = pde_residual(v_pred, seis_batch).mean()
        return l1 + self.λ * pde 


# %%
# ## Training



# %%
import torch, random, numpy as np
from config import CFG, save_cfg
from data_utils import list_family_files, make_loader
from specproj_unet import SpecProjUNet
from losses import HybridLoss
from pathlib import Path
from tqdm import tqdm
import argparse

def set_seed(seed:int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def train(dryrun: bool = False):
    set_seed(CFG.seed)
    save_dir = Path('outputs'); save_dir.mkdir(exist_ok=True)
    save_cfg(save_dir)

    # ----------- data ----------------------------------------------------- #
    train_seis, train_vel = [], []
    for fam in ('FlatVel_A','FlatVel_B'):
        s, v = list_family_files(fam)
        train_seis += s; train_vel += v
    train_loader = make_loader(train_seis, train_vel, CFG.batch, shuffle=True)

    # ----------- model/optim --------------------------------------------- #
    model = SpecProjUNet().to(CFG.env.device)
    if CFG.env.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    opt    = torch.optim.AdamW(model.parameters(),
                               lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn= HybridLoss()

    best_mae = 1e9
    for epoch in range(CFG.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for seis, vel in pbar:
            seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
            with torch.amp.autocast('cuda'):
                pred = model(seis)
                loss = loss_fn(pred, vel, seis)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            pbar.set_postfix(loss=float(loss))
            if dryrun:
                break
        # TODO: add validation & MAE tracking
        torch.save(model.state_dict(), save_dir/'last.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dryrun', action='store_true', help='Train one minibatch only')
    args = parser.parse_args()
    train(args.dryrun) 


# %%
# ## Inference



# %%
import torch, pandas as pd
from pathlib import Path
from config import CFG
from data_utils import list_family_files, make_loader
from specproj_unet import SpecProjUNet

def format_submission(vel_map, oid):
    # keep *odd* x-columns only (1,3,…,69)
    cols = vel_map[:,1::2]
    rows = []
    for y,row in enumerate(cols):
        rows.append({'oid_ypos': f'{oid}_y_{y}', **{f'x_{2*i+1}':v
                      for i,v in enumerate(row.tolist())}})
    return rows

def infer(weights='outputs/last.pth'):
    model = SpecProjUNet().to(CFG.env.device)
    model.load_state_dict(torch.load(weights, map_location=CFG.env.device))
    model.eval()

    test_files = sorted(CFG.paths.test.glob('*.npy'))
    test_loader= make_loader(test_files, vel=None,
                             batch=1, shuffle=False)

    rows=[]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for seis, oid in test_loader:
            seis = seis.to(CFG.env.device)
            vel  = model(seis).cpu().float().numpy()[0,0]
            rows += format_submission(vel, Path(oid[0]).stem)
    pd.DataFrame(rows).to_csv('submission.csv', index=False)

if __name__ == '__main__':
    infer() 

