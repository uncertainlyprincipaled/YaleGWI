# %% [markdown]
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

            # Enable joint training by default in Kaggle
            cls._inst.enable_joint = cls._inst.env.kind == 'kaggle'

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