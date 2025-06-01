# %% [markdown]
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