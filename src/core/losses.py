# %% [markdown]
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