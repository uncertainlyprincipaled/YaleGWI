# %% [markdown]
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