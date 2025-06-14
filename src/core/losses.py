# %% [markdown]
# ## Loss Functions

# %%
import torch
import torch.nn.functional as F
from src.core.config import CFG
import torch.cuda.amp as amp
from typing import Dict, Tuple, Optional

def track_memory_usage():
    """Track current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0.0

def pde_residual(v_pred: torch.Tensor,
                 seis: torch.Tensor,
                 dt: float = 1e-3,
                 dx: float = 10.,
                 use_amp: bool = True) -> torch.Tensor:
    """
    Enhanced acoustic wave equation residual with mixed precision support.
    
    Args:
        v_pred: Predicted velocity field
        seis: Seismic data
        dt: Time step
        dx: Spatial step
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        PDE residual loss
    """
    with amp.autocast(enabled=use_amp):
        p = seis[:,0]  # (B,T,R)
        
        # Second-order time derivative
        d2t = (p[:,2:] - 2*p[:,1:-1] + p[:,:-2]) / (dt*dt)
        
        # Laplacian in both dimensions
        lap_x = (p[:,:,2:] - 2*p[:,:,1:-1] + p[:,:,:-2]) / (dx*dx)
        lap_y = (p[:,2:,:] - 2*p[:,1:-1,:] + p[:,:-2,:]) / (dx*dx)
        lap = lap_x + lap_y
        
        # Velocity squared
        v2 = v_pred[...,1:-1,1:-1]**2
        
        # Full residual
        res = d2t[...,1:-1] - v2*lap[:,1:-1]
        
        return res.abs()

class JointLoss(torch.nn.Module):
    """
    Enhanced joint loss with memory tracking and mixed precision support.
    """
    def __init__(self,
                 λ_inv: float = 1.0,
                 λ_fwd: float = 1.0,
                 λ_pde: float = 0.1,
                 λ_reg: float = 0.01,
                 use_amp: bool = True):
        super().__init__()
        self.λ_inv = λ_inv
        self.λ_fwd = λ_fwd
        self.λ_pde = λ_pde
        self.λ_reg = λ_reg
        self.use_amp = use_amp
        
    def forward(self,
                v_pred: torch.Tensor,
                v_true: torch.Tensor,
                p_pred: Optional[torch.Tensor] = None,
                p_true: Optional[torch.Tensor] = None,
                seis_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        with amp.autocast(enabled=self.use_amp):
            # Inverse loss (velocity)
            l_inv = F.l1_loss(v_pred, v_true)
            
            # Forward loss (wavefield)
            l_fwd = torch.tensor(0.0, device=v_pred.device)
            if p_pred is not None and p_true is not None:
                l_fwd = F.l1_loss(p_pred, p_true)
                
            # PDE residual
            l_pde = torch.tensor(0.0, device=v_pred.device)
            if seis_batch is not None:
                l_pde = pde_residual(v_pred, seis_batch, use_amp=self.use_amp).mean()
                
            # Regularization loss (optional)
            l_reg = torch.tensor(0.0, device=v_pred.device)
            if hasattr(self, 'λ_reg') and self.λ_reg > 0:
                l_reg = F.mse_loss(v_pred[:,:,1:] - v_pred[:,:,:-1], 
                                 torch.zeros_like(v_pred[:,:,1:]))
            
            # Combine losses
            total = (self.λ_inv * l_inv + 
                    self.λ_fwd * l_fwd + 
                    self.λ_pde * l_pde +
                    self.λ_reg * l_reg)
            
            # Track memory usage
            memory_used = track_memory_usage()
                    
            return total, {
                'l_inv': l_inv.item(),
                'l_fwd': l_fwd.item() if isinstance(l_fwd, torch.Tensor) else l_fwd,
                'l_pde': l_pde.item() if isinstance(l_pde, torch.Tensor) else l_pde,
                'l_reg': l_reg.item() if isinstance(l_reg, torch.Tensor) else l_reg,
                'memory_mb': memory_used
            }

def get_loss_fn() -> torch.nn.Module:
    """Get the appropriate loss function based on configuration."""
    if CFG.is_joint():
        return JointLoss(
            λ_inv=CFG.lambda_inv,
            λ_fwd=CFG.lambda_fwd,
            λ_pde=CFG.lambda_pde,
            λ_reg=CFG.lambda_reg if hasattr(CFG, 'lambda_reg') else 0.01,
            use_amp=CFG.use_amp if hasattr(CFG, 'use_amp') else True
        )
    else:
        return torch.nn.L1Loss()  # Default to L1 loss for non-joint training

# For backwards compatibility
HybridLoss = JointLoss 