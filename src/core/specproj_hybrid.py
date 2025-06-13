import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG
from .proj_mask import PhysMask, SpectralAssembler
from .iunet import create_iunet
from .specproj_unet import SmallUNet
import logging

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
        # Log input tensor stats
        logging.info(f"Input seis shape: {seis.shape}, dtype: {seis.dtype}, min: {seis.min().item()}, max: {seis.max().item()}, mean: {seis.mean().item()}")
        
        # Split wavefield
        up, down = self.mask(seis)
        
        # Log split tensors
        logging.info(f"Split tensors - up: {up.shape}, down: {down.shape}")
        
        if not CFG.is_joint() or mode == "inverse":
            # Use original SmallUNet path
            B,S,T,R = up.shape
            up = up.reshape(B*S, 1, T, R).repeat(1,5,1,1)
            down = down.reshape(B*S, 1, T, R).repeat(1,5,1,1)
            
            # Log reshaped tensors
            logging.info(f"Reshaped tensors - up: {up.shape}, down: {down.shape}")
            
            v_pred = self.vel_decoder(up)
            logging.info(f"v_pred shape: {v_pred.shape}, dtype: {v_pred.dtype}, min: {v_pred.min().item()}, max: {v_pred.max().item()}, mean: {v_pred.mean().item()}")
            return v_pred, None
            
        # Joint mode with IU-Net translation
        # 1. Translate up/down to velocity space
        v_up = self.iunet(up, "p→v")
        v_down = self.iunet(down, "p→v")
        
        # Log translated tensors
        logging.info(f"Translated tensors - v_up: {v_up.shape}, v_down: {v_down.shape}")
        
        # 2. Decode velocity
        v_pred = self.vel_decoder(torch.cat([v_up, v_down], dim=1))
        logging.info(f"v_pred shape: {v_pred.shape}, dtype: {v_pred.dtype}, min: {v_pred.min().item()}, max: {v_pred.max().item()}, mean: {v_pred.mean().item()}")
        
        # 3. Translate back to wavefield space
        p_up = self.iunet(v_up, "v→p")
        p_down = self.iunet(v_down, "v→p")
        
        # 4. Reassemble and decode wavefield
        p_recon = self.assembler(p_up, p_down)
        p_pred = self.wave_decoder(p_recon)
        
        # Log final predictions
        logging.info(f"p_pred shape: {p_pred.shape}, dtype: {p_pred.dtype}, min: {p_pred.min().item()}, max: {p_pred.max().item()}, mean: {p_pred.mean().item()}")
        
        return v_pred, p_pred

# For backwards compatibility
SpecProjUNet = HybridSpecProj 