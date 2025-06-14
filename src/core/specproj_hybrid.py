import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.config import CFG
from src.core.proj_mask import PhysMask, SpectralAssembler
from src.core.iunet import create_iunet
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
    Memory-efficient implementation for Kaggle.
    """
    def __init__(self, chans=32, depth=4):
        super().__init__()
        self.mask = PhysMask()
        self.assembler = SpectralAssembler()
        
        # IU-Net for latent translation (only if joint training)
        self.iunet = create_iunet() if CFG.is_joint() else None
        
        # Decoders
        self.vel_decoder = SpecProjNet(
            backbone=CFG.backbone,
            pretrained=False,
            ema_decay=CFG.ema_decay
        )
        self.wave_decoder = WaveDecoder() if CFG.is_joint() else None
        
        # Enable gradient checkpointing if configured
        if CFG.gradient_checkpointing:
            self.vel_decoder.backbone.use_checkpoint = True
            if self.iunet is not None:
                self.iunet.use_checkpoint = True
        
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
            # Use original SpecProjNet path
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