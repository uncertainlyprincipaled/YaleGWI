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