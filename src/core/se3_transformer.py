import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)

class SE3Attention(nn.Module):
    """
    SE(3)-equivariant attention mechanism.
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Initialize SE(3) attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self,
                x: torch.Tensor,
                pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of SE(3) attention.
        
        Args:
            x: Input features [batch_size, num_nodes, hidden_dim]
            pos: Node positions [batch_size, num_nodes, 3]
            mask: Attention mask [batch_size, num_nodes, num_nodes]
            
        Returns:
            Attended features [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, num_nodes, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add positional bias (distance-based)
        if pos is not None:
            # Compute pairwise distances
            dist = torch.cdist(pos, pos)  # [batch_size, num_nodes, num_nodes]
            dist = dist.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Distance-based bias
            dist_bias = -dist / 10.0  # Soft distance penalty
            scores = scores + dist_bias
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        
        return out

class SE3TransformerLayer(nn.Module):
    """
    Single layer of SE(3)-Transformer.
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        """
        Initialize SE(3) transformer layer.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        
        # Self-attention
        self.attention = SE3Attention(hidden_dim, num_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self,
                x: torch.Tensor,
                pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of SE(3) transformer layer.
        
        Args:
            x: Input features [batch_size, num_nodes, hidden_dim]
            pos: Node positions [batch_size, num_nodes, 3]
            mask: Attention mask [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated features [batch_size, num_nodes, hidden_dim]
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, pos, mask)
        x = self.attention_norm(x + attn_out)
        
        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = self.mlp_norm(x + mlp_out)
        
        return x

class SE3Transformer(nn.Module):
    """
    SE(3)-Transformer for 3D equivariance.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 max_positions: int = 1000):
        """
        Initialize SE(3)-Transformer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            dropout: Dropout rate
            max_positions: Maximum number of positions for positional encoding
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_positions = max_positions
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_positions, hidden_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SE3TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def _get_positional_encoding(self, num_nodes: int) -> torch.Tensor:
        """Get positional encoding for given number of nodes."""
        if num_nodes <= self.max_positions:
            return self.pos_encoding[:, :num_nodes, :]
        else:
            # Extend positional encoding if needed
            extended_pos = torch.randn(1, num_nodes, self.hidden_dim, device=self.pos_encoding.device)
            extended_pos[:, :self.max_positions, :] = self.pos_encoding
            return extended_pos
    
    def forward(self,
                x: torch.Tensor,
                pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of SE(3)-Transformer.
        
        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            pos: Node positions [batch_size, num_nodes, 3]
            mask: Attention mask [batch_size, num_nodes, num_nodes]
            
        Returns:
            Transformed features [batch_size, num_nodes, input_dim]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)
        
        # Add positional encoding
        pos_enc = self._get_positional_encoding(num_nodes)
        h = h + pos_enc
        
        # Process through transformer layers
        for layer in self.layers:
            h = layer(h, pos, mask)
        
        # Final normalization and output projection
        h = self.final_norm(h)
        out = self.output_proj(h)
        
        return out

class SE3WavefieldProcessor(nn.Module):
    """
    SE(3)-equivariant wavefield processor for 3D seismic data.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        """
        Initialize SE(3) wavefield processor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # SE(3)-Transformer
        self.transformer = SE3Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Feature extraction for 3D coordinates
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def _extract_3d_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features and positions from 3D wavefield data.
        
        Args:
            x: 3D wavefield data [batch_size, channels, depth, height, width]
            
        Returns:
            Tuple of (features, positions)
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Reshape to 2D for processing
        x_2d = x.view(batch_size, channels, depth * height, width)
        
        # Create 3D coordinate grid
        z_coords = torch.linspace(-1, 1, depth, device=x.device)
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = torch.linspace(-1, 1, width, device=x.device)
        
        Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Expand to batch size
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, depth*height*width, 3]
        
        # Extract features
        features = x_2d.transpose(1, 2)  # [batch_size, depth*height*width, channels]
        
        return features, coords
    
    def _reconstruct_3d(self, features: torch.Tensor, depth: int, height: int, width: int) -> torch.Tensor:
        """
        Reconstruct 3D tensor from features.
        
        Args:
            features: Features [batch_size, depth*height*width, channels]
            depth: Depth dimension
            height: Height dimension
            width: Width dimension
            
        Returns:
            3D tensor [batch_size, channels, depth, height, width]
        """
        batch_size, _, channels = features.shape
        
        # Reshape back to 3D
        x_3d = features.transpose(1, 2).view(batch_size, channels, depth, height, width)
        
        return x_3d
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE(3) wavefield processor.
        
        Args:
            x: 3D wavefield data [batch_size, channels, depth, height, width]
            
        Returns:
            Processed 3D wavefield [batch_size, channels, depth, height, width]
        """
        batch_size, channels, depth, height, width = x.shape
        
        # Extract features and coordinates
        features, coords = self._extract_3d_features(x)
        
        # Process through SE(3)-Transformer
        processed_features = self.transformer(features, coords)
        
        # Reconstruct 3D tensor
        output = self._reconstruct_3d(processed_features, depth, height, width)
        
        return output 