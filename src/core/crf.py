import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ContinuousCRF(nn.Module):
    """
    Continuous Conditional Random Field for post-processing.
    Implements locally-connected CRF with mean-field inference.
    """
    
    def __init__(self,
                 num_classes: int = 1,
                 window_size: int = 5,
                 num_iterations: int = 5,
                 theta_alpha: float = 1.0,
                 theta_beta: float = 1.0,
                 theta_gamma: float = 1.0,
                 temperature: float = 1.0,
                 trainable: bool = False):
        """
        Initialize continuous CRF.
        
        Args:
            num_classes: Number of output classes
            window_size: Size of local connectivity window
            num_iterations: Number of mean-field iterations
            theta_alpha: Weight for appearance kernel
            theta_beta: Weight for smoothness kernel
            theta_gamma: Weight for compatibility kernel
            temperature: Temperature for soft assignments
            trainable: Whether CRF parameters are trainable
        """
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        self.num_iterations = num_iterations
        self.temperature = temperature
        
        # CRF parameters
        if trainable:
            self.theta_alpha = nn.Parameter(torch.tensor(theta_alpha))
            self.theta_beta = nn.Parameter(torch.tensor(theta_beta))
            self.theta_gamma = nn.Parameter(torch.tensor(theta_gamma))
        else:
            self.register_buffer('theta_alpha', torch.tensor(theta_alpha))
            self.register_buffer('theta_beta', torch.tensor(theta_beta))
            self.register_buffer('theta_gamma', torch.tensor(theta_gamma))
        
        # Compatibility matrix
        self.compatibility = nn.Parameter(torch.eye(num_classes))
        
        # Boundary-aware pairwise potentials
        self.boundary_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        torch.nn.init.constant_(self.boundary_conv.weight, 1.0/9.0)
    
    def _create_local_connectivity(self, height: int, width: int) -> torch.Tensor:
        """
        Create local connectivity pattern.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Connectivity matrix [height*width, height*width]
        """
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        
        # Flatten coordinates
        coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=-1)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(coords, coords)
        
        # Create local connectivity mask
        connectivity = (dist_matrix <= self.window_size).float()
        
        # Remove self-connections
        connectivity.fill_diagonal_(0.0)
        
        return connectivity
    
    def _compute_appearance_kernel(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute appearance-based pairwise potentials.
        
        Args:
            features: Input features [batch_size, channels, height, width]
            
        Returns:
            Appearance kernel [batch_size, height*width, height*width]
        """
        batch_size, channels, height, width = features.shape
        
        # Reshape features
        features_flat = features.view(batch_size, channels, -1)  # [batch_size, channels, height*width]
        
        # Compute feature similarities
        features_norm = F.normalize(features_flat, dim=1)
        similarity = torch.bmm(features_norm.transpose(1, 2), features_norm)  # [batch_size, height*width, height*width]
        
        # Apply appearance weight
        appearance_kernel = torch.exp(-self.theta_alpha * (1.0 - similarity))
        
        return appearance_kernel
    
    def _compute_smoothness_kernel(self, height: int, width: int) -> torch.Tensor:
        """
        Compute smoothness-based pairwise potentials.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Smoothness kernel [height*width, height*width]
        """
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        
        # Flatten coordinates
        coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=-1)
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(coords, coords)
        
        # Apply smoothness weight
        smoothness_kernel = torch.exp(-self.theta_beta * dist_matrix)
        
        return smoothness_kernel
    
    def _compute_boundary_kernel(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware pairwise potentials.
        
        Args:
            input_tensor: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Boundary kernel [batch_size, height*width, height*width]
        """
        batch_size, channels, height, width = input_tensor.shape
        
        # Detect boundaries using gradient
        grad_x = torch.diff(input_tensor, dim=-1, prepend=input_tensor[:, :, :, :1])
        grad_y = torch.diff(input_tensor, dim=-2, prepend=input_tensor[:, :, :1, :])
        
        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Apply boundary convolution
        boundary_response = self.boundary_conv(gradient_magnitude)
        
        # Reshape to flat
        boundary_flat = boundary_response.view(batch_size, -1)  # [batch_size, height*width]
        
        # Create boundary kernel
        boundary_kernel = torch.bmm(boundary_flat.unsqueeze(-1), boundary_flat.unsqueeze(1))
        boundary_kernel = torch.exp(-self.theta_gamma * boundary_kernel)
        
        return boundary_kernel
    
    def _mean_field_step(self,
                        unary: torch.Tensor,
                        pairwise: torch.Tensor,
                        current_q: torch.Tensor) -> torch.Tensor:
        """
        Single mean-field update step.
        
        Args:
            unary: Unary potentials [batch_size, num_classes, height*width]
            pairwise: Pairwise potentials [batch_size, height*width, height*width]
            current_q: Current Q distribution [batch_size, num_classes, height*width]
            
        Returns:
            Updated Q distribution [batch_size, num_classes, height*width]
        """
        batch_size, num_classes, num_pixels = unary.shape
        
        # Message passing
        messages = torch.bmm(pairwise, current_q.transpose(1, 2))  # [batch_size, height*width, num_classes]
        messages = messages.transpose(1, 2)  # [batch_size, num_classes, height*width]
        
        # Apply compatibility
        compatibility_messages = torch.bmm(
            self.compatibility.unsqueeze(0).expand(batch_size, -1, -1),
            messages
        )
        
        # Combine with unary potentials
        combined = unary + compatibility_messages
        
        # Apply temperature and softmax
        q_new = F.softmax(combined / self.temperature, dim=1)
        
        return q_new
    
    def forward(self,
                unary: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of continuous CRF.
        
        Args:
            unary: Unary potentials [batch_size, num_classes, height, width]
            features: Optional features for appearance kernel [batch_size, channels, height, width]
            
        Returns:
            Refined predictions [batch_size, num_classes, height, width]
        """
        batch_size, num_classes, height, width = unary.shape
        
        # Reshape unary potentials
        unary_flat = unary.view(batch_size, num_classes, -1)  # [batch_size, num_classes, height*width]
        
        # Initialize Q distribution
        q = F.softmax(unary_flat / self.temperature, dim=1)
        
        # Create local connectivity
        connectivity = self._create_local_connectivity(height, width)
        
        # Compute pairwise potentials
        if features is not None:
            appearance_kernel = self._compute_appearance_kernel(features)
            smoothness_kernel = self._compute_smoothness_kernel(height, width)
            boundary_kernel = self._compute_boundary_kernel(unary)
            
            # Combine kernels
            pairwise = (appearance_kernel * smoothness_kernel.unsqueeze(0) * boundary_kernel) * connectivity.unsqueeze(0)
        else:
            smoothness_kernel = self._compute_smoothness_kernel(height, width)
            pairwise = smoothness_kernel.unsqueeze(0) * connectivity.unsqueeze(0)
        
        # Mean-field iterations
        for _ in range(self.num_iterations):
            q = self._mean_field_step(unary_flat, pairwise, q)
        
        # Reshape back to original dimensions
        output = q.view(batch_size, num_classes, height, width)
        
        return output
    
    @property
    def device(self) -> torch.device:
        """Get device of CRF parameters."""
        return self.theta_alpha.device

class CRFPostProcessor(nn.Module):
    """
    Post-processor using continuous CRF.
    """
    
    def __init__(self,
                 num_classes: int = 1,
                 window_size: int = 5,
                 num_iterations: int = 5,
                 theta_alpha: float = 1.0,
                 theta_beta: float = 1.0,
                 theta_gamma: float = 1.0,
                 temperature: float = 1.0,
                 trainable: bool = False):
        """
        Initialize CRF post-processor.
        
        Args:
            num_classes: Number of output classes
            window_size: Size of local connectivity window
            num_iterations: Number of mean-field iterations
            theta_alpha: Weight for appearance kernel
            theta_beta: Weight for smoothness kernel
            theta_gamma: Weight for compatibility kernel
            temperature: Temperature for soft assignments
            trainable: Whether CRF parameters are trainable
        """
        super().__init__()
        
        # CRF layer
        self.crf = ContinuousCRF(
            num_classes=num_classes,
            window_size=window_size,
            num_iterations=num_iterations,
            theta_alpha=theta_alpha,
            theta_beta=theta_beta,
            theta_gamma=theta_gamma,
            temperature=temperature,
            trainable=trainable
        )
        
        # Feature extractor for appearance kernel
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CRF post-processor.
        
        Args:
            x: Input tensor [batch_size, num_classes, height, width]
            
        Returns:
            Post-processed tensor [batch_size, num_classes, height, width]
        """
        # Extract features for appearance kernel
        features = self.feature_extractor(x)
        
        # Apply CRF
        output = self.crf(x, features)
        
        return output
    
    def compute_crf_energy(self, unary: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute CRF energy for monitoring.
        
        Args:
            unary: Unary potentials
            features: Input features
            
        Returns:
            CRF energy
        """
        # This is a simplified energy computation
        # In practice, you'd compute the full CRF energy
        energy = torch.mean(unary**2)
        return energy 