import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class EnsembleBase(ABC, nn.Module):
    """
    Base class for ensemble models.
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Initialize ensemble base.
        
        Args:
            models: List of base models
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if self.num_models == 0:
            raise ValueError("Ensemble must contain at least one model")
    
    @abstractmethod
    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble weights.
        
        Args:
            x: Input tensor
            
        Returns:
            Weights tensor [num_models]
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=0)  # [num_models, batch_size, ...]
        
        # Compute weights
        weights = self.compute_weights(x)  # [num_models]
        
        # Apply weights
        weighted_preds = stacked_preds * weights.view(-1, 1, 1, 1, 1)
        
        # Average predictions
        ensemble_pred = weighted_preds.sum(dim=0)
        
        return ensemble_pred

class StaticEnsemble(EnsembleBase):
    """
    Ensemble with static (learnable) weights.
    """
    
    def __init__(self, 
                 models: List[nn.Module],
                 initial_weights: Optional[List[float]] = None):
        """
        Initialize static ensemble.
        
        Args:
            models: List of base models
            initial_weights: Initial weights for models (if None, equal weights)
        """
        super().__init__(models)
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = [1.0 / self.num_models] * self.num_models
        
        if len(initial_weights) != self.num_models:
            raise ValueError(f"Expected {self.num_models} weights, got {len(initial_weights)}")
        
        # Learnable weights
        self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
    
    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute static weights.
        
        Args:
            x: Input tensor (not used for static weights)
            
        Returns:
            Weights tensor [num_models]
        """
        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(self.weights, dim=0)
        return weights

class DynamicEnsemble(EnsembleBase):
    """
    Ensemble with dynamic weights based on input features.
    """
    
    def __init__(self,
                 models: List[nn.Module],
                 feature_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize dynamic ensemble.
        
        Args:
            models: List of base models
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for weight predictor
            num_layers: Number of layers in weight predictor
            dropout: Dropout rate
        """
        super().__init__(models)
        self.feature_dim = feature_dim
        
        # Weight predictor network
        layers = []
        input_dim = feature_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.num_models))
        
        self.weight_predictor = nn.Sequential(*layers)
    
    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic weights based on input features.
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            Weights tensor [num_models]
        """
        # Extract features (use mean pooling for spatial dimensions)
        if x.dim() > 2:
            # Global average pooling
            features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            features = x
        
        # Predict weights
        logits = self.weight_predictor(features)
        
        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(logits, dim=-1)
        
        # Average across batch dimension
        weights = weights.mean(dim=0)  # [num_models]
        
        return weights

class BayesianEnsemble(EnsembleBase):
    """
    Ensemble with Bayesian uncertainty estimation.
    """
    
    def __init__(self,
                 models: List[nn.Module],
                 mc_samples: int = 10,
                 dropout_rate: float = 0.1):
        """
        Initialize Bayesian ensemble.
        
        Args:
            models: List of base models
            mc_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate for uncertainty estimation
        """
        super().__init__(models)
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        
        # Enable dropout for all models
        for model in self.models:
            self._enable_dropout(model)
    
    def _enable_dropout(self, model: nn.Module):
        """Enable dropout in all layers of the model."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                module.training = True
    
    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute weights based on uncertainty.
        
        Args:
            x: Input tensor (not used for uncertainty-based weights)
            
        Returns:
            Weights tensor [num_models]
        """
        # Equal weights for Bayesian ensemble
        weights = torch.ones(self.num_models, device=x.device) / self.num_models
        return weights
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        # Collect Monte Carlo samples
        all_predictions = []
        
        for _ in range(self.mc_samples):
            # Get predictions from all models
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
            
            # Stack predictions
            stacked_preds = torch.stack(predictions, dim=0)
            all_predictions.append(stacked_preds)
        
        # Stack all samples
        all_preds = torch.stack(all_predictions, dim=0)  # [mc_samples, num_models, batch_size, ...]
        
        # Compute mean and variance
        mean_pred = all_preds.mean(dim=(0, 1))  # Average over samples and models
        var_pred = all_preds.var(dim=(0, 1))    # Variance over samples and models
        
        return mean_pred, var_pred

class MetaLearner(nn.Module):
    """
    Lightweight meta-learner for dynamic ensemble weighting.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize meta-learner.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of meta-learner.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Meta-prediction [batch_size, 1]
        """
        features = self.feature_extractor(x)
        output = self.output_proj(features)
        return output

class GeometricEnsemble(EnsembleBase):
    """
    Ensemble with geometric-aware weighting.
    """
    
    def __init__(self,
                 models: List[nn.Module],
                 feature_dim: int,
                 num_geometric_features: int = 8,
                 hidden_dim: int = 128):
        """
        Initialize geometric ensemble.
        
        Args:
            models: List of base models
            feature_dim: Input feature dimension
            num_geometric_features: Number of geometric features
            hidden_dim: Hidden dimension
        """
        super().__init__(models)
        self.feature_dim = feature_dim
        self.num_geometric_features = num_geometric_features
        
        # Geometric feature extractor
        self.geometric_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_geometric_features),
            nn.Tanh()
        )
        
        # Weight predictor based on geometric features
        self.weight_predictor = nn.Sequential(
            nn.Linear(num_geometric_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_models),
            nn.Softmax(dim=-1)
        )
    
    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute weights based on geometric features.
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            Weights tensor [num_models]
        """
        # Extract features
        if x.dim() > 2:
            features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        else:
            features = x
        
        # Extract geometric features
        geometric_features = self.geometric_extractor(features)
        
        # Predict weights
        weights = self.weight_predictor(geometric_features)
        
        # Average across batch dimension
        weights = weights.mean(dim=0)  # [num_models]
        
        return weights

class EnsembleManager:
    """
    Manager for multiple ensemble types.
    """
    
    def __init__(self):
        """Initialize ensemble manager."""
        self.ensembles = {}
        self.current_ensemble = None
    
    def add_ensemble(self, name: str, ensemble: EnsembleBase):
        """
        Add an ensemble to the manager.
        
        Args:
            name: Ensemble name
            ensemble: Ensemble instance
        """
        self.ensembles[name] = ensemble
        if self.current_ensemble is None:
            self.current_ensemble = name
    
    def set_current_ensemble(self, name: str):
        """
        Set the current ensemble.
        
        Args:
            name: Ensemble name
        """
        if name not in self.ensembles:
            raise ValueError(f"Ensemble '{name}' not found")
        self.current_ensemble = name
    
    def get_current_ensemble(self) -> Optional[EnsembleBase]:
        """Get the current ensemble."""
        if self.current_ensemble is None:
            return None
        return self.ensembles[self.current_ensemble]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make prediction using current ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        if self.current_ensemble is None:
            raise ValueError("No current ensemble set")
        
        ensemble = self.ensembles[self.current_ensemble]
        return ensemble(x)
    
    def get_ensemble_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get weights from all ensembles.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of ensemble weights
        """
        weights = {}
        for name, ensemble in self.ensembles.items():
            if hasattr(ensemble, 'compute_weights'):
                weights[name] = ensemble.compute_weights(x)
        
        return weights 