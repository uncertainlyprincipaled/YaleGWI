"""
Checkpoint manager for saving and loading model states with geometric metadata.
"""
import os
from typing import Dict, Any, Tuple, Optional
import torch

from .model_registry import ModelRegistry

class CheckpointManager:
    """
    Saves and loads model checkpoints while preserving geometric metadata.
    
    Attributes:
        registry (ModelRegistry): Registry containing model configurations and properties
        checkpoint_dir (str): Directory to store checkpoints
    """
    
    def __init__(self, registry: ModelRegistry, checkpoint_dir: str):
        """
        Initialize checkpoint manager.
        
        Args:
            registry: Model registry instance
            checkpoint_dir: Directory to store checkpoints
        """
        self.registry = registry
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model_id: str, state: Dict[str, Any], epoch: int) -> str:
        """
        Save model state dict + metadata to a file.
        
        Args:
            model_id: Unique identifier for the model
            state: Model state dictionary
            epoch: Current training epoch
            
        Returns:
            Path to saved checkpoint
        """
        meta = {
            'model_id': model_id,
            'geom_props': self.registry.get_geometric_properties(model_id),
            'epoch': epoch
        }
        
        path = os.path.join(self.checkpoint_dir, f"{model_id}_epoch{epoch}.pth")
        torch.save({'state': state, 'meta': meta}, path)
        return path
    
    def load_checkpoint(self, path: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Load checkpoint and validate geometric metadata.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Tuple of (model_id, state_dict, meta)
            
        Raises:
            AssertionError: If geometric metadata doesn't match registry
        """
        loaded = torch.load(path)
        meta = loaded['meta']
        model_id = meta['model_id']
        
        # Validate geometric properties match registry
        expected = self.registry.get_geometric_properties(model_id)
        assert meta['geom_props'] == expected, "Geometric metadata mismatch"
        
        return model_id, loaded['state'], meta 