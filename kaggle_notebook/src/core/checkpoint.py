import os
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Singleton class for managing model checkpoints with geometric metadata.
    Ensures only one checkpoint manager exists across the application.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint manager if not already initialized.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        if self._initialized:
            return
            
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        self._initialized = True
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"checkpoints": {}}
    
    def _save_metadata(self):
        """Save current metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       model_id: str,
                       family: str,
                       equivariance: List[str],
                       metrics: Dict[str, float],
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model checkpoint with geometric metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current training epoch
            model_id: Unique identifier for the model
            family: Geological family being trained
            equivariance: List of geometric transformations the model is equivariant to
            metrics: Dictionary of training metrics
            metadata: Additional metadata to store
            
        Returns:
            str: Checkpoint ID
        """
        # Generate checkpoint ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{model_id}_epoch{epoch}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, checkpoint_path / "checkpoint.pt")
        
        # Prepare metadata
        checkpoint_metadata = {
            "checkpoint_id": checkpoint_id,
            "model_id": model_id,
            "family": family,
            "epoch": epoch,
            "equivariance": equivariance,
            "timestamp": timestamp,
            "metrics": metrics,
            **(metadata or {})
        }
        
        # Update registry
        self.metadata["checkpoints"][checkpoint_id] = checkpoint_metadata
        self._save_metadata()
        
        logger.info(f"Saved checkpoint {checkpoint_id} at epoch {epoch}")
        return checkpoint_id
    
    def load_checkpoint(self,
                       checkpoint_id: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, Dict[str, float]]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            model: PyTorch model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Tuple[int, Dict[str, float]]: (epoch, metrics)
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint {checkpoint_id} from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def list_checkpoints(self,
                        model_id: Optional[str] = None,
                        family: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints, optionally filtered by model or family.
        
        Args:
            model_id: Optional model ID to filter by
            family: Optional family to filter by
            
        Returns:
            List[Dict[str, Any]]: List of checkpoint metadata
        """
        checkpoints = self.metadata["checkpoints"]
        filtered = checkpoints.values()
        
        if model_id:
            filtered = [c for c in filtered if c["model_id"] == model_id]
        if family:
            filtered = [c for c in filtered if c["family"] == family]
            
        return sorted(filtered, key=lambda x: x["epoch"])
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint
            
        Returns:
            Dict[str, Any]: Checkpoint metadata
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        return self.metadata["checkpoints"][checkpoint_id]
    
    def update_metadata(self, checkpoint_id: str, metadata: Dict[str, Any]):
        """
        Update metadata for a specific checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint
            metadata: New metadata to add/update
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        self.metadata["checkpoints"][checkpoint_id].update(metadata)
        self._save_metadata()
        logger.info(f"Updated metadata for checkpoint {checkpoint_id}")
    
    def delete_checkpoint(self, checkpoint_id: str):
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Remove checkpoint files
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        
        # Remove from metadata
        del self.metadata["checkpoints"][checkpoint_id]
        self._save_metadata()
        logger.info(f"Deleted checkpoint {checkpoint_id}")
    
    def get_best_checkpoint(self,
                          model_id: str,
                          metric: str = "val_loss",
                          family: Optional[str] = None) -> Optional[str]:
        """
        Get the best checkpoint based on a metric.
        
        Args:
            model_id: Model ID to search for
            metric: Metric to optimize (default: val_loss)
            family: Optional family to filter by
            
        Returns:
            Optional[str]: ID of the best checkpoint, or None if no checkpoints found
        """
        checkpoints = self.list_checkpoints(model_id, family)
        if not checkpoints:
            return None
            
        # Sort by metric (lower is better for loss metrics)
        is_loss = "loss" in metric.lower()
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda x: x["metrics"].get(metric, float('inf') if is_loss else float('-inf')),
            reverse=not is_loss
        )
        
        return sorted_checkpoints[0]["checkpoint_id"] 