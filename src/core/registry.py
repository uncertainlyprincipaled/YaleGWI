import os
from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for managing model versions with geometric metadata.
    Tracks model versions, preserves equivariance properties, and handles initialization.
    """
    
    def __init__(self, registry_dir: str = "models"):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store model registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}}
    
    def _save_metadata(self):
        """Save current metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, 
                      model: nn.Module,
                      model_id: str,
                      family: str,
                      equivariance: List[str],
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new model version with geometric metadata.
        
        Args:
            model: PyTorch model to register
            model_id: Unique identifier for the model
            family: Geological family the model is trained on
            equivariance: List of geometric transformations the model is equivariant to
            metadata: Additional metadata to store
            
        Returns:
            str: Version ID of the registered model
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_id}_v{timestamp}"
        
        # Create model directory
        model_dir = self.registry_dir / version_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), model_dir / "model.pt")
        
        # Prepare metadata
        model_metadata = {
            "model_id": model_id,
            "version": version_id,
            "family": family,
            "equivariance": equivariance,
            "timestamp": timestamp,
            "architecture": str(model),
            "state_dict_keys": list(model.state_dict().keys()),
            **(metadata or {})
        }
        
        # Update registry
        self.metadata["models"][version_id] = model_metadata
        self._save_metadata()
        
        logger.info(f"Registered model {version_id} with {len(equivariance)} equivariance properties")
        return version_id
    
    def load_model(self, version_id: str, model_class: type) -> nn.Module:
        """
        Load a registered model version.
        
        Args:
            version_id: Version ID of the model to load
            model_class: PyTorch model class to instantiate
            
        Returns:
            nn.Module: Loaded model
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        model_dir = self.registry_dir / version_id
        state_dict = torch.load(model_dir / "model.pt")
        
        # Instantiate model and load state
        model = model_class()
        model.load_state_dict(state_dict)
        
        logger.info(f"Loaded model {version_id} with {len(self.metadata['models'][version_id]['equivariance'])} equivariance properties")
        return model
    
    def list_models(self, family: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered models, optionally filtered by family.
        
        Args:
            family: Optional family to filter by
            
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        models = self.metadata["models"]
        if family:
            return [m for m in models.values() if m["family"] == family]
        return list(models.values())
    
    def get_model_info(self, version_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model version.
        
        Args:
            version_id: Version ID of the model
            
        Returns:
            Dict[str, Any]: Model metadata
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        return self.metadata["models"][version_id]
    
    def update_metadata(self, version_id: str, metadata: Dict[str, Any]):
        """
        Update metadata for a specific model version.
        
        Args:
            version_id: Version ID of the model
            metadata: New metadata to add/update
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        self.metadata["models"][version_id].update(metadata)
        self._save_metadata()
        logger.info(f"Updated metadata for model {version_id}")
    
    def delete_model(self, version_id: str):
        """
        Delete a model version from the registry.
        
        Args:
            version_id: Version ID of the model to delete
        """
        if version_id not in self.metadata["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        # Remove model files
        model_dir = self.registry_dir / version_id
        if model_dir.exists():
            for file in model_dir.glob("*"):
                file.unlink()
            model_dir.rmdir()
        
        # Remove from metadata
        del self.metadata["models"][version_id]
        self._save_metadata()
        logger.info(f"Deleted model {version_id}") 