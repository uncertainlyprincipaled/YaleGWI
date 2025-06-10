"""
Model registry for managing model configurations and geometric properties.
"""
from typing import Dict, Any, Optional

class ModelRegistry:
    """
    Manages registration and retrieval of model configurations along with
    their geometric properties.
    
    Attributes:
        models (Dict[str, Dict]): Maps model_id to model configuration
        geom_props (Dict[str, Dict]): Maps model_id to geometric properties
    """
    
    def __init__(self):
        """Initialize empty registry dictionaries."""
        self.models: Dict[str, Dict[str, Any]] = {}
        self.geom_props: Dict[str, Dict[str, Any]] = {}
    
    def register_model(self, model_id: str, model_config: Dict[str, Any], 
                      geometric_properties: Dict[str, Any]) -> None:
        """
        Register a new model architecture with its geometric metadata.
        
        Args:
            model_id: Unique identifier for the model
            model_config: Dictionary containing model architecture configuration
            geometric_properties: Dictionary containing geometric properties
        """
        self.models[model_id] = model_config
        self.geom_props[model_id] = geometric_properties
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model configuration by ID.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Model configuration dictionary if found, None otherwise
        """
        return self.models.get(model_id)
    
    def get_geometric_properties(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve geometric metadata by model ID.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Geometric properties dictionary if found, None otherwise
        """
        return self.geom_props.get(model_id) 