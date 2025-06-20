import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
import json

# Optional MLflow import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

logger = logging.getLogger(__name__)

class GeometricCrossValidator:
    """
    Singleton class for implementing geometric-aware cross-validation.
    Ensures consistent validation strategy across training.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize the cross-validator if not already initialized.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed for reproducibility
        """
        if self._initialized:
            return
            
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._initialized = True
    
    def set_parameters(self, n_splits: int = None, shuffle: bool = None, random_state: int = None):
        """Update parameters if needed after initialization"""
        if n_splits is not None:
            self.n_splits = n_splits
        if shuffle is not None:
            self.shuffle = shuffle
        if random_state is not None:
            self.random_state = random_state
        # Reinitialize kfold objects with new parameters
        self.kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        self.stratified_kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
    
    def compute_geometric_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute geometric metrics between true and predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of geometric metrics
        """
        metrics = {}
        
        # Compute SSIM
        metrics['ssim'] = ssim(y_true, y_pred, data_range=y_true.max() - y_true.min())
        
        # Compute gradient magnitude similarity
        grad_true = np.gradient(y_true)[0]
        grad_pred = np.gradient(y_pred)[0]
        metrics['gradient_similarity'] = ssim(grad_true, grad_pred, data_range=grad_true.max() - grad_true.min())
        
        # Compute boundary preservation
        edges_true = canny(y_true, sigma=2.0)
        edges_pred = canny(y_pred, sigma=2.0)
        metrics['boundary_iou'] = np.logical_and(edges_true, edges_pred).sum() / np.logical_or(edges_true, edges_pred).sum()
        
        return metrics
    
    def split_by_family(self,
                       dataset: Dataset,
                       family_labels: List[str]) -> List[Tuple[Subset, Subset]]:
        """
        Split dataset by geological family.
        
        Args:
            dataset: PyTorch dataset
            family_labels: List of family labels for each sample
            
        Returns:
            List[Tuple[Subset, Subset]]: List of (train, val) splits
        """
        splits = []
        
        # Get unique families
        families = np.unique(family_labels)
        
        for family in families:
            # Get indices for this family
            family_indices = np.where(np.array(family_labels) == family)[0]
            
            # Split indices
            for train_idx, val_idx in self.kfold.split(family_indices):
                train_subset = Subset(dataset, family_indices[train_idx])
                val_subset = Subset(dataset, family_indices[val_idx])
                splits.append((train_subset, val_subset))
        
        return splits
    
    def split_by_geometry(self,
                         dataset: Dataset,
                         geometric_features: Dict[str, np.ndarray]) -> List[Tuple[Subset, Subset]]:
        """
        Split dataset by geometric features.
        
        Args:
            dataset: PyTorch dataset
            geometric_features: Dictionary of geometric features
            
        Returns:
            List[Tuple[Subset, Subset]]: List of (train, val) splits
        """
        splits = []
        
        # Combine geometric features into a single feature vector
        feature_matrix = np.column_stack([
            features.reshape(len(dataset), -1)
            for features in geometric_features.values()
        ])
        
        # Use stratified k-fold on geometric features
        for train_idx, val_idx in self.stratified_kfold.split(feature_matrix, np.zeros(len(dataset))):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            splits.append((train_subset, val_subset))
        
        return splits
    
    def log_geometric_metrics_mlflow(self, metrics: dict, prefix: str = ""):
        """Utility to log geometric metrics to MLflow if available."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}{k}", v)
        except Exception:
            pass
    
    def evaluate_fold(self,
                     model: torch.nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader,
                     device: torch.device) -> Dict[str, float]:
        """
        Evaluate a single fold.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to run evaluation on
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        model.eval()
        metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_ssim': 0.0,
            'val_ssim': 0.0,
            'train_boundary_iou': 0.0,
            'val_boundary_iou': 0.0
        }
        
        # Evaluate training set
        with torch.no_grad():
            for batch in train_loader:
                data = batch['data'].to(device)
                target = batch['target'].to(device)
                output = model(data)
                
                # Compute geometric metrics
                geom_metrics = self.compute_geometric_metrics(
                    target.cpu().numpy(),
                    output.cpu().numpy()
                )
                
                metrics['train_ssim'] += geom_metrics['ssim']
                metrics['train_boundary_iou'] += geom_metrics['boundary_iou']
        
        # Evaluate validation set
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(device)
                target = batch['target'].to(device)
                output = model(data)
                
                # Compute geometric metrics
                geom_metrics = self.compute_geometric_metrics(
                    target.cpu().numpy(),
                    output.cpu().numpy()
                )
                
                metrics['val_ssim'] += geom_metrics['ssim']
                metrics['val_boundary_iou'] += geom_metrics['boundary_iou']
        
        # Average metrics
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        metrics['train_ssim'] /= n_train
        metrics['train_boundary_iou'] /= n_train
        metrics['val_ssim'] /= n_val
        metrics['val_boundary_iou'] /= n_val
        
        # Log to MLflow
        self.log_geometric_metrics_mlflow(metrics, prefix="fold_")
        
        return metrics
    
    def cross_validate(self,
                      model: torch.nn.Module,
                      dataset: Dataset,
                      family_labels: List[str],
                      geometric_features: Dict[str, np.ndarray],
                      batch_size: int = 32,
                      device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, List[float]]:
        """
        Perform cross-validation with both family and geometric stratification.
        
        Args:
            model: PyTorch model
            dataset: PyTorch dataset
            family_labels: List of family labels
            geometric_features: Dictionary of geometric features
            batch_size: Batch size for data loaders
            device: Device to run evaluation on
            
        Returns:
            Dict[str, List[float]]: Dictionary of metrics for each fold
        """
        # Get splits
        family_splits = self.split_by_family(dataset, family_labels)
        geometry_splits = self.split_by_geometry(dataset, geometric_features)
        
        # Combine splits
        splits = []
        for (train_fam, val_fam), (train_geom, val_geom) in zip(family_splits, geometry_splits):
            # Combine train and val sets
            train_indices = list(set(train_fam.indices) & set(train_geom.indices))
            val_indices = list(set(val_fam.indices) & set(val_geom.indices))
            
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            splits.append((train_subset, val_subset))
        
        # Evaluate each fold
        fold_metrics = []
        for i, (train_subset, val_subset) in enumerate(splits):
            logger.info(f"Evaluating fold {i+1}/{len(splits)}")
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Evaluate fold
            metrics = self.evaluate_fold(model, train_loader, val_loader, device)
            # Log fold metrics to MLflow
            self.log_geometric_metrics_mlflow(metrics, prefix=f"fold{i+1}_")
            fold_metrics.append(metrics)
        
        # Aggregate results
        results = {
            'train_ssim': [m['train_ssim'] for m in fold_metrics],
            'val_ssim': [m['val_ssim'] for m in fold_metrics],
            'train_boundary_iou': [m['train_boundary_iou'] for m in fold_metrics],
            'val_boundary_iou': [m['val_boundary_iou'] for m in fold_metrics]
        }
        
        # Add mean and std
        for metric in results:
            values = results[metric]
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        
        return results
    
    def save_results(self,
                    results: Dict[str, List[float]],
                    output_path: str):
        """
        Save cross-validation results to file.
        
        Args:
            results: Dictionary of results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved cross-validation results to {output_path}") 