import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

class GeometricDataset(Dataset):
    """
    Dataset class for handling seismic data with geometric features.
    Extracts and manages geometric features for different geological families.
    """
    
    def __init__(self,
                 data_path: str,
                 family: str,
                 transform: Optional[Any] = None,
                 extract_features: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the zarr dataset
            family: Geological family name
            transform: Optional data transformations
            extract_features: Whether to extract geometric features
        """
        self.data_path = Path(data_path)
        self.family = family
        self.transform = transform
        self.extract_features = extract_features
        
        # Load zarr array
        self.data = zarr.open(self.data_path / 'seis', mode='r')
        
        # Load family metadata if exists
        metadata_path = self.data_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def __len__(self) -> int:
        return len(self.data)
    
    def extract_geometric_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract geometric features from seismic data.
        
        Args:
            data: Seismic data array
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of geometric features
        """
        features = {}
        
        # Extract structural features
        features['gradient_magnitude'] = np.gradient(data)[0]  # Time gradient
        features['gradient_direction'] = np.arctan2(np.gradient(data)[1], np.gradient(data)[0])
        
        # Extract boundary features using Canny edge detection
        features['edges'] = canny(data, sigma=2.0)
        
        # Extract spectral features
        fft_data = np.fft.rfft(data, axis=0)
        features['spectral_energy'] = np.abs(fft_data)
        features['spectral_phase'] = np.angle(fft_data)
        
        # Extract multi-scale features using Gaussian blur
        features['gaussian_1'] = gaussian_filter(data, sigma=1.0)
        features['gaussian_2'] = gaussian_filter(data, sigma=2.0)
        features['gaussian_4'] = gaussian_filter(data, sigma=4.0)
        
        return features
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a data sample with geometric features.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing data and features
        """
        # Load seismic data
        data = self.data[idx]
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(data).float()
        
        # Extract geometric features if requested
        if self.extract_features:
            features = self.extract_geometric_features(data)
            feature_tensors = {
                k: torch.from_numpy(v).float()
                for k, v in features.items()
            }
        else:
            feature_tensors = {}
        
        # Apply transformations if any
        if self.transform is not None:
            data_tensor = self.transform(data_tensor)
            if self.extract_features:
                feature_tensors = {
                    k: self.transform(v)
                    for k, v in feature_tensors.items()
                }
        
        return {
            'data': data_tensor,
            'features': feature_tensors,
            'family': self.family,
            'index': idx
        }

class FamilyDataLoader:
    """
    Data loader for handling family-specific data loading with geometric features.
    Manages data loading for different geological families with proper batching.
    """
    
    def __init__(self,
                 data_root: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 transform: Optional[Any] = None,
                 extract_features: bool = True):
        """
        Initialize the data loader.
        
        Args:
            data_root: Root directory containing family datasets
            batch_size: Batch size for loading
            num_workers: Number of worker processes
            transform: Optional data transformations
            extract_features: Whether to extract geometric features
        """
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.extract_features = extract_features
        
        # Initialize datasets for each family
        self.datasets = {}
        self.loaders = {}
        
        # Find all family directories
        family_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        
        for family_dir in family_dirs:
            family = family_dir.name
            dataset = GeometricDataset(
                family_dir,
                family,
                transform=transform,
                extract_features=extract_features
            )
            self.datasets[family] = dataset
            
            # Create data loader
            self.loaders[family] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
    
    def get_loader(self, family: str) -> DataLoader:
        """
        Get data loader for a specific family.
        
        Args:
            family: Family name
            
        Returns:
            DataLoader: Data loader for the family
        """
        if family not in self.loaders:
            raise ValueError(f"Family {family} not found")
        return self.loaders[family]
    
    def get_dataset(self, family: str) -> GeometricDataset:
        """
        Get dataset for a specific family.
        
        Args:
            family: Family name
            
        Returns:
            GeometricDataset: Dataset for the family
        """
        if family not in self.datasets:
            raise ValueError(f"Family {family} not found")
        return self.datasets[family]
    
    def get_all_loaders(self) -> Dict[str, DataLoader]:
        """
        Get all data loaders.
        
        Returns:
            Dict[str, DataLoader]: Dictionary of all data loaders
        """
        return self.loaders
    
    def get_all_datasets(self) -> Dict[str, GeometricDataset]:
        """
        Get all datasets.
        
        Returns:
            Dict[str, GeometricDataset]: Dictionary of all datasets
        """
        return self.datasets
    
    def get_family_stats(self, family: str) -> Dict[str, float]:
        """
        Get statistics for a specific family.
        
        Args:
            family: Family name
            
        Returns:
            Dict[str, float]: Dictionary of statistics
        """
        dataset = self.get_dataset(family)
        data = dataset.data
        
        stats = {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'samples': len(dataset)
        }
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all families.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of statistics for each family
        """
        return {
            family: self.get_family_stats(family)
            for family in self.datasets.keys()
        } 