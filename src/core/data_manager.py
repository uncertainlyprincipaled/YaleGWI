"""
DataManager is the single source of truth for all data IO in this project.
All data loading, streaming, and batching must go through DataManager.
"""
# Standard library imports
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import os
import socket
import tempfile
import shutil
import mmap
import logging
import json

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Local imports
from src.core.config import CFG, FAMILY_FILE_MAP

class MemoryTracker:
    """Tracks memory usage during data loading."""
    def __init__(self):
        self.total_memory = 0
        self.max_memory = 0
        
    def update(self, memory_used: int):
        self.total_memory += memory_used
        self.max_memory = max(self.max_memory, memory_used)
        
    def get_stats(self) -> Dict[str, float]:
        return {
            'total_memory_gb': self.total_memory / 1e9,
            'max_memory_gb': self.max_memory / 1e9
        }

class DataManager:
    """
    Singleton class for managing all data IO operations.
    Ensures consistent data access and prevents memory leaks in Kaggle notebook environment.
    Handles both local and S3 data operations.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, data_root: str = None, use_s3: bool = False):
        """
        Initialize the data manager if not already initialized.
        
        Args:
            data_root: Root directory for data
            use_s3: Whether to use S3 for data operations
        """
        if self._initialized:
            return
            
        self.data_root = Path(data_root) if data_root else None
        self._cache = {}
        self.use_s3 = use_s3
        self.use_mmap = True
        self.memory_tracker = MemoryTracker()
        
        if use_s3:
            self._setup_s3()
            
        self._initialized = True
    
    def s3_download(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3 to a local path.
        
        Args:
            s3_key: The key of the object in the S3 bucket.
            local_path: The local path to save the file to.
            
        Returns:
            True if download was successful, False otherwise.
        """
        if not self.use_s3:
            raise RuntimeError("S3 operations not enabled")
        
        try:
            with open(local_path, 'wb') as f:
                self.s3.download_fileobj(self.s3_bucket, s3_key, f)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.error(f"S3 object not found: {s3_key}")
            else:
                logging.error(f"Failed to download {s3_key} from S3: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during S3 download: {e}")
            return False

    def _setup_s3(self):
        """Set up S3 client and configuration."""
        try:
            # Try to load credentials from .env/aws/credentials.json
            creds_path = Path('.env/aws/credentials.json')
            if creds_path.exists():
                with open(creds_path, 'r') as f:
                    credentials = json.load(f)
            else:
                # Fall back to environment variables
                credentials = {
                    'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
                    'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    'region_name': os.environ.get('AWS_REGION', 'us-east-1')
                }
            
            self.s3 = boto3.client('s3', **credentials)
            self.s3_bucket = credentials.get('s3_bucket') or os.environ.get('AWS_S3_BUCKET')
            self.cache_dir = Path(tempfile.gettempdir()) / 'gwi_cache'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.s3_bucket:
                raise ValueError("S3 bucket not specified in credentials or environment variable AWS_S3_BUCKET")
                
        except Exception as e:
            logging.error(f"Failed to set up S3: {e}")
            self.use_s3 = False
            raise
    
    def stream_from_s3(self, s3_key: str, chunk_size: int = 1024*1024) -> np.ndarray:
        """
        Stream a file from S3 in chunks with memory mapping.
        
        Args:
            s3_key: S3 object key
            chunk_size: Size of chunks to download
            
        Returns:
            np.ndarray: Memory-mapped array of the data
        """
        if not self.use_s3:
            raise RuntimeError("S3 streaming not enabled")
            
        try:
            # Get object size
            response = self.s3.head_object(Bucket=self.s3_bucket, Key=s3_key)
            file_size = response['ContentLength']
            
            # Create memory-mapped array
            temp_file = self.cache_dir / s3_key.replace('/', '_')
            if not temp_file.exists():
                with open(temp_file, 'wb') as f:
                    for i in range(0, file_size, chunk_size):
                        end = min(i + chunk_size, file_size)
                        response = self.s3.get_object(
                            Bucket=self.s3_bucket,
                            Key=s3_key,
                            Range=f'bytes={i}-{end-1}'
                        )
                        f.write(response['Body'].read())
            
            return np.load(temp_file, mmap_mode='r')
            
        except ClientError as e:
            logging.error(f"Error streaming file {s3_key}: {e}")
            raise
    
    def upload_to_s3(self, data: np.ndarray, s3_key: str) -> bool:
        """
        Upload processed data to S3.
        
        Args:
            data: Data to upload
            s3_key: S3 object key
            
        Returns:
            bool: True if upload successful
        """
        if not self.use_s3:
            raise RuntimeError("S3 upload not enabled")
            
        try:
            # Save to temporary file
            temp_file = self.cache_dir / f"temp_{s3_key.replace('/', '_')}"
            np.save(temp_file, data)
            
            # Upload to S3
            self.s3.upload_file(
                str(temp_file),
                self.s3_bucket,
                s3_key
            )
            
            # Clean up
            temp_file.unlink()
            return True
            
        except Exception as e:
            logging.error(f"Error uploading to S3: {e}")
            return False
    
    def list_s3_files(self, prefix: str) -> List[str]:
        """
        List files in S3 bucket with given prefix.
        
        Args:
            prefix: S3 key prefix
            
        Returns:
            List[str]: List of S3 keys
        """
        if not self.use_s3:
            raise RuntimeError("S3 operations not enabled")
            
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix,
                MaxKeys=1000
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logging.error(f"Error listing S3 objects: {e}")
            return []
    
    def list_family_files(self, family: str) -> Tuple[List[Path], List[Path], str]:
        """
        Return (seis_files, vel_files, family_type) for a given family.
        Handles both local and S3 data sources.
        """
        if CFG.debug_mode and family in CFG.families_to_exclude:
            logging.info(f"Skipping excluded family in debug mode: {family}")
            return None, None, None
        
        if self.use_s3:
            # Use S3Paths config and FAMILY_FILE_MAP for all S3 prefixes
            s3_family_prefix = CFG.s3_paths.families[family]
            info = FAMILY_FILE_MAP[family]
            # Vel/Style: data/model subfolders (batched)
            seis_prefix = f"{s3_family_prefix}/{info['seis_dir']}/" if info['seis_dir'] else f"{s3_family_prefix}/"
            vel_prefix = f"{s3_family_prefix}/{info['vel_dir']}/" if info['vel_dir'] else f"{s3_family_prefix}/"
            seis_files = self.list_s3_files(seis_prefix)
            vel_files = self.list_s3_files(vel_prefix)
            if seis_files and vel_files:
                if CFG.debug_mode:
                    seis_files = seis_files[:1]
                    vel_files = vel_files[:1]
                return seis_files, vel_files, 'VelStyle' if info['seis_dir'] else 'Fault'
        else:
            # Original local file handling
            root = CFG.paths.families[family]
            info = FAMILY_FILE_MAP[family]
            if not root.exists():
                raise ValueError(f"Family directory not found: {root}")
            # Vel/Style: data/model subfolders (batched)
            if info['seis_dir'] and info['vel_dir']:
                seis_files = sorted((root/info['seis_dir']).glob('*.npy'))
                vel_files = sorted((root/info['vel_dir']).glob('*.npy'))
                if seis_files and vel_files:
                    if CFG.debug_mode:
                        seis_files = seis_files[:1]
                        vel_files = vel_files[:1]
                    return seis_files, vel_files, 'VelStyle'
            # Fault: seis*.npy and vel*.npy directly in folder
            seis_files = sorted(root.glob('seis*.npy'))
            vel_files = sorted(root.glob('vel*.npy'))
            if seis_files and vel_files:
                if CFG.debug_mode:
                    seis_files = seis_files[:1]
                    vel_files = vel_files[:1]
                return seis_files, vel_files, 'Fault'
        raise ValueError(f"Could not find valid data structure for family {family}")

    def create_dataset(self, seis_files: Union[List[Path], List[str]], 
                      vel_files: Union[List[Path], List[str]], 
                      family_type: str, 
                      augment: bool = False) -> Dataset:
        """Create a dataset for the given files."""
        if family_type == 'test':
            return TestDataset(seis_files, self)
        return SeismicDataset(
            seis_files,
            vel_files,
            family_type,
            augment,
            use_mmap=self.use_mmap,
            memory_tracker=self.memory_tracker,
            data_manager=self
        )

    def create_loader(self, seis_files: List[Path], vel_files: List[Path],
                     family_type: str, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 4,
                     distributed: bool = False) -> Optional[DataLoader]:
        """Create a DataLoader for the given files. Returns None if files are empty."""
        if not seis_files:
            logging.info("No files to load, skipping loader creation")
            return None
            
        # For test data, vel_files can be None
        if family_type != 'test' and not vel_files:
            logging.info("No velocity files to load, skipping loader creation")
            return None
            
        dataset = self.create_dataset(seis_files, vel_files, family_type)
        
        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and not distributed,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )

    def get_test_files(self) -> List[Path]:
        return sorted(CFG.paths.test.glob('*.npy'))

    def upload_best_model_and_metadata(self, artefact_dir: Path, message: str = "Update best model"):
        logging.info(f"Starting upload_best_model_and_metadata with artefact_dir={artefact_dir}, message={message}")
        if CFG.debug_mode:
            logging.info("Debug mode: Simulating model upload")
            logging.info(f"Would upload from: {artefact_dir}")
            logging.info(f"With message: {message}")
            return
        if CFG.env.kind == "kaggle":
            if not artefact_dir.exists():
                logging.error(f"Artefact directory not found: {artefact_dir}")
                return
            logging.info("Calling push_to_kaggle for model upload...")
            push_to_kaggle(artefact_dir, message)
            logging.info(f"Successfully uploaded to Kaggle dataset with message: {message}")
            return
        if CFG.env.kind == "aws":
            logging.info("AWS environment stub - not implemented.")
            return
        if CFG.env.kind == "colab":
            logging.info("Colab environment stub - not implemented.")
            return
        if CFG.env.kind == "sagemaker":
            logging.info("SageMaker environment stub - not implemented.")
            return
        logging.info("Unknown environment stub - not implemented.")
        return

class SeismicDataset(Dataset):
    """
    Memory-efficient dataset for all families.
    Handles both local and S3 data sources.
    """
    def __init__(self, seis_files: Union[List[Path], List[str]], 
                 vel_files: Optional[Union[List[Path], List[str]]], 
                 family_type: str, 
                 augment: bool = False, 
                 use_mmap: bool = True, 
                 memory_tracker: MemoryTracker = None,
                 data_manager: DataManager = None):
        self.family_type = family_type
        self.augment = augment
        self.index = []
        self.use_mmap = use_mmap
        self.memory_tracker = memory_tracker
        self.vel_files = vel_files
        self.data_manager = data_manager
        
        # Build index of (file, sample_idx) pairs
        if vel_files is None:
            for sfile in seis_files:
                if self.data_manager and self.data_manager.use_s3:
                    # For S3 files, we need to get the shape without loading
                    response = self.data_manager.s3.head_object(
                        Bucket=self.data_manager.s3_bucket,
                        Key=sfile
                    )
                    # Assuming the first dimension is batch size
                    n_samples = 500  # Default for batched files
                else:
                    if self.use_mmap:
                        f = np.load(sfile, mmap_mode='r')
                        shape = f.shape
                        n_samples = shape[0] if len(shape) == 4 else 1
                        del f
                    else:
                        data = np.load(sfile)
                        shape = data.shape
                        n_samples = shape[0] if len(shape) == 4 else 1
                for i in range(n_samples):
                    self.index.append((sfile, None, i))
        else:
            for sfile, vfile in zip(seis_files, vel_files):
                if self.data_manager and self.data_manager.use_s3:
                    n_samples = 500  # Default for batched files
                else:
                    if self.use_mmap:
                        f = np.load(sfile, mmap_mode='r')
                        shape = f.shape
                        n_samples = shape[0] if len(shape) == 4 else 1
                        del f
                    else:
                        data = np.load(sfile)
                        shape = data.shape
                        n_samples = shape[0] if len(shape) == 4 else 1
                for i in range(n_samples):
                    self.index.append((sfile, vfile, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sfile, vfile, i = self.index[idx]
        
        # Load seismic data
        if self.data_manager and self.data_manager.use_s3:
            x = self.data_manager.stream_from_s3(sfile)
        else:
            if self.use_mmap:
                x = np.load(sfile, mmap_mode='r')
            else:
                x = np.load(sfile)
                
        # Handle data shapes
        if len(x.shape) == 4:
            x = x[i]
        if x.shape[1] == 70 and x.shape[2] == 1000:
            x = x.transpose(0, 2, 1)
            
        # Load velocity data
        if self.vel_files is None:
            y = np.zeros((1, 70, 70), np.float16)
        else:
            if self.data_manager and self.data_manager.use_s3:
                y = self.data_manager.stream_from_s3(vfile)
            else:
                if self.use_mmap:
                    y = np.load(vfile, mmap_mode='r')
                else:
                    y = np.load(vfile)
            if len(y.shape) == 4:
                y = y[i]
            elif len(y.shape) == 3:
                y = y[i]
                
        # Convert to float16 and normalize
        x = x.astype(np.float16)
        y = y.astype(np.float16)
        mu = x.mean(axis=(1,2), keepdims=True)
        std = x.std(axis=(1,2), keepdims=True) + 1e-6
        x = (x - mu) / std
        
        # Track memory usage
        if self.memory_tracker:
            self.memory_tracker.update(x.nbytes + y.nbytes)
            
        return torch.from_numpy(x), torch.from_numpy(y)

class TestDataset(Dataset):
    """Dataset for test files that returns the test data and its identifier."""
    def __init__(self, files: Union[List[Path], List[str]], data_manager: DataManager = None):
        self.files = files
        self.data_manager = data_manager

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        
        if self.data_manager and self.data_manager.use_s3:
            data = self.data_manager.stream_from_s3(file_path)
        else:
            data = np.load(file_path)
            
        # Convert to float16 and normalize
        data = data.astype(np.float16)
        mu = data.mean(axis=(1,2), keepdims=True)
        std = data.std(axis=(1,2), keepdims=True) + 1e-6
        data = (data - mu) / std
        
        return torch.from_numpy(data), Path(file_path).stem 