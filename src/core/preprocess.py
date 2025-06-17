"""
Seismic Data Preprocessing Pipeline

This module implements a comprehensive preprocessing pipeline for seismic data, focusing on:
1. Data downsampling while preserving signal integrity (Nyquist-Shannon theorem)
2. Memory-efficient processing using memory mapping and chunked operations
3. Distributed storage using Zarr and S3
4. GPU-optimized data splitting

Key Concepts:
- Nyquist-Shannon Theorem: Ensures we don't lose information during downsampling
- Memory Mapping: Allows processing large files without loading them entirely into memory
- Chunked Processing: Enables parallel processing and efficient memory usage
- Zarr Storage: Provides efficient compression and chunked storage for large datasets
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
# For Kaggle/Colab, install zarr
# !pip install zarr
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.signal import decimate
import logging
from typing import Tuple, List, Optional, Dict, Any
import warnings
import boto3
from botocore.exceptions import ClientError
import json
from src.core.config import CFG, FAMILY_FILE_MAP
import tempfile
from src.core.data_manager import DataManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for preprocessing
CHUNK_TIME = 256  # After decimating by 4 - optimized for GPU memory
CHUNK_SRC_REC = 8  # Chunk size for source-receiver dimensions
DT_DECIMATE = 4  # 1 kHz → 250 Hz - reduces data size while preserving signal
NYQUIST_FREQ = 500  # Hz (half of original sampling rate) - critical for downsampling

def validate_nyquist(data: np.ndarray, original_fs: int = 1000) -> bool:
    """
    Validate that the data satisfies Nyquist criterion after downsampling.
    
    Intuition:
    - Nyquist-Shannon Theorem: Sampling rate must be > 2x highest frequency
    - Energy Analysis: Check if significant energy exists above Nyquist frequency
    - Safety Margin: Warn if >1% of energy is above Nyquist (potential aliasing)
    
    Args:
        data: Input seismic data array
        original_fs: Original sampling frequency in Hz
        
    Returns:
        bool: True if data satisfies Nyquist criterion
    """
    # Compute FFT
    fft_data = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=1/original_fs)
    
    # Check if significant energy exists above Nyquist frequency
    nyquist_mask = freqs > (original_fs / (2 * DT_DECIMATE))
    high_freq_energy = np.abs(fft_data[:, nyquist_mask]).mean()
    total_energy = np.abs(fft_data).mean()
    
    # If more than 1% of energy is above Nyquist, warn
    if high_freq_energy / total_energy > 0.01:
        warnings.warn(f"Significant energy above Nyquist frequency detected: {high_freq_energy/total_energy:.2%}")
        return False
    return True

def preprocess_one(arr: np.ndarray) -> np.ndarray:
    """
    Preprocess a single seismic array with downsampling and normalization.
    
    Intuition:
    - Downsampling: Reduce data size while preserving signal integrity
    - Anti-aliasing: Prevent frequency folding during downsampling
    - Memory Efficiency: Use float16 for reduced memory footprint
    - Robust Normalization: Handle outliers using percentiles
    
    Processing Steps:
    1. Validate Nyquist criterion
    2. Apply anti-aliasing filter and downsample
    3. Convert to float16
    4. Normalize using robust statistics
    
    Args:
        arr: Input seismic array
        
    Returns:
        np.ndarray: Preprocessed array
    """
    try:
        # Validate Nyquist criterion
        if not validate_nyquist(arr):
            logger.warning("Data may violate Nyquist criterion after downsampling")
        
        # Decimate time axis with anti-aliasing filter
        arr = decimate(arr, DT_DECIMATE, axis=1, ftype='fir')
        
        # Convert to float16
        arr = arr.astype('float16')
        
        # Robust normalization per trace
        μ = np.median(arr, keepdims=True)
        σ = np.percentile(arr, 95, keepdims=True) - np.percentile(arr, 5, keepdims=True)
        arr = (arr - μ) / (σ + 1e-8)  # Add small epsilon to avoid division by zero
        
        return arr
    except Exception as e:
        logger.error(f"Error preprocessing array: {str(e)}")
        raise

def process_family(family: str, input_dir: Path, output_dir: Path, data_manager: Optional[DataManager] = None) -> List[str]:
    """
    Process all files in a family and return paths to processed files.
    
    Args:
        family: Name of the geological family
        input_dir: Input directory containing raw data
        output_dir: Output directory for processed data
        data_manager: Optional DataManager for S3 operations
        
    Returns:
        List[str]: Paths to processed files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_paths = []
    
    if data_manager and data_manager.use_s3:
        # Process files from S3
        s3_prefix = f"raw/train_samples/{family}/"
        s3_files = data_manager.list_s3_files(s3_prefix)
        
        for s3_key in tqdm(s3_files, desc=f"Processing {family} from S3"):
            try:
                arr = data_manager.stream_from_s3(s3_key)
                if arr is not None:
                    if arr.ndim == 4:
                        n_samples = arr.shape[0]
                        for i in range(n_samples):
                            sample = arr[i]
                            processed = preprocess_one(sample)
                            out_key = f"preprocessed/{family}/sample_{len(processed_paths):06d}.npy"
                            if data_manager.upload_to_s3(processed, out_key):
                                processed_paths.append(out_key)
                    elif arr.ndim == 3:
                        processed = preprocess_one(arr)
                        out_key = f"preprocessed/{family}/sample_{len(processed_paths):06d}.npy"
                        if data_manager.upload_to_s3(processed, out_key):
                            processed_paths.append(out_key)
            except Exception as e:
                logger.error(f"Error processing S3 file {s3_key}: {str(e)}")
                continue
    else:
        # Process local files
        files = sorted(input_dir.glob('*.npy'))
        for f in tqdm(files, desc=f"Processing {family}"):
            try:
                arr = np.load(f, mmap_mode='r')
                if arr.ndim == 4:
                    n_samples = arr.shape[0]
                    for i in range(n_samples):
                        sample = arr[i]
                        processed = preprocess_one(sample)
                        out_path = output_dir / f"sample_{len(processed_paths):06d}.npy"
                        np.save(out_path, processed)
                        processed_paths.append(str(out_path))
                elif arr.ndim == 3:
                    processed = preprocess_one(arr)
                    out_path = output_dir / f"sample_{len(processed_paths):06d}.npy"
                    np.save(out_path, processed)
                    processed_paths.append(str(out_path))
                else:
                    logger.warning(f"Unexpected shape {arr.shape} in {f}")
            except Exception as e:
                logger.error(f"Error processing file {f}: {str(e)}")
                continue
    
    return processed_paths

def create_zarr_dataset(processed_paths: List[str], output_path: Path, chunk_size: Tuple[int, ...], data_manager: Optional[DataManager] = None) -> None:
    """
    Create a zarr dataset from processed files and optionally upload to S3.
    
    Intuition:
    - Lazy Loading: Use Dask for out-of-memory computations
    - Chunked Storage: Enable parallel access and efficient compression
    - Cloud Storage: Optional S3 upload for distributed access
    - Memory Management: Clean up local files after successful upload
    
    Args:
        processed_paths: List of paths to processed files
        output_path: Path to save zarr dataset
        chunk_size: Chunk size for zarr array
        data_manager: Optional DataManager for S3 operations
    """
    try:
        # Dynamically determine shape from first file
        if not processed_paths:
            raise ValueError("No processed paths provided.")
            
        if data_manager and data_manager.use_s3:
            first_arr = data_manager.stream_from_s3(processed_paths[0])
        else:
            first_arr = np.load(processed_paths[0], mmap_mode='r')
            
        arr_shape = first_arr.shape
        arr_dtype = first_arr.dtype
        
        # Validate expected dimensions
        if len(arr_shape) == 4:  # Seismic data
            expected_shape = (500, 5, 1000, 70)
            if arr_shape != expected_shape:
                logger.warning(f"Unexpected seismic data shape: {arr_shape}, expected {expected_shape}")
        elif len(arr_shape) == 4 and arr_shape[1] == 1:  # Velocity model
            expected_shape = (500, 1, 70, 70)
            if arr_shape != expected_shape:
                logger.warning(f"Unexpected velocity model shape: {arr_shape}, expected {expected_shape}")
        else:
            raise ValueError(f"Unexpected array shape: {arr_shape}")
            
        # Create lazy Dask arrays
        lazy_arrays = []
        for path in processed_paths:
            if data_manager and data_manager.use_s3:
                x = da.from_delayed(
                    dask.delayed(data_manager.stream_from_s3)(path),
                    shape=arr_shape,
                    dtype=arr_dtype
                )
            else:
                x = da.from_delayed(
                    dask.delayed(np.load)(path),
                    shape=arr_shape,
                    dtype=arr_dtype
                )
            lazy_arrays.append(x)
            
        # Stack arrays
        stack = da.stack(lazy_arrays, axis=0)
        
        # Save to zarr with compression
        stack.to_zarr(
            output_path,
            component='seis',
            compressor=zarr.Blosc(cname='zstd', clevel=3),
            chunks=chunk_size
        )
        
        # Upload to S3 if using S3
        if data_manager and data_manager.use_s3:
            s3_key = f"preprocessed/{output_path.name}"
            if data_manager.upload_to_s3(np.load(output_path), s3_key):
                # Clean up local file after successful upload
                if output_path.exists():
                    output_path.unlink()
                    logger.info(f"Cleaned up local file {output_path}")
                
    except Exception as e:
        logger.error(f"Error creating/uploading zarr dataset: {str(e)}")
        raise

def split_for_gpus(processed_paths: List[str], output_base: Path, data_manager: Optional[DataManager] = None, by_family: bool = True) -> None:
    """
    Split processed files into two datasets for the two T4 GPUs and optionally upload to S3.
    If by_family is True, split by family (half families to each GPU), else split within families.
    Ensure all data is downsampled to float16.
    """
    try:
        if by_family:
            # Group processed_paths by family
            family_groups = {}
            for path in processed_paths:
                # Assume path contains family name as a parent directory
                family = Path(path).parent.parent.name if Path(path).parent.name in ['data', 'model'] else Path(path).parent.name
                family_groups.setdefault(family, []).append(path)
            families = sorted(family_groups.keys())
            mid = len(families) // 2
            gpu0_fams = families[:mid]
            gpu1_fams = families[mid:]
            gpu0_paths = [p for fam in gpu0_fams for p in family_groups[fam]]
            gpu1_paths = [p for fam in gpu1_fams for p in family_groups[fam]]
        else:
            n_samples = len(processed_paths)
            mid_point = n_samples // 2
            gpu0_paths = processed_paths[:mid_point]
            gpu1_paths = processed_paths[mid_point:]
            
        # Create GPU-specific directories
        gpu0_dir = output_base / 'gpu0'
        gpu1_dir = output_base / 'gpu1'
        gpu0_dir.mkdir(parents=True, exist_ok=True)
        gpu1_dir.mkdir(parents=True, exist_ok=True)
        
        # Create zarr datasets for each GPU
        create_zarr_dataset(
            gpu0_paths,
            gpu0_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            data_manager
        )
        create_zarr_dataset(
            gpu1_paths,
            gpu1_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            data_manager
        )
        logger.info(f"Created GPU datasets with {len(gpu0_paths)} and {len(gpu1_paths)} samples")
    except Exception as e:
        logger.error(f"Error splitting data for GPUs: {str(e)}")
        raise

def main():
    """
    Main preprocessing pipeline.
    
    Intuition:
    - Command Line Interface: Flexible configuration
    - Family Processing: Handle different geological families
    - GPU Optimization: Split data for parallel processing
    - Cloud Integration: Optional S3 upload
    - Error Handling: Robust error reporting and logging
    """
    # Enable debug mode for testing
    os.environ['DEBUG_MODE'] = '1'
    
    # Filter out Jupyter/Colab specific arguments
    import sys
    filtered_args = [arg for arg in sys.argv if not arg.startswith('-f') and not arg.endswith('.json')]
    sys.argv = filtered_args

    parser = argparse.ArgumentParser(description="Preprocess seismic data for distributed training on T4 GPUs")
    parser.add_argument('--input_root', type=str, default=str(CFG.paths.train), help='Input train_samples root directory')
    parser.add_argument('--output_root', type=str, default='/kaggle/working/preprocessed', help='Output directory for processed files')
    parser.add_argument('--use_s3', action='store_true', help='Use S3 for data processing')
    args = parser.parse_args()

    try:
        input_root = Path(args.input_root)
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        # Initialize DataManager with S3 support if requested
        data_manager = DataManager(use_s3=args.use_s3) if args.use_s3 else None

        # Process each family
        families = list(CFG.paths.families.keys())
        all_processed_paths = []
        
        for family in families:
            logger.info(f"\nProcessing family: {family}")
            input_dir = input_root / family
            temp_dir = output_root / 'temp' / family
            processed_paths = process_family(family, input_dir, temp_dir, data_manager)
            all_processed_paths.extend(processed_paths)
            logger.info(f"Family {family}: {len(processed_paths)} samples processed")

        # Split and create zarr datasets for GPUs
        logger.info("\nCreating GPU-specific datasets...")
        split_for_gpus(all_processed_paths, output_root, data_manager)
        
        # Clean up temporary files
        temp_dir = output_root / 'temp'
        if temp_dir.exists():
            subprocess.run(['rm', '-rf', str(temp_dir)])
        
        logger.info("\nPreprocessing complete!")
        if data_manager and data_manager.use_s3:
            logger.info(f"Data uploaded to s3://{data_manager.s3_bucket}/preprocessed/")
    except Exception as e:
        logger.error(f"Error in main preprocessing: {str(e)}")
        raise

def load_data(input_root, output_root, use_s3=False):
    """
    High-level entry point for preprocessing pipeline. Sets up DataManager, processes all families, and splits for GPUs.
    Args:
        input_root (str or Path): Root directory for input data
        output_root (str or Path): Directory to write processed data
        use_s3 (bool): Whether to use S3 for IO
    Returns:
        List[str]: All processed file paths
    """
    from pathlib import Path
    import subprocess
    from src.core.data_manager import DataManager
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_manager = DataManager(use_s3=use_s3) if use_s3 else None

    from src.core.config import CFG
    families = list(CFG.paths.families.keys())
    all_processed_paths = []
    for family in families:
        print(f"Processing family: {family}")
        input_dir = input_root / family
        temp_dir = output_root / 'temp' / family
        processed_paths = process_family(family, input_dir, temp_dir, data_manager)
        all_processed_paths.extend(processed_paths)
        print(f"Family {family}: {len(processed_paths)} samples processed")
    print("\nCreating GPU-specific datasets...")
    split_for_gpus(all_processed_paths, output_root, data_manager)
    # Clean up temporary files
    temp_dir = output_root / 'temp'
    if temp_dir.exists():
        subprocess.run(['rm', '-rf', str(temp_dir)])
    print("\nPreprocessing complete!")
    if data_manager and data_manager.use_s3:
        print(f"Data uploaded to s3://{data_manager.s3_bucket}/preprocessed/")
    return all_processed_paths

if __name__ == "__main__":
    main() 