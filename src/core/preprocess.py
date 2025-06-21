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
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.signal import decimate
import logging
from typing import Tuple, List, Optional, Dict, Any, Union
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
NYQUIST_FREQ = 500  # Hz (half of original sampling rate) - critical for downsampling

class PreprocessingFeedback:
    """A simple class to collect feedback during preprocessing."""
    def __init__(self):
        self.nyquist_warnings = 0
        self.arrays_processed = 0

    def add_nyquist_warning(self):
        self.nyquist_warnings += 1

    def increment_arrays_processed(self):
        self.arrays_processed += 1

    @property
    def warning_percentage(self) -> float:
        if self.arrays_processed == 0:
            return 0.0
        return (self.nyquist_warnings / self.arrays_processed) * 100

def verify_data_structure(data_root: Path) -> bool:
    """
    Verify that the data structure is correct before preprocessing.
    
    Args:
        data_root: Root directory containing the training data
        
    Returns:
        bool: True if data structure is valid
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        logger.error(f"Data root directory does not exist: {data_root}")
        return False
    
    # Expected families
    expected_families = [
        'FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B',
        'Style_A', 'Style_B', 'FlatFault_A', 'FlatFault_B',
        'CurveFault_A', 'CurveFault_B'
    ]
    
    print("Verifying data structure...")
    print(f"Data root: {data_root}")
    print()
    
    all_valid = True
    
    for family in expected_families:
        family_dir = data_root / family
        if not family_dir.exists():
            print(f"✗ {family}: Directory not found")
            all_valid = False
            continue
        
        # Check for .npy files
        npy_files = list(family_dir.glob('*.npy'))
        if not npy_files:
            print(f"✗ {family}: No .npy files found")
            all_valid = False
            continue
        
        print(f"✓ {family}: {len(npy_files)} files found")
        
        # Check first file structure
        try:
            sample_file = npy_files[0]
            sample_data = np.load(sample_file, mmap_mode='r')
            
            if sample_data.ndim == 4:
                print(f"  - Shape: {sample_data.shape} (batch, sources, time, receivers)")
                print(f"  - Dtype: {sample_data.dtype}")
                
                # Check expected dimensions
                if sample_data.shape[2] != 2000:  # Time dimension
                    print(f"  ⚠ Warning: Expected time dimension 2000, got {sample_data.shape[2]}")
                if sample_data.shape[3] != 70:  # Receiver dimension
                    print(f"  ⚠ Warning: Expected receiver dimension 70, got {sample_data.shape[3]}")
                    
            elif sample_data.ndim == 3:
                print(f"  - Shape: {sample_data.shape} (sources, time, receivers)")
                print(f"  - Dtype: {sample_data.dtype}")
                
                # Check expected dimensions
                if sample_data.shape[1] != 2000:  # Time dimension
                    print(f"  ⚠ Warning: Expected time dimension 2000, got {sample_data.shape[1]}")
                if sample_data.shape[2] != 70:  # Receiver dimension
                    print(f"  ⚠ Warning: Expected receiver dimension 70, got {sample_data.shape[2]}")
            else:
                print(f"  ⚠ Warning: Unexpected number of dimensions: {sample_data.ndim}")
                
        except Exception as e:
            print(f"  ✗ Error loading sample file: {e}")
            all_valid = False
    
    print()
    if all_valid:
        print("✓ Data structure verification passed!")
        return True
    else:
        print("✗ Data structure verification failed!")
        return False

def validate_nyquist(data: np.ndarray, original_fs: int = 1000, dt_decimate: int = 4, feedback: Optional[PreprocessingFeedback] = None) -> bool:
    """
    Validate that the data satisfies Nyquist criterion after downsampling.
    
    Intuition:
    - Nyquist-Shannon Theorem: Sampling rate must be > 2x highest frequency
    - Energy Analysis: Check if significant energy exists above Nyquist frequency
    - Safety Margin: Warn if >1% of energy is above Nyquist (potential aliasing)
    
    Args:
        data: Input seismic data array
        original_fs: Original sampling frequency in Hz
        dt_decimate: The factor by which the data will be downsampled
        feedback: An optional feedback collector.
        
    Returns:
        bool: True if data satisfies Nyquist criterion
    """
    if data.ndim not in [3, 4]:
        logger.warning(f"Unexpected data dimension {data.ndim} in validate_nyquist. Skipping.")
        return True
    
    # Handle different data shapes more robustly
    if data.ndim == 4:
        # (batch, sources, time, receivers) or (batch, channels, time, receivers)
        if data.shape[1] == 5:  # sources
            time_axis = 2
        elif data.shape[1] == 1:  # channels
            time_axis = 2
        else:
            # Try to infer time axis - look for the longest dimension
            time_axis = np.argmax(data.shape[1:]) + 1
    else:  # 3D
        # (sources, time, receivers) or (time, receivers, sources)
        if data.shape[0] == 5:  # sources first
            time_axis = 1
        elif data.shape[2] == 5:  # sources last
            time_axis = 0
        else:
            # Try to infer time axis - look for the longest dimension
            time_axis = np.argmax(data.shape)

    # Ensure time_axis is valid
    if time_axis >= data.ndim:
        logger.warning(f"Invalid time_axis {time_axis} for data shape {data.shape}. Skipping validation.")
        return True

    try:
        # Compute FFT
        fft_data = np.fft.rfft(data, axis=time_axis)
        freqs = np.fft.rfftfreq(data.shape[time_axis], d=1/original_fs)
        
        # Check if significant energy exists above Nyquist frequency
        nyquist_mask = freqs > (original_fs / (2 * dt_decimate))
        
        # Handle case where nyquist_mask might be empty or have wrong shape
        if not np.any(nyquist_mask):
            return True
            
        # Ensure mask has correct shape for broadcasting
        mask_shape = [1] * data.ndim
        mask_shape[time_axis] = -1
        nyquist_mask = nyquist_mask.reshape(mask_shape)
        
        high_freq_energy = np.abs(fft_data * nyquist_mask).mean()
        total_energy = np.abs(fft_data).mean()
        
        # If more than 1% of energy is above Nyquist, warn
        if total_energy > 1e-9 and high_freq_energy / total_energy > 0.01:
            warnings.warn(f"Significant energy above Nyquist frequency detected: {high_freq_energy/total_energy:.2%}")
            if feedback:
                feedback.add_nyquist_warning()
            return False
        return True
        
    except Exception as e:
        logger.warning(f"Error in validate_nyquist: {e}. Skipping validation.")
        return True

def preprocess_one(arr: np.ndarray, dt_decimate: int = 4, is_seismic: bool = True, feedback: Optional[PreprocessingFeedback] = None) -> np.ndarray:
    """
    Preprocess a single seismic array with downsampling and normalization.
    
    Intuition:
    - Downsampling: Reduce data size while preserving signal integrity
    - Anti-aliasing: Prevent frequency folding during downsampling
    - Memory Efficiency: Use float16 for reduced memory footprint
    - Robust Normalization: Handle outliers using percentiles
    
    Processing Steps:
    1. Validate Nyquist criterion (if seismic)
    2. Apply anti-aliasing filter and downsample (if seismic and dt_decimate > 1)
    3. Normalize using robust statistics (in float32/64)
    4. Convert to float16 for storage
    
    Args:
        arr: Input seismic array
        dt_decimate: The factor by which to downsample the data
        is_seismic: Flag to indicate if the data is seismic or a velocity model
        feedback: An optional feedback collector.
        
    Returns:
        np.ndarray: Preprocessed array
    """
    try:
        if feedback:
            feedback.increment_arrays_processed()

        if is_seismic and dt_decimate > 1:
            if arr.ndim not in [3, 4]:
                logger.warning(f"Unexpected data dimension {arr.ndim} for seismic data. Skipping decimation.")
            else:
                # Determine time axis more robustly
                if arr.ndim == 4:
                    # (batch, sources/channels, time, receivers)
                    if arr.shape[1] == 5:  # sources
                        time_axis = 2
                    elif arr.shape[1] == 1:  # channels
                        time_axis = 2
                    else:
                        # Try to infer time axis - look for the longest dimension
                        time_axis = np.argmax(arr.shape[1:]) + 1
                else:  # 3D
                    # (sources, time, receivers) or (time, receivers, sources)
                    if arr.shape[0] == 5:  # sources first
                        time_axis = 1
                    elif arr.shape[2] == 5:  # sources last
                        time_axis = 0
                    else:
                        # Try to infer time axis - look for the longest dimension
                        time_axis = np.argmax(arr.shape)
                
                # Ensure time_axis is valid
                if time_axis >= arr.ndim:
                    logger.warning(f"Invalid time_axis {time_axis} for data shape {arr.shape}. Skipping decimation.")
                else:
                    # Validate Nyquist criterion
                    if not validate_nyquist(arr, dt_decimate=dt_decimate, feedback=feedback):
                        logger.warning("Data may violate Nyquist criterion after downsampling")
                    
                    # Decimate time axis with anti-aliasing filter
                    try:
                        # Check if the time dimension is large enough for decimation
                        time_dim_size = arr.shape[time_axis]
                        if time_dim_size < dt_decimate * 2:
                            logger.warning(f"Time dimension {time_dim_size} too small for decimation factor {dt_decimate}. Skipping decimation.")
                        else:
                            arr = decimate(arr, dt_decimate, axis=time_axis, ftype='fir')
                    except Exception as e:
                        logger.warning(f"Decimation failed: {e}. Skipping decimation.")
        elif is_seismic and dt_decimate == 1:
            logger.info("No downsampling applied (dt_decimate=1)")
        
        # Robust normalization per trace (in original precision)
        try:
            μ = np.median(arr, keepdims=True)
            σ = np.percentile(arr, 95, keepdims=True) - np.percentile(arr, 5, keepdims=True)
            
            # Avoid division by zero and handle overflow
            if np.isscalar(σ):
                if σ > 1e-6:
                    arr = (arr - μ) / σ
                else:
                    arr = arr - μ
            else:
                # Handle array case
                safe_σ = np.where(σ > 1e-6, σ, 1e-6)
                arr = (arr - μ) / safe_σ
        except Exception as e:
            logger.warning(f"Normalization failed: {e}. Using simple normalization.")
            # Fallback to simple normalization
            try:
                arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            except Exception as e2:
                logger.warning(f"Simple normalization also failed: {e2}. Skipping normalization.")

        # Convert to float16 for storage efficiency AFTER all calculations
        arr = arr.astype('float16')
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        # Return original array on error to avoid crashing the whole pipeline
        return arr
        
    return arr

def process_family(family: str, input_path: Union[str, Path], output_dir: Path, data_manager: Optional[DataManager] = None) -> Tuple[List[str], PreprocessingFeedback]:
    """
    Process all files for a given geological family.
    
    Args:
        family: The name of the family to process.
        input_path: The local directory path or the S3 prefix for the family's data.
        output_dir: The directory to save processed files.
        data_manager: Optional DataManager for S3 operations.
        
    Returns:
        A tuple containing a list of processed file paths and a feedback object.
    """
    logger.info(f"Processing family: {family}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feedback = PreprocessingFeedback()

    # Get family-specific settings
    family_config = FAMILY_FILE_MAP.get(family, {})
    seis_glob = family_config.get('seis_glob', '*.npy')
    vel_glob = family_config.get('vel_glob', '*.npy')
    downsample_factor = family_config.get('downsample_factor', 4) # Default to 4 if not specified
    
    logger.info(f"Processing family '{family}' with downsample_factor={downsample_factor}")
    processed_paths = []

    # === S3 Processing Path ===
    if data_manager and data_manager.use_s3:
        if not isinstance(input_path, str):
            raise ValueError(f"For S3 processing, input_path must be a string prefix, but got {type(input_path)}")
            
        # 1. List files directly from S3
        family_s3_prefix = input_path
        
        family_config = FAMILY_FILE_MAP.get(family, {})
        seis_dir = family_config.get('seis_dir', '')
        vel_dir = family_config.get('vel_dir', '')

        full_seis_prefix = f"{family_s3_prefix}/{seis_dir}/" if seis_dir else f"{family_s3_prefix}/"
        full_vel_prefix = f"{family_s3_prefix}/{vel_dir}/" if vel_dir else f"{family_s3_prefix}/"
        
        seis_keys = data_manager.list_s3_files(full_seis_prefix)
        vel_keys = data_manager.list_s3_files(full_vel_prefix)
        
        # 2. Check if files exist in S3
        if not seis_keys or not vel_keys:
            logger.warning(f"No data files found for family {family} in S3 at prefixes: {full_seis_prefix}, {full_vel_prefix}")
            return [], feedback

        # 3. Loop and process from S3
        pbar = tqdm(zip(sorted(seis_keys), sorted(vel_keys)), total=len(seis_keys), desc=f"Processing {family} from S3")
        for seis_key, vel_key in pbar:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_seis_path = Path(tmpdir) / Path(seis_key).name
                local_vel_path = Path(tmpdir) / Path(vel_key).name
                
                data_manager.s3_download(seis_key, str(local_seis_path))
                data_manager.s3_download(vel_key, str(local_vel_path))

                seis_arr = np.load(local_seis_path, mmap_mode='r')
                vel_arr = np.load(local_vel_path, mmap_mode='r')
                
                # Apply preprocessing
                seis_arr = preprocess_one(seis_arr, dt_decimate=downsample_factor, is_seismic=True, feedback=feedback)
                vel_arr = preprocess_one(vel_arr, is_seismic=False, feedback=feedback)
                
                out_seis_path = output_dir / f"seis_{Path(seis_key).stem}.npy"
                out_vel_path = output_dir / f"vel_{Path(vel_key).stem}.npy"
                
                np.save(out_seis_path, seis_arr)
                np.save(out_vel_path, vel_arr)
                processed_paths.append(str(out_seis_path))
    # === Local Processing Path ===
    else:
        if not isinstance(input_path, Path):
            raise ValueError(f"For local processing, input_path must be a Path object, but got {type(input_path)}")

        # 1. List files from the local directory
        input_dir = input_path
        seis_files = sorted(input_dir.glob(seis_glob))
        vel_files = sorted(input_dir.glob(vel_glob))

        # 2. Check if files exist locally
        if not seis_files or not vel_files:
            logger.warning(f"No data files found for family {family} in {input_dir}")
            return [], feedback

        # 3. Loop and process local files
        pbar = tqdm(zip(seis_files, vel_files), total=len(seis_files), desc=f"Processing {family} locally")
        for sfile, vfile in pbar:
            seis_arr = np.load(sfile, mmap_mode='r')
            vel_arr = np.load(vfile, mmap_mode='r')
            
            # Apply preprocessing
            seis_arr = preprocess_one(seis_arr, dt_decimate=downsample_factor, is_seismic=True, feedback=feedback)
            vel_arr = preprocess_one(vel_arr, is_seismic=False, feedback=feedback)

            out_seis_path = output_dir / f"seis_{sfile.stem}.npy"
            out_vel_path = output_dir / f"vel_{vfile.stem}.npy"
            
            np.save(out_seis_path, seis_arr)
            np.save(out_vel_path, vel_arr)
            processed_paths.append(str(out_seis_path))
            
    return processed_paths, feedback

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
            logger.info("No processed paths provided. Skipping Zarr creation.")
            return
            
        first_arr = np.load(processed_paths[0], mmap_mode='r')
            
        arr_shape = first_arr.shape
        arr_dtype = first_arr.dtype
        
        # Log the actual shape for debugging
        logger.info(f"Creating zarr dataset with shape: {arr_shape}, dtype: {arr_dtype}")
        
        # More flexible shape validation - just log warnings instead of raising errors
        if len(arr_shape) == 4:  # Seismic data
            logger.info(f"Processing 4D seismic data with shape: {arr_shape}")
        elif len(arr_shape) == 3:  # Single sample seismic data
            logger.info(f"Processing 3D seismic data with shape: {arr_shape}")
        else:
            logger.warning(f"Unexpected array shape: {arr_shape}")
            
        # Create lazy Dask arrays from local files
        lazy_arrays = [
            da.from_delayed(dask.delayed(np.load)(p, allow_pickle=True), shape=arr_shape, dtype=arr_dtype)
            for p in processed_paths
        ]
            
        # Stack arrays
        stack = da.stack(lazy_arrays, axis=0)
        
        # Get the actual shape after stacking
        stack_shape = stack.shape
        logger.info(f"Stack shape after stacking: {stack_shape}")
        
        # Adjust chunk size based on actual data shape and rechunk the array
        if len(stack_shape) == 5:
            # For 5D data (batch, samples, sources, time, receivers)
            # Use appropriate chunks for each dimension
            adjusted_chunk_size = (
                1,  # batch dimension - keep small for memory efficiency
                min(4, stack_shape[1]),  # samples dimension
                min(4, stack_shape[2]),  # sources dimension  
                min(64, stack_shape[3]),  # time dimension
                min(8, stack_shape[4])   # receivers dimension
            )
        elif len(stack_shape) == 4:
            # For 4D data, use smaller chunks
            adjusted_chunk_size = (1, min(4, stack_shape[1]), min(64, stack_shape[2]), min(8, stack_shape[3]))
        elif len(stack_shape) == 3:
            # For 3D data, use appropriate chunks
            adjusted_chunk_size = (1, min(64, stack_shape[0]), min(8, stack_shape[1]))
        else:
            # For other dimensions, try to use the provided chunk_size
            # but ensure it matches the number of dimensions
            if len(chunk_size) == len(stack_shape):
                adjusted_chunk_size = chunk_size
            else:
                # Create a default chunk size that matches the dimensions
                adjusted_chunk_size = tuple(1 for _ in range(len(stack_shape)))
                logger.warning(f"Using default chunk size {adjusted_chunk_size} for unexpected shape {stack_shape}")
            
        # Rechunk the array to the desired chunk size
        stack = stack.rechunk(adjusted_chunk_size)
        logger.info(f"Using chunk size: {adjusted_chunk_size}")
        logger.info(f"Stack shape: {stack.shape}, chunks: {stack.chunks}")

        # --- Save to Zarr ---
        # If using S3, save directly to S3. Otherwise, save locally.
        if data_manager and data_manager.use_s3:
            import s3fs
            s3_path = f"s3://{data_manager.s3_bucket}/{output_path.parent.name}/{output_path.name}"
            logger.info(f"Saving zarr dataset directly to S3: {s3_path}")
            fs = s3fs.S3FileSystem()
            store = s3fs.S3Map(root=s3_path, s3=fs, check=False)
            
            # Use zarr 3.0.8 compatible compression - try different approaches
            try:
                # Try using zarr 3.0.8 compatible compression
                import numcodecs
                codec = numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)
                logger.info("Attempting to save with zarr 3.0.8 compatible compression...")
                stack.to_zarr(store, codec=codec)
                logger.info("Successfully saved to S3 with zarr 3.0.8 compatible compression.")
            except Exception as comp_error:
                logger.warning(f"Zarr 3.0.8 compression failed: {comp_error}")
                # Fallback to no compression
                try:
                    logger.info("Attempting to save without compression...")
                    stack.to_zarr(store, codec=None)
                    logger.info("Successfully saved to S3 without compression.")
                except Exception as no_comp_error:
                    logger.warning(f"No compression also failed: {no_comp_error}")
                    # Final fallback: compute and save as numpy arrays
                    logger.info("Attempting to save as computed arrays...")
                    computed_stack = stack.compute()
                    zarr.save(store, computed_stack)
                    logger.info("Successfully saved to S3 as computed arrays.")
        else:
            logger.info(f"Saving zarr dataset locally: {output_path}")
            try:
                # Try using zarr 3.0.8 compatible compression
                import numcodecs
                codec = numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE)
                logger.info("Attempting to save with zarr 3.0.8 compatible compression...")
                stack.to_zarr(
                    output_path,
                    component='data', # Using 'data' as component for local
                    codec=codec
                )
                logger.info("Successfully saved locally with zarr 3.0.8 compatible compression.")
            except Exception as comp_error:
                logger.warning(f"Zarr 3.0.8 compression failed: {comp_error}")
                # Fallback to no compression
                try:
                    logger.info("Attempting to save without compression...")
                    stack.to_zarr(
                        output_path,
                        component='data',
                        codec=None
                    )
                    logger.info("Successfully saved locally without compression.")
                except Exception as no_comp_error:
                    logger.warning(f"No compression also failed: {no_comp_error}")
                    # Final fallback: compute and save as numpy arrays
                    logger.info("Attempting to save as computed arrays...")
                    computed_stack = stack.compute()
                    zarr.save(output_path, computed_stack)
                    logger.info("Successfully saved locally as computed arrays.")
                
    except Exception as e:
        logger.error(f"Error creating/uploading zarr dataset: {str(e)}")
        raise

def split_for_gpus(processed_paths: List[str], output_base: Path, data_manager: Optional[DataManager] = None) -> None:
    """
    Split processed files into two datasets for the two T4 GPUs and optionally upload to S3.
    Simple family-based splitting: put half the families in each GPU dataset.
    """
    try:
        # Get all families from FAMILY_FILE_MAP
        all_families = list(FAMILY_FILE_MAP.keys())
        mid = len(all_families) // 2
        
        # Split families: first half to GPU0, second half to GPU1
        gpu0_families = all_families[:mid]
        gpu1_families = all_families[mid:]
        
        logger.info(f"GPU0 families: {gpu0_families}")
        logger.info(f"GPU1 families: {gpu1_families}")
        
        # Group processed_paths by family
        family_groups = {}
        for path in processed_paths:
            # Extract family name from path
            family = Path(path).parent.name
            family_groups.setdefault(family, []).append(path)
        
        # Assign paths to GPUs based on family
        gpu0_paths = []
        gpu1_paths = []
        
        for family in gpu0_families:
            if family in family_groups:
                gpu0_paths.extend(family_groups[family])
                
        for family in gpu1_families:
            if family in family_groups:
                gpu1_paths.extend(family_groups[family])
            
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
        all_feedback = {}
        
        for family in families:
            logger.info(f"\nProcessing family: {family}")
            input_dir = input_root / family
            temp_dir = output_root / 'temp' / family
            processed_paths, feedback = process_family(family, input_dir, temp_dir, data_manager)
            all_processed_paths.extend(processed_paths)
            all_feedback[family] = feedback
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
    Main function to run the complete preprocessing pipeline.
    This function discovers data families, processes them, and stores them in a
    GPU-optimized format (Zarr).
    
    Args:
        input_root (str): Path to the root of the raw data.
        output_root (str): Path where the processed data will be saved.
        use_s3 (bool): Whether to use S3 for data I/O.
        
    Returns:
        A dictionary containing feedback from the preprocessing run.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    data_manager = DataManager(use_s3=use_s3)
    families = FAMILY_FILE_MAP.keys()
    
    all_processed_paths = []
    all_feedback = {}

    for family in families:
        logger.info(f"--- Starting family: {family} ---")
        family_output_dir = output_root / family

        if use_s3:
            # For S3, the input_path is a prefix string
            family_input_path = f"{input_root}/{family}"
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)
        else:
            # For local, the input_path is a Path object
            family_input_path = Path(input_root) / family
            if not family_input_path.exists():
                logger.warning(f"Skipping family {family}: directory not found at {family_input_path}")
                continue
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)

        all_processed_paths.extend(processed_paths)
        all_feedback[family] = feedback

    # Create GPU-specific datasets
    logger.info("--- Creating GPU-specific datasets ---")
    split_for_gpus(all_processed_paths, output_root, data_manager)
    
    # Clean up temporary family directories
    for family in families:
        family_dir = output_root / family
        if family_dir.exists():
            import shutil
            shutil.rmtree(family_dir)
            logger.info(f"Cleaned up temporary family directory: {family_dir}")
    
    logger.info("--- Preprocessing pipeline complete ---")
    return all_feedback

if __name__ == "__main__":
    main() 