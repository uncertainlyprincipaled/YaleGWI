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
import pickle
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable debug logging for detailed S3 processing output
if os.environ.get('DEBUG_MODE', '0') == '1':
    logger.setLevel(logging.DEBUG)
    # Also set up debug handler for more detailed output
    debug_handler = logging.StreamHandler()
    debug_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

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

class PreprocessingCache:
    """Cache for preprocessing results to avoid reprocessing test data."""
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path('/tmp/preprocessing_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, data_hash: str, dt_decimate: int, is_seismic: bool) -> str:
        """Generate cache key for preprocessing parameters."""
        key_data = f"{data_hash}_{dt_decimate}_{is_seismic}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, data_hash: str, dt_decimate: int, is_seismic: bool) -> Optional[np.ndarray]:
        """Get cached preprocessing result if available."""
        cache_key = self._get_cache_key(data_hash, dt_decimate, is_seismic)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def cache_result(self, data_hash: str, dt_decimate: int, is_seismic: bool, result: np.ndarray):
        """Cache preprocessing result."""
        cache_key = self._get_cache_key(data_hash, dt_decimate, is_seismic)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

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
        
        # Calculate energy ratio
        energy_ratio = high_freq_energy / total_energy if total_energy > 1e-9 else 0.0
        
        # More detailed logging for high energy cases
        if energy_ratio > 0.01:  # More than 1% energy above Nyquist
            logger.warning(f"High energy above Nyquist frequency detected: {energy_ratio:.2%}")
            logger.warning(f"  - Data shape: {data.shape}")
            logger.warning(f"  - Time axis: {time_axis}")
            logger.warning(f"  - Original sampling rate: {original_fs} Hz")
            logger.warning(f"  - Decimation factor: {dt_decimate}")
            logger.warning(f"  - New Nyquist frequency: {original_fs / (2 * dt_decimate)} Hz")
            
            # Check if this might be due to incorrect sampling rate assumption
            if energy_ratio > 0.5:  # More than 50% energy above Nyquist
                logger.error(f"CRITICAL: {energy_ratio:.2%} energy above Nyquist frequency!")
                logger.error("This suggests either:")
                logger.error("  1. Incorrect sampling rate assumption (not 1000 Hz)")
                logger.error("  2. Data is already downsampled")
                logger.error("  3. Data is corrupted or has high-frequency noise")
                logger.error("  4. Decimation factor is too aggressive")
                
                # Suggest alternative sampling rates
                suggested_rates = [500, 250, 125, 62.5]
                for rate in suggested_rates:
                    alt_nyquist = rate / (2 * dt_decimate)
                    alt_mask = freqs > alt_nyquist
                    if np.any(alt_mask):
                        alt_mask = alt_mask.reshape(mask_shape)
                        alt_high_energy = np.abs(fft_data * alt_mask).mean()
                        alt_ratio = alt_high_energy / total_energy if total_energy > 1e-9 else 0.0
                        logger.info(f"  - If sampling rate was {rate} Hz: {alt_ratio:.2%} energy above Nyquist")
            
            if feedback:
                feedback.add_nyquist_warning()
            return False
        return True
        
    except Exception as e:
        logger.warning(f"Error in validate_nyquist: {e}. Skipping validation.")
        return True

def detect_sampling_rate(data: np.ndarray, original_fs: int = 1000) -> int:
    """
    Detect the actual sampling rate of the data by analyzing frequency content.
    
    Args:
        data: Input seismic data array
        original_fs: Assumed original sampling frequency in Hz
        
    Returns:
        int: Detected sampling rate in Hz
    """
    if data.ndim not in [3, 4]:
        logger.warning(f"Unexpected data dimension {data.ndim} in detect_sampling_rate. Using assumed rate.")
        return original_fs
    
    # Handle different data shapes
    if data.ndim == 4:
        if data.shape[1] == 5:  # sources
            time_axis = 2
        elif data.shape[1] == 1:  # channels
            time_axis = 2
        else:
            time_axis = np.argmax(data.shape[1:]) + 1
    else:  # 3D
        if data.shape[0] == 5:  # sources first
            time_axis = 1
        elif data.shape[2] == 5:  # sources last
            time_axis = 0
        else:
            time_axis = np.argmax(data.shape)

    if time_axis >= data.ndim:
        logger.warning(f"Invalid time_axis {time_axis} for data shape {data.shape}. Using assumed rate.")
        return original_fs

    try:
        # Compute FFT
        fft_data = np.fft.rfft(data, axis=time_axis)
        freqs = np.fft.rfftfreq(data.shape[time_axis], d=1/original_fs)
        
        # Find the frequency where 95% of energy is below
        total_energy = np.abs(fft_data).sum()
        cumulative_energy = np.cumsum(np.abs(fft_data).sum(axis=tuple(i for i in range(data.ndim) if i != time_axis)))
        
        # Find frequency where 95% of energy is contained
        energy_threshold = 0.95 * total_energy
        energy_idx = np.where(cumulative_energy >= energy_threshold)[0]
        
        if len(energy_idx) > 0:
            detected_freq = freqs[energy_idx[0]]
            
            # Round to common sampling rates
            common_rates = [1000, 500, 250, 125, 62.5]
            detected_rate = min(common_rates, key=lambda x: abs(x - detected_freq))
            
            logger.info(f"Detected sampling rate: {detected_rate} Hz (95% energy below {detected_freq:.1f} Hz)")
            return detected_rate
        else:
            logger.warning("Could not detect sampling rate. Using assumed rate.")
            return original_fs
            
    except Exception as e:
        logger.warning(f"Error in detect_sampling_rate: {e}. Using assumed rate.")
        return original_fs

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
                    # Detect actual sampling rate first
                    detected_fs = detect_sampling_rate(arr, original_fs=1000)
                    
                    # Validate Nyquist criterion with detected sampling rate
                    if not validate_nyquist(arr, original_fs=detected_fs, dt_decimate=dt_decimate, feedback=feedback):
                        logger.warning(f"Data may violate Nyquist criterion after downsampling (detected fs: {detected_fs} Hz)")
                        
                        # If data appears to be already downsampled, skip further decimation
                        if detected_fs < 1000:
                            logger.info(f"Data appears to be already downsampled (detected fs: {detected_fs} Hz). Skipping decimation.")
                            dt_decimate = 1
                    
                    # Decimate time axis with anti-aliasing filter (only if not already downsampled)
                    if dt_decimate > 1:
                        try:
                            # Check if the time dimension is large enough for decimation
                            time_dim_size = arr.shape[time_axis]
                            
                            # If data is detected as already downsampled but still has large time dimension,
                            # we need to downsample to get to the target time dimension
                            target_time_dim = 500  # Target time dimension after preprocessing
                            
                            if time_dim_size > target_time_dim:
                                # Calculate required decimation factor to reach target
                                required_decimate = time_dim_size // target_time_dim
                                logger.info(f"Time dimension {time_dim_size} > {target_time_dim}. Applying decimation factor {required_decimate}")
                                
                                if time_dim_size >= required_decimate * 2:
                                    arr = decimate(arr, required_decimate, axis=time_axis, ftype='fir')
                                    logger.info(f"Decimated from {time_dim_size} to {arr.shape[time_axis]} time steps")
                                else:
                                    logger.warning(f"Time dimension {time_dim_size} too small for decimation factor {required_decimate}. Skipping decimation.")
                            else:
                                logger.info(f"Time dimension {time_dim_size} already at or below target {target_time_dim}. No decimation needed.")
                                
                        except Exception as e:
                            logger.warning(f"Decimation failed: {e}. Skipping decimation.")
                    else:
                        logger.info(f"No decimation applied (dt_decimate={dt_decimate})")
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

def preprocess_one_cached(arr: np.ndarray, dt_decimate: int = 4, is_seismic: bool = True, 
                         feedback: Optional[PreprocessingFeedback] = None,
                         use_cache: bool = True) -> np.ndarray:
    """
    Cached version of preprocess_one for inference efficiency.
    
    Args:
        arr: Input seismic array
        dt_decimate: The factor by which to downsample the data
        is_seismic: Flag to indicate if the data is seismic or a velocity model
        feedback: An optional feedback collector
        use_cache: Whether to use caching (recommended for inference)
        
    Returns:
        np.ndarray: Preprocessed array
    """
    if not use_cache:
        return preprocess_one(arr, dt_decimate, is_seismic, feedback)
    
    # Generate hash of input data for caching
    data_hash = hashlib.md5(arr.tobytes()).hexdigest()
    
    # Check cache first
    cache = PreprocessingCache()
    cached_result = cache.get_cached_result(data_hash, dt_decimate, is_seismic)
    
    if cached_result is not None:
        logger.debug(f"Using cached preprocessing result for {data_hash[:8]}...")
        return cached_result
    
    # Process and cache result
    result = preprocess_one(arr, dt_decimate, is_seismic, feedback)
    cache.cache_result(data_hash, dt_decimate, is_seismic, result)
    
    return result

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
        
        logger.info(f"🐛 S3 processing for family {family}")
        logger.info(f"🐛 Family S3 prefix: {family_s3_prefix}")
        logger.info(f"🐛 Seismic prefix: {full_seis_prefix}")
        logger.info(f"🐛 Velocity prefix: {full_vel_prefix}")
        
        seis_keys = data_manager.list_s3_files(full_seis_prefix)
        vel_keys = data_manager.list_s3_files(full_vel_prefix)
        
        logger.info(f"🐛 Found {len(seis_keys)} seismic files and {len(vel_keys)} velocity files in S3")
        
        # If no files found with expected structure, try alternative patterns
        if not seis_keys or not vel_keys:
            logger.warning(f"🐛 No files found with expected structure. Trying alternative patterns...")
            
            # Try different possible patterns
            alternative_patterns = [
                (f"{family_s3_prefix}/", f"{family_s3_prefix}/"),  # Same prefix for both
                (f"{family_s3_prefix}/seis/", f"{family_s3_prefix}/vel/"),  # seis/vel subdirs
                (f"{family_s3_prefix}/data/", f"{family_s3_prefix}/model/"),  # data/model subdirs
                (f"{family_s3_prefix}/", f"{family_s3_prefix}/"),  # Root level
            ]
            
            for seis_pattern, vel_pattern in alternative_patterns:
                logger.info(f"🐛 Trying pattern: seis={seis_pattern}, vel={vel_pattern}")
                test_seis_keys = data_manager.list_s3_files(seis_pattern)
                test_vel_keys = data_manager.list_s3_files(vel_pattern)
                
                logger.info(f"🐛   Found {len(test_seis_keys)} seismic, {len(test_vel_keys)} velocity")
                
                if test_seis_keys and test_vel_keys:
                    logger.info(f"🐛   Success! Using alternative pattern")
                    seis_keys = test_seis_keys
                    vel_keys = test_vel_keys
                    full_seis_prefix = seis_pattern
                    full_vel_prefix = vel_pattern
                    break
            else:
                logger.error(f"🐛 No alternative patterns worked. Available files in {family_s3_prefix}:")
                all_files = data_manager.list_s3_files(family_s3_prefix)
                logger.error(f"🐛   {all_files}")
        
        # 2. Check if files exist in S3
        if not seis_keys or not vel_keys:
            logger.warning(f"No data files found for family {family} in S3 at prefixes: {full_seis_prefix}, {full_vel_prefix}")
            return [], feedback

        # 3. Loop and process from S3
        pbar = tqdm(zip(sorted(seis_keys), sorted(vel_keys)), total=len(seis_keys), desc=f"Processing {family} from S3")
        for seis_key, vel_key in pbar:
            # Create temporary files in the output directory instead of a temp directory
            local_seis_path = output_dir / f"temp_seis_{Path(seis_key).name}"
            local_vel_path = output_dir / f"temp_vel_{Path(vel_key).name}"
            
            logger.debug(f"🐛 Processing S3 files: {seis_key} -> {local_seis_path}")
            logger.debug(f"🐛 Processing S3 files: {vel_key} -> {local_vel_path}")
            
            try:
                # Download files from S3
                logger.debug(f"🐛 Downloading {seis_key}...")
                data_manager.s3_download(seis_key, str(local_seis_path))
                logger.debug(f"🐛 Downloading {vel_key}...")
                data_manager.s3_download(vel_key, str(local_vel_path))

                # Check if files were downloaded successfully
                if not local_seis_path.exists():
                    logger.error(f"🐛 Failed to download seismic file: {seis_key}")
                    continue
                if not local_vel_path.exists():
                    logger.error(f"🐛 Failed to download velocity file: {vel_key}")
                    continue

                logger.debug(f"🐛 Loading arrays from downloaded files...")
                seis_arr = np.load(local_seis_path, mmap_mode='r')
                vel_arr = np.load(local_vel_path, mmap_mode='r')
                
                logger.debug(f"🐛 Loaded arrays - Seismic: {seis_arr.shape}, Velocity: {vel_arr.shape}")
                
                # Apply preprocessing
                logger.debug(f"🐛 Applying preprocessing to seismic data...")
                seis_arr = preprocess_one_cached(seis_arr, dt_decimate=downsample_factor, is_seismic=True, feedback=feedback)
                logger.debug(f"🐛 Applying preprocessing to velocity data...")
                vel_arr = preprocess_one_cached(vel_arr, is_seismic=False, feedback=feedback)
                
                logger.debug(f"🐛 Preprocessed arrays - Seismic: {seis_arr.shape}, Velocity: {vel_arr.shape}")
                
                out_seis_path = output_dir / f"seis_{Path(seis_key).stem}.npy"
                out_vel_path = output_dir / f"vel_{Path(vel_key).stem}.npy"
                
                logger.debug(f"🐛 Saving processed files...")
                np.save(out_seis_path, seis_arr)
                np.save(out_vel_path, vel_arr)
                
                # Verify files were saved
                if out_seis_path.exists() and out_vel_path.exists():
                    processed_paths.append(str(out_seis_path))
                    processed_paths.append(str(out_vel_path))
                    logger.debug(f"🐛 Successfully saved and added to processed paths: {out_seis_path}, {out_vel_path}")
                else:
                    logger.error(f"🐛 Failed to save processed files: {out_seis_path}, {out_vel_path}")
                
                # Clean up temporary files
                local_seis_path.unlink(missing_ok=True)
                local_vel_path.unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"🐛 Failed to process {seis_key}: {e}")
                # Clean up temporary files on error
                local_seis_path.unlink(missing_ok=True)
                local_vel_path.unlink(missing_ok=True)
                continue
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
            seis_arr = preprocess_one_cached(seis_arr, dt_decimate=downsample_factor, is_seismic=True, feedback=feedback)
            vel_arr = preprocess_one_cached(vel_arr, is_seismic=False, feedback=feedback)

            out_seis_path = output_dir / f"seis_{sfile.stem}.npy"
            out_vel_path = output_dir / f"vel_{vfile.stem}.npy"
            
            np.save(out_seis_path, seis_arr)
            np.save(out_vel_path, vel_arr)
            processed_paths.append(str(out_seis_path))
            processed_paths.append(str(out_vel_path))
            
    logger.info(f"🐛 Processed {len(processed_paths)} files for family {family}")
    logger.info(f"🐛 Processed paths: {processed_paths}")
    
    # Check if files actually exist
    existing_paths = []
    for path in processed_paths:
        if Path(path).exists():
            existing_paths.append(path)
        else:
            logger.warning(f"🐛 Processed path does not exist: {path}")
    
    logger.info(f"🐛 Existing paths: {len(existing_paths)}/{len(processed_paths)}")
    
    # Additional debugging for S3 processing
    if data_manager and data_manager.use_s3:
        logger.info(f"🐛 S3 processing summary for {family}:")
        logger.info(f"🐛 - Seismic keys found: {len(seis_keys) if 'seis_keys' in locals() else 'N/A'}")
        logger.info(f"🐛 - Velocity keys found: {len(vel_keys) if 'vel_keys' in locals() else 'N/A'}")
        logger.info(f"🐛 - Seismic prefix: {full_seis_prefix if 'full_seis_prefix' in locals() else 'N/A'}")
        logger.info(f"🐛 - Velocity prefix: {full_vel_prefix if 'full_vel_prefix' in locals() else 'N/A'}")
    else:
        logger.info(f"🐛 Local processing summary for {family}:")
        logger.info(f"🐛 - Seismic files found: {len(seis_files) if 'seis_files' in locals() else 'N/A'}")
        logger.info(f"🐛 - Velocity files found: {len(vel_files) if 'vel_files' in locals() else 'N/A'}")
        logger.info(f"🐛 - Input directory: {input_dir if 'input_dir' in locals() else 'N/A'}")
            
    return processed_paths, feedback

def create_zarr_dataset(processed_paths: List[str], output_path: Path, chunk_size: Tuple[int, ...], data_manager: Optional[DataManager] = None) -> None:
    """
    Create a zarr dataset from processed numpy files with proper shape handling.
    
    Args:
        processed_paths: List of paths to processed numpy files
        output_path: Path where to save the zarr dataset
        chunk_size: Chunk size for the zarr dataset
        data_manager: Optional DataManager for S3 operations
    """
    try:
        if not processed_paths:
            logger.info("No processed paths provided. Skipping Zarr creation.")
            return
            
        # Separate seismic and velocity files
        seismic_paths = []
        velocity_paths = []
        
        for path in processed_paths:
            filename = Path(path).name
            # Check for velocity files first (they contain 'vel' in the name)
            if 'vel' in filename:
                velocity_paths.append(path)
            # Then check for pure seismic files (contain 'seis' but not 'vel')
            elif 'seis' in filename and 'vel' not in filename:
                seismic_paths.append(path)
            else:
                logger.warning(f"Unknown file type: {path}")
        
        logger.info(f"Found {len(seismic_paths)} seismic files and {len(velocity_paths)} velocity files")
        
        # Create a single zarr group with both seismic and velocity data
        if seismic_paths and velocity_paths:
            # Process seismic data
            first_seismic = np.load(seismic_paths[0], mmap_mode='r')
            seismic_shape = first_seismic.shape
            seismic_dtype = first_seismic.dtype
            
            logger.info(f"Processing seismic data with shape: {seismic_shape}, dtype: {seismic_dtype}")
            
            # Create lazy Dask arrays for seismic data
            seismic_arrays = []
            valid_seismic = 0
            for p in seismic_paths:
                try:
                    arr = np.load(p, mmap_mode='r')
                    if arr.shape != seismic_shape:
                        logger.warning(f"Seismic shape mismatch in {p}: expected {seismic_shape}, got {arr.shape}")
                        continue
                    seismic_arrays.append(
                        da.from_delayed(dask.delayed(np.load)(p, allow_pickle=True), shape=seismic_shape, dtype=seismic_dtype)
                    )
                    valid_seismic += 1
                except Exception as e:
                    logger.warning(f"Failed to load seismic file {p}: {e}")
                    continue
            
            logger.info(f"Valid seismic arrays: {valid_seismic}/{len(seismic_paths)}")
            
            # Process velocity data
            first_velocity = np.load(velocity_paths[0], mmap_mode='r')
            velocity_shape = first_velocity.shape
            velocity_dtype = first_velocity.dtype
            
            logger.info(f"Processing velocity data with shape: {velocity_shape}, dtype: {velocity_dtype}")
            
            # Create lazy Dask arrays for velocity data
            velocity_arrays = []
            valid_velocity = 0
            for p in velocity_paths:
                try:
                    arr = np.load(p, mmap_mode='r')
                    if arr.shape != velocity_shape:
                        logger.warning(f"Velocity shape mismatch in {p}: expected {velocity_shape}, got {arr.shape}")
                        continue
                    velocity_arrays.append(
                        da.from_delayed(dask.delayed(np.load)(p, allow_pickle=True), shape=velocity_shape, dtype=velocity_dtype)
                    )
                    valid_velocity += 1
                except Exception as e:
                    logger.warning(f"Failed to load velocity file {p}: {e}")
                    continue
            
            logger.info(f"Valid velocity arrays: {valid_velocity}/{len(velocity_paths)}")
            
            if seismic_arrays and velocity_arrays:
                # Stack arrays
                seismic_stack = da.stack(seismic_arrays, axis=0)
                velocity_stack = da.stack(velocity_arrays, axis=0)
                
                logger.info(f"Seismic stack shape: {seismic_stack.shape}")
                logger.info(f"Velocity stack shape: {velocity_stack.shape}")
                
                # Save both seismic and velocity data in a single zarr file
                save_combined_zarr_data(seismic_stack, velocity_stack, output_path, data_manager)
            else:
                logger.error("No valid arrays found to save")
                return
        else:
            logger.error("Need both seismic and velocity data to create dataset")
            return
                
    except Exception as e:
        logger.error(f"Error creating/uploading zarr dataset: {str(e)}")
        raise

def save_zarr_data(stack, output_path, data_manager):
    """
    Save stacked data to zarr format with proper chunking and S3/local saving.
    
    Args:
        stack: Dask array to save
        output_path: Path to save the data
        data_manager: DataManager instance for S3 operations
    """
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
        # For other dimensions, create a default chunk size that matches the dimensions
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
        
        # Use fsspec-based approach for zarr 3.0.8 with old s3fs compatibility
        try:
            logger.info("Saving to S3 without compression...")
            # Use direct fsspec URL instead of s3fs.S3Map for better compatibility
            stack.to_zarr(s3_path)
            logger.info("Successfully saved to S3 without compression.")
        except Exception as e:
            logger.warning(f"S3 save failed: {e}")
            # Try alternative approach for old s3fs versions
            try:
                logger.info("Trying alternative S3 save method...")
                # Compute the data first, then save
                computed_stack = stack.compute()
                # Use zarr.save with fsspec URL
                zarr.save(s3_path, computed_stack)
                logger.info("Successfully saved to S3 using alternative method.")
            except Exception as e2:
                logger.error(f"All S3 save methods failed: {e2}")
                logger.info("Falling back to local save only...")
                # Save locally as fallback
                try:
                    stack.to_zarr(output_path, component='data')
                    logger.info("Saved locally as fallback.")
                except Exception as e3:
                    logger.error(f"Local fallback also failed: {e3}")
                    # Final fallback - save as numpy arrays
                    try:
                        logger.info("Final fallback: saving as numpy arrays...")
                        computed_stack = stack.compute()
                        np.save(output_path.with_suffix('.npy'), computed_stack)
                        logger.info("Saved as numpy arrays as final fallback.")
                    except Exception as e4:
                        logger.error(f"All save methods failed: {e4}")
                        raise
    else:
        logger.info(f"Saving zarr dataset locally: {output_path}")
        
        # Save without compression
        try:
            logger.info("Saving locally without compression...")
            stack.to_zarr(
                output_path,
                component='data' # Using 'data' as component for local
            )
            logger.info("Successfully saved locally without compression.")
        except Exception as e:
            logger.warning(f"Local save failed: {e}")
            # Final fallback - compute and save
            logger.info("Attempting to save as computed arrays...")
            computed_stack = stack.compute()
            zarr.save(output_path, computed_stack)
            logger.info("Successfully saved locally as computed arrays.")

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
            logger.info(f"🐛 Total processed paths so far: {len(all_processed_paths)}")
            logger.info(f"🐛 Sample processed paths: {processed_paths[:3] if processed_paths else 'None'}")

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

def load_data_debug(input_root, output_root, use_s3=False, debug_family='FlatVel_A'):
    """
    Debug version of load_data that processes only one family for quick S3 I/O testing.
    
    Args:
        input_root (str): Path to the root of the raw data.
        output_root (str): Path where the processed data will be saved.
        use_s3 (bool): Whether to use S3 for data I/O.
        debug_family (str): Which family to process (default: 'FlatVel_A').
        
    Returns:
        A dictionary containing feedback from the preprocessing run.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    data_manager = DataManager(use_s3=use_s3)
    
    # Process only the specified family
    families = [debug_family]
    
    logger.info(f"🐛 DEBUG MODE: Processing only family '{debug_family}'")
    logger.info(f"🐛 This will help identify S3 I/O issues quickly")
    
    all_processed_paths = []
    all_feedback = {}

    for family in families:
        logger.info(f"🐛 --- Starting debug family: {family} ---")
        family_output_dir = output_root / family

        if use_s3:
            # For S3, the input_path is a prefix string
            family_input_path = f"{input_root}/{family}"
            logger.info(f"🐛 S3 input path: {family_input_path}")
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)
        else:
            # For local, the input_path is a Path object
            family_input_path = Path(input_root) / family
            logger.info(f"🐛 Local input path: {family_input_path}")
            if not family_input_path.exists():
                logger.warning(f"🐛 Skipping family {family}: directory not found at {family_input_path}")
                continue
            processed_paths, feedback = process_family(family, family_input_path, family_output_dir, data_manager)

        all_processed_paths.extend(processed_paths)
        all_feedback[family] = feedback
        
        logger.info(f"🐛 Family {family}: {len(processed_paths)} files processed")
        logger.info(f"🐛 Total processed paths so far: {len(all_processed_paths)}")
        logger.info(f"🐛 Sample processed paths: {processed_paths[:3] if processed_paths else 'None'}")

        # Additional debugging for debug mode
        if len(processed_paths) == 0:
            logger.warning(f"🐛 WARNING: No files processed for family {family}")
            logger.warning(f"🐛 - Input path: {family_input_path}")
            logger.warning(f"🐛 - Output dir: {family_output_dir}")
            logger.warning(f"🐛 - Use S3: {use_s3}")
            if use_s3:
                logger.warning(f"🐛 - S3 prefix: {family_input_path}")
            else:
                logger.warning(f"🐛 - Local path exists: {Path(family_input_path).exists()}")
                if Path(family_input_path).exists():
                    logger.warning(f"🐛 - Local path contents: {list(Path(family_input_path).glob('*.npy'))}")

    # Create GPU-specific datasets (simplified for debug mode)
    logger.info("🐛 --- Creating GPU-specific datasets (debug mode) ---")
    
    if all_processed_paths:
        # In debug mode, put all processed files in GPU0 for simplicity
        gpu0_dir = output_root / 'gpu0'
        gpu1_dir = output_root / 'gpu1'
        gpu0_dir.mkdir(parents=True, exist_ok=True)
        gpu1_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a minimal zarr dataset for GPU0
        create_zarr_dataset(
            all_processed_paths,
            gpu0_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            data_manager
        )
        
        # Create an empty zarr dataset for GPU1 (to maintain structure)
        logger.info("🐛 Creating empty GPU1 dataset to maintain structure")
        try:
            import zarr
            # Create empty combined zarr with both seismic and velocity arrays
            root = zarr.group(str(gpu1_dir / 'seismic.zarr'))
            root.create_dataset(
                'seismic', 
                data=np.zeros((0, 5, 500, 70), dtype='float16'),
                shape=(0, 5, 500, 70),  # Explicit shape parameter
                dtype='float16'
            )
            root.create_dataset(
                'velocity', 
                data=np.zeros((0, 1, 70, 70), dtype='float16'),
                shape=(0, 1, 70, 70),  # Explicit shape parameter
                dtype='float16'
            )
        except Exception as e:
            logger.warning(f"🐛 Could not create empty GPU1 dataset: {e}")
        
        logger.info(f"🐛 Created debug GPU datasets with {len(all_processed_paths)} samples in GPU0")
    else:
        logger.warning("🐛 No files processed - cannot create GPU datasets")
    
    # Clean up temporary family directories
    for family in families:
        family_dir = output_root / family
        if family_dir.exists():
            import shutil
            shutil.rmtree(family_dir)
            logger.info(f"🐛 Cleaned up temporary family directory: {family_dir}")
    
    logger.info("🐛 --- Debug preprocessing pipeline complete ---")
    return all_feedback

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

def save_combined_zarr_data(seismic_stack, velocity_stack, output_path, data_manager):
    """
    Save both seismic and velocity data in a single zarr file with proper structure.
    
    Args:
        seismic_stack: Dask array of seismic data
        velocity_stack: Dask array of velocity data  
        output_path: Path to save the zarr file
        data_manager: DataManager instance for S3 operations
    """
    # Get the actual shapes after stacking
    seismic_shape = seismic_stack.shape
    velocity_shape = velocity_stack.shape
    logger.info(f"Seismic stack shape: {seismic_shape}")
    logger.info(f"Velocity stack shape: {velocity_shape}")
    
    # Adjust chunk size based on actual data shape and rechunk the arrays
    if len(seismic_shape) == 5:
        # For 5D data (batch, samples, sources, time, receivers)
        seismic_chunk_size = (
            1,  # batch dimension - keep small for memory efficiency
            min(4, seismic_shape[1]),  # samples dimension
            min(4, seismic_shape[2]),  # sources dimension  
            min(64, seismic_shape[3]),  # time dimension
            min(8, seismic_shape[4])   # receivers dimension
        )
    elif len(seismic_shape) == 4:
        # For 4D data, use smaller chunks
        seismic_chunk_size = (1, min(4, seismic_shape[1]), min(64, seismic_shape[2]), min(8, seismic_shape[3]))
    else:
        # For other dimensions, create a default chunk size
        seismic_chunk_size = tuple(1 for _ in range(len(seismic_shape)))
        logger.warning(f"Using default chunk size {seismic_chunk_size} for unexpected seismic shape {seismic_shape}")
    
    # Similar chunking for velocity data
    if len(velocity_shape) == 5:
        # For 5D velocity data (batch, samples, channels, height, width)
        velocity_chunk_size = (
            1,  # batch dimension - keep small for memory efficiency
            min(4, velocity_shape[1]),  # samples dimension
            min(1, velocity_shape[2]),  # channels dimension (usually 1 for velocity)
            min(8, velocity_shape[3]),  # height dimension
            min(8, velocity_shape[4])   # width dimension
        )
    elif len(velocity_shape) == 4:
        # For 4D velocity data (batch, channels, height, width)
        velocity_chunk_size = (1, min(4, velocity_shape[1]), min(8, velocity_shape[2]), min(8, velocity_shape[3]))
    elif len(velocity_shape) == 3:
        # For 3D velocity data (channels, height, width)
        velocity_chunk_size = (1, min(8, velocity_shape[1]), min(8, velocity_shape[2]))
    else:
        velocity_chunk_size = tuple(1 for _ in range(len(velocity_shape)))
        logger.warning(f"Using default chunk size {velocity_chunk_size} for unexpected velocity shape {velocity_shape}")
    
    # Rechunk the arrays
    seismic_stack = seismic_stack.rechunk(seismic_chunk_size)
    velocity_stack = velocity_stack.rechunk(velocity_chunk_size)
    
    logger.info(f"Using seismic chunk size: {seismic_chunk_size}")
    logger.info(f"Using velocity chunk size: {velocity_chunk_size}")

    # --- Save to Zarr ---
    # If using S3, save directly to S3. Otherwise, save locally.
    if data_manager and data_manager.use_s3:
        import s3fs
        s3_path = f"s3://{data_manager.s3_bucket}/{output_path.parent.name}/{output_path.name}"
        logger.info(f"Saving combined zarr dataset to S3: {s3_path}")
        
        try:
            # Use dask's to_zarr method which handles S3 better
            logger.info("Attempting S3 save with dask.to_zarr...")
            
            # Compute the data first to avoid S3 compatibility issues
            seismic_data = seismic_stack.compute()
            velocity_data = velocity_stack.compute()
            
            # Create a temporary local zarr file first
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_zarr_path = Path(tmpdir) / "temp.zarr"
                
                # Save locally first
                import zarr
                root = zarr.group(str(temp_zarr_path))
                
                # Save seismic data with explicit shape
                # Convert dask chunks to flat tuple for zarr
                seismic_chunks = tuple(chunk[0] if isinstance(chunk, tuple) else chunk for chunk in seismic_stack.chunks)
                seismic_array = root.create_dataset(
                    'seismic',
                    data=seismic_data,
                    chunks=seismic_chunks,
                    dtype='float16',
                    shape=seismic_data.shape  # Explicitly provide shape
                )
                
                # Save velocity data with explicit shape
                # Convert dask chunks to flat tuple for zarr
                velocity_chunks = tuple(chunk[0] if isinstance(chunk, tuple) else chunk for chunk in velocity_stack.chunks)
                velocity_array = root.create_dataset(
                    'velocity',
                    data=velocity_data, 
                    chunks=velocity_chunks,
                    dtype='float16',
                    shape=velocity_data.shape  # Explicitly provide shape
                )
                
                # Now upload the entire zarr directory to S3
                import shutil
                import subprocess
                
                # Use aws CLI to sync the zarr directory to S3
                s3_uri = f"s3://{data_manager.s3_bucket}/{output_path.parent.name}/{output_path.name}"
                try:
                    subprocess.run([
                        'aws', 's3', 'sync', str(temp_zarr_path), s3_uri, '--quiet'
                    ], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback: use boto3 to upload files
                    logger.info("AWS CLI not available, using boto3 fallback...")
                    import boto3
                    s3_client = boto3.client('s3')
                    
                    # Upload each file in the zarr directory
                    for file_path in temp_zarr_path.rglob('*'):
                        if file_path.is_file():
                            relative_path = file_path.relative_to(temp_zarr_path)
                            s3_key = f"{output_path.parent.name}/{output_path.name}/{relative_path}"
                            s3_client.upload_file(str(file_path), data_manager.s3_bucket, str(s3_key))
                
                logger.info("Successfully saved combined dataset to S3")
            
        except Exception as e:
            logger.warning(f"S3 save failed: {e}")
            # Fallback to local save
            logger.info("Falling back to local save...")
            save_combined_zarr_local(seismic_stack, velocity_stack, output_path)
            
    else:
        logger.info(f"Saving combined zarr dataset locally: {output_path}")
        save_combined_zarr_local(seismic_stack, velocity_stack, output_path)

def save_combined_zarr_local(seismic_stack, velocity_stack, output_path):
    """
    Save combined zarr dataset locally.
    """
    try:
        import zarr
        
        # Create zarr group
        root = zarr.group(str(output_path))
        
        # Compute the data first to get actual shapes
        seismic_data = seismic_stack.compute()
        velocity_data = velocity_stack.compute()
        
        # Save seismic data with explicit shape
        # Convert dask chunks to flat tuple for zarr
        seismic_chunks = tuple(chunk[0] if isinstance(chunk, tuple) else chunk for chunk in seismic_stack.chunks)
        seismic_array = root.create_dataset(
            'seismic',
            data=seismic_data,
            chunks=seismic_chunks,
            dtype='float16',
            shape=seismic_data.shape  # Explicitly provide shape
        )
        
        # Save velocity data with explicit shape
        # Convert dask chunks to flat tuple for zarr
        velocity_chunks = tuple(chunk[0] if isinstance(chunk, tuple) else chunk for chunk in velocity_stack.chunks)
        velocity_array = root.create_dataset(
            'velocity',
            data=velocity_data,
            chunks=velocity_chunks,
            dtype='float16',
            shape=velocity_data.shape  # Explicitly provide shape
        )
        
        logger.info("Successfully saved combined dataset locally")
        
    except Exception as e:
        logger.error(f"Local save failed: {e}")
        # Final fallback - save as numpy arrays
        try:
            logger.info("Final fallback: saving as numpy arrays...")
            seismic_data = seismic_stack.compute()
            velocity_data = velocity_stack.compute()
            np.save(output_path.with_suffix('.seismic.npy'), seismic_data)
            np.save(output_path.with_suffix('.velocity.npy'), velocity_data)
            logger.info("Saved as numpy arrays as final fallback")
        except Exception as e2:
            logger.error(f"All save methods failed: {e2}")
            raise

def preprocess_test_data_batch(test_files: List[Path], 
                              output_dir: Path,
                              dt_decimate: int = 4,
                              batch_size: int = 100,
                              num_workers: int = 4,
                              use_cache: bool = True) -> List[Path]:
    """
    Efficiently preprocess test data in batches for inference.
    
    Args:
        test_files: List of test file paths
        output_dir: Directory to save preprocessed files
        dt_decimate: Downsampling factor
        batch_size: Number of files to process in parallel
        num_workers: Number of parallel workers
        use_cache: Whether to use preprocessing cache
        
    Returns:
        List of preprocessed file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_files = []
    
    logger.info(f"Preprocessing {len(test_files)} test files in batches of {batch_size}")
    
    def process_file_batch(file_batch):
        """Process a batch of files."""
        batch_results = []
        for file_path in file_batch:
            try:
                # Load data
                data = np.load(file_path, mmap_mode='r')
                
                # Preprocess with caching
                processed_data = preprocess_one_cached(
                    data, dt_decimate=dt_decimate, is_seismic=True, 
                    use_cache=use_cache
                )
                
                # Save preprocessed file
                output_file = output_dir / f"preprocessed_{file_path.stem}.npy"
                np.save(output_file, processed_data)
                batch_results.append(output_file)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
                
        return batch_results
    
    # Process files in batches
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for i in range(0, len(test_files), batch_size):
            batch = test_files[i:i + batch_size]
            future = executor.submit(process_file_batch, batch)
            futures.append(future)
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                batch_results = future.result()
                processed_files.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
    
    logger.info(f"Successfully processed {len(processed_files)} test files")
    return processed_files

def create_inference_optimized_dataset(test_files: List[Path],
                                     output_dir: Path,
                                     chunk_size: Tuple[int, ...] = (1, 4, 64, 8),
                                     use_cache: bool = True) -> Path:
    """
    Create an inference-optimized zarr dataset for test data.
    
    Args:
        test_files: List of test file paths
        output_dir: Directory to save the dataset
        chunk_size: Chunk size for zarr dataset
        use_cache: Whether to use preprocessing cache
        
    Returns:
        Path to the created zarr dataset
    """
    logger.info(f"Creating inference-optimized dataset for {len(test_files)} test files")
    
    # Preprocess all test files
    preprocessed_files = preprocess_test_data_batch(
        test_files, output_dir / 'temp', use_cache=use_cache
    )
    
    # Create zarr dataset
    zarr_path = output_dir / 'test_data.zarr'
    
    try:
        # Load all preprocessed data
        all_data = []
        for file_path in tqdm(preprocessed_files, desc="Loading preprocessed data"):
            data = np.load(file_path)
            all_data.append(data)
        
        # Stack into single array
        stacked_data = np.stack(all_data, axis=0)
        logger.info(f"Stacked data shape: {stacked_data.shape}")
        
        # Save as zarr dataset
        import dask.array as da
        dask_array = da.from_array(stacked_data, chunks=chunk_size)
        dask_array.to_zarr(str(zarr_path))
        
        logger.info(f"Created inference dataset at {zarr_path}")
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(output_dir / 'temp')
        
        return zarr_path
        
    except Exception as e:
        logger.error(f"Failed to create inference dataset: {e}")
        raise

def preprocess_for_inference(data: np.ndarray, 
                           dt_decimate: int = 4,
                           use_cache: bool = True) -> np.ndarray:
    """
    Inference-optimized preprocessing function that matches training preprocessing exactly.
    
    Args:
        data: Input seismic data
        dt_decimate: Downsampling factor (must match training)
        use_cache: Whether to use caching
        
    Returns:
        Preprocessed data in float16
    """
    return preprocess_one_cached(
        data, dt_decimate=dt_decimate, is_seismic=True, use_cache=use_cache
    )

if __name__ == "__main__":
    main() 