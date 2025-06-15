import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
import zarr
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.signal import decimate
import logging
from typing import Tuple, List, Optional
import warnings
import boto3
from botocore.exceptions import ClientError
import json
from src.core.config import CFG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for preprocessing
CHUNK_TIME = 256  # After decimating by 4
CHUNK_SRC_REC = 8
DT_DECIMATE = 4  # 1 kHz → 250 Hz
NYQUIST_FREQ = 500  # Hz (half of original sampling rate)

def get_aws_credentials() -> dict:
    """
    Get AWS credentials from Kaggle secrets or environment variables.
    Prioritizes Kaggle secrets for security in Kaggle environment.
    """
    credentials = {}
    
    try:
        # Try Kaggle secrets first
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        
        # Get AWS credentials from Kaggle secrets
        credentials = {
            'aws_access_key_id': secrets.get_secret('aws_access_key_id'),
            'aws_secret_access_key': secrets.get_secret('aws_secret_access_key'),
            'region_name': secrets.get_secret('aws_region', default='us-east-1'),
            's3_bucket': secrets.get_secret('aws_s3_bucket')  # Add S3 bucket from secrets
        }
        
        # Validate that we got the required credentials
        if not all(v for k, v in credentials.items() if k != 'region_name'):  # region_name is optional
            raise ValueError("Missing required AWS credentials in Kaggle secrets")
            
    except ImportError:
        # Not in Kaggle environment, try environment variables
        if all(k in os.environ for k in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_S3_BUCKET']):
            credentials = {
                'aws_access_key_id': os.environ['AWS_ACCESS_KEY_ID'],
                'aws_secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY'],
                'region_name': os.environ.get('AWS_REGION', 'us-east-1'),
                's3_bucket': os.environ['AWS_S3_BUCKET']
            }
        else:
            # Fall back to config file
            try:
                with open('.env/aws/credentials.json', 'r') as f:
                    credentials = json.load(f)
            except FileNotFoundError:
                raise RuntimeError("AWS credentials not found in Kaggle secrets, environment variables, or config file")
    
    return credentials

def upload_to_s3(local_path: Path, s3_bucket: str, s3_key: str) -> None:
    """
    Upload a file to S3 with progress tracking.
    
    Args:
        local_path: Local file path
        s3_bucket: S3 bucket name
        s3_key: S3 object key
    """
    try:
        credentials = get_aws_credentials()
        s3_client = boto3.client('s3', **credentials)
        
        # Get file size for progress bar
        file_size = os.path.getsize(local_path)
        
        # Upload with progress tracking
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {local_path.name}") as pbar:
            s3_client.upload_file(
                str(local_path),
                s3_bucket,
                s3_key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
        
        logger.info(f"Successfully uploaded {local_path.name} to s3://{s3_bucket}/{s3_key}")
    except ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise

def validate_nyquist(data: np.ndarray, original_fs: int = 1000) -> bool:
    """
    Validate that the data satisfies Nyquist criterion after downsampling.
    
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

def process_family(family: str, input_dir: Path, output_dir: Path) -> List[str]:
    """
    Process all files in a family and return paths to processed files.
    
    Args:
        family: Name of the geological family
        input_dir: Input directory containing raw data
        output_dir: Output directory for processed data
        
    Returns:
        List[str]: Paths to processed files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_dir.glob('*.npy'))
    processed_paths = []
    
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

def create_zarr_dataset(processed_paths: List[str], output_path: Path, chunk_size: Tuple[int, ...], s3_bucket: Optional[str] = None) -> None:
    """
    Create a zarr dataset from processed files and optionally upload to S3.
    
    Args:
        processed_paths: List of paths to processed files
        output_path: Path to save zarr dataset
        chunk_size: Chunk size for zarr array
        s3_bucket: Optional S3 bucket for offloading
    """
    try:
        # Create lazy Dask arrays
        lazy_arrays = []
        for path in processed_paths:
            x = da.from_delayed(
                dask.delayed(np.load)(path),
                shape=(32, 256, 64),  # Example dims after decimation
                dtype='float16'
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
        
        # Upload to S3 if bucket specified
        if s3_bucket:
            s3_key = f"preprocessed/{output_path.name}"
            upload_to_s3(output_path, s3_bucket, s3_key)
            
            # Clean up local file after successful upload
            if output_path.exists():
                output_path.unlink()
                logger.info(f"Cleaned up local file {output_path}")
    except Exception as e:
        logger.error(f"Error creating/uploading zarr dataset: {str(e)}")
        raise

def split_for_gpus(processed_paths: List[str], output_base: Path, s3_bucket: Optional[str] = None) -> None:
    """
    Split processed files into two datasets for the two T4 GPUs and optionally upload to S3.
    
    Args:
        processed_paths: List of paths to processed files
        output_base: Base directory for output
        s3_bucket: Optional S3 bucket for offloading
    """
    try:
        n_samples = len(processed_paths)
        mid_point = n_samples // 2
        
        # Create GPU-specific directories
        gpu0_dir = output_base / 'gpu0'
        gpu1_dir = output_base / 'gpu1'
        gpu0_dir.mkdir(parents=True, exist_ok=True)
        gpu1_dir.mkdir(parents=True, exist_ok=True)
        
        # Split paths
        gpu0_paths = processed_paths[:mid_point]
        gpu1_paths = processed_paths[mid_point:]
        
        # Create zarr datasets for each GPU
        create_zarr_dataset(
            gpu0_paths,
            gpu0_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            s3_bucket
        )
        create_zarr_dataset(
            gpu1_paths,
            gpu1_dir / 'seismic.zarr',
            (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC),
            s3_bucket
        )
        
        logger.info(f"Created GPU datasets with {len(gpu0_paths)} and {len(gpu1_paths)} samples")
    except Exception as e:
        logger.error(f"Error splitting data for GPUs: {str(e)}")
        raise

def main():
    # Filter out Jupyter/Colab specific arguments
    import sys
    filtered_args = [arg for arg in sys.argv if not arg.startswith('-f') and not arg.endswith('.json')]
    sys.argv = filtered_args

    parser = argparse.ArgumentParser(description="Preprocess seismic data for distributed training on T4 GPUs")
    parser.add_argument('--input_root', type=str, default=str(CFG.paths.train), help='Input train_samples root directory')
    parser.add_argument('--output_root', type=str, default='/kaggle/working/preprocessed', help='Output directory for processed files')
    parser.add_argument('--s3_bucket', type=str, help='S3 bucket for offloading preprocessed data (overrides secret/config)')
    args = parser.parse_args()

    try:
        input_root = Path(args.input_root)
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        # Get credentials including S3 bucket
        credentials = get_aws_credentials()
        s3_bucket = args.s3_bucket or credentials.get('s3_bucket')
        
        if not s3_bucket:
            raise ValueError("S3 bucket not specified. Please provide it via --s3_bucket argument or in credentials")

        # Process each family
        families = list(CFG.paths.families.keys())
        all_processed_paths = []
        
        for family in families:
            logger.info(f"\nProcessing family: {family}")
            input_dir = input_root / family
            temp_dir = output_root / 'temp' / family
            processed_paths = process_family(family, input_dir, temp_dir)
            all_processed_paths.extend(processed_paths)
            logger.info(f"Family {family}: {len(processed_paths)} samples processed")

        # Split and create zarr datasets for GPUs
        logger.info("\nCreating GPU-specific datasets...")
        split_for_gpus(all_processed_paths, output_root, s3_bucket)
        
        # Clean up temporary files
        temp_dir = output_root / 'temp'
        if temp_dir.exists():
            subprocess.run(['rm', '-rf', str(temp_dir)])
        
        logger.info("\nPreprocessing complete!")
        logger.info(f"Data uploaded to s3://{s3_bucket}/preprocessed/")
    except Exception as e:
        logger.error(f"Error in main preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 