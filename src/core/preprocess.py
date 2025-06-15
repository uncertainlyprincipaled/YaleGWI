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
from src.core.config import CFG

# Constants for preprocessing
CHUNK_TIME = 256  # After decimating by 4
CHUNK_SRC_REC = 8
DT_DECIMATE = 4  # 1 kHz → 250 Hz

def preprocess_one(arr: np.ndarray) -> np.ndarray:
    """Preprocess a single seismic array with downsampling and normalization."""
    # Decimate time axis with anti-aliasing filter
    arr = decimate(arr, DT_DECIMATE, axis=1, ftype='fir')
    
    # Convert to float16
    arr = arr.astype('float16')
    
    # Robust normalization per trace
    μ = np.median(arr, keepdims=True)
    σ = np.percentile(arr, 95, keepdims=True) - np.percentile(arr, 5, keepdims=True)
    arr = (arr - μ) / σ
    
    return arr

def process_family(family: str, input_dir: Path, output_dir: Path) -> list:
    """Process all files in a family and return paths to processed files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_dir.glob('*.npy'))
    processed_paths = []
    
    for f in tqdm(files, desc=f"Processing {family}"):
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
            print(f"[!] Unexpected shape {arr.shape} in {f}")
    
    return processed_paths

def create_zarr_dataset(processed_paths: list, output_path: Path, chunk_size: tuple):
    """Create a zarr dataset from processed files."""
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

def split_for_gpus(processed_paths: list, output_base: Path):
    """Split processed files into two datasets for the two T4 GPUs."""
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
        (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC)
    )
    create_zarr_dataset(
        gpu1_paths,
        gpu1_dir / 'seismic.zarr',
        (1, CHUNK_SRC_REC, CHUNK_TIME, CHUNK_SRC_REC)
    )

def main():
    parser = argparse.ArgumentParser(description="Preprocess seismic data for distributed training on T4 GPUs")
    parser.add_argument('--input_root', type=str, default=str(CFG.paths.train), help='Input train_samples root directory')
    parser.add_argument('--output_root', type=str, default='/kaggle/working/preprocessed', help='Output directory for processed files')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Process each family
    families = list(CFG.paths.families.keys())
    all_processed_paths = []
    
    for family in families:
        print(f"\nProcessing family: {family}")
        input_dir = input_root / family
        temp_dir = output_root / 'temp' / family
        processed_paths = process_family(family, input_dir, temp_dir)
        all_processed_paths.extend(processed_paths)
        print(f"Family {family}: {len(processed_paths)} samples processed")

    # Split and create zarr datasets for GPUs
    print("\nCreating GPU-specific datasets...")
    split_for_gpus(all_processed_paths, output_root)
    
    # Clean up temporary files
    temp_dir = output_root / 'temp'
    if temp_dir.exists():
        subprocess.run(['rm', '-rf', str(temp_dir)])
    
    print("\nPreprocessing complete!")
    print(f"GPU 0 dataset: {output_root}/gpu0/seismic.zarr")
    print(f"GPU 1 dataset: {output_root}/gpu1/seismic.zarr")

if __name__ == "__main__":
    main() 