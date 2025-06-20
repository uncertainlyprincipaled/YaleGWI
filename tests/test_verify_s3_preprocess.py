"""
Verify Preprocessed Data on S3

This script checks for the existence and integrity of the preprocessed,
GPU-specific Zarr datasets stored in an S3 bucket.

Workflow:
1.  Connects to S3 using credentials from environment variables.
2.  Reads the S3 bucket and prefix from the project's config file.
3.  Looks for 'gpu0/seismic.zarr' and 'gpu1/seismic.zarr'.
4.  Opens the Zarr datasets directly from S3.
5.  Prints a verification report with shapes, dtypes, and sample counts.

Usage:
    # Ensure AWS credentials are set in your environment:
    # export AWS_ACCESS_KEY_ID="YOUR_KEY"
    # export AWS_SECRET_ACCESS_KEY="YOUR_SECRET"
    # export AWS_REGION="us-east-1"

    # Run the verification from the project root
    python -m tests.test_verify_s3_preprocess
"""
import os
import sys
import zarr
import s3fs
from pathlib import Path

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.core.config import CFG

def verify_s3_data():
    """
    Connects to S3 and verifies the preprocessed Zarr datasets.
    """
    print("🚀 Starting S3 Preprocessed Data Verification...")

    # 1. Get S3 configuration from CFG
    try:
        s3_bucket = CFG.s3_paths.bucket
        s3_prefix = CFG.s3_paths.preprocessed_prefix
        if not s3_bucket or not s3_prefix:
            print("❌ Error: S3 bucket or prefix not defined in config.py.")
            print("Please check `CFG.s3_paths.bucket` and `CFG.s3_paths.preprocessed_prefix`.")
            return
    except Exception as e:
        print(f"❌ Error reading S3 configuration: {e}")
        return

    print(f"✅ S3 Config Loaded: Checking bucket '{s3_bucket}' at prefix '{s3_prefix}'")

    # 2. Connect to S3
    try:
        s3 = s3fs.S3FileSystem(
            key=os.environ.get('AWS_ACCESS_KEY_ID'),
            secret=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            client_kwargs={'region_name': os.environ.get('AWS_REGION', 'us-east-1')}
        )
        print("✅ Connected to S3 successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to S3: {e}")
        print("Please ensure your AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) are set as environment variables.")
        return

    # 3. Define paths and check for existence
    gpu0_path_s3 = f"{s3_bucket}/{s3_prefix}/gpu0/seismic.zarr"
    gpu1_path_s3 = f"{s3_bucket}/{s3_prefix}/gpu1/seismic.zarr"

    gpu0_exists = s3.exists(gpu0_path_s3)
    gpu1_exists = s3.exists(gpu1_path_s3)

    if not (gpu0_exists and gpu1_exists):
        print("❌ Verification Failed: GPU-specific Zarr datasets not found on S3.")
        if not gpu0_exists:
            print(f"  - Missing: s3://{gpu0_path_s3}")
        if not gpu1_exists:
            print(f"  - Missing: s3://{gpu1_path_s3}")
        return

    print("✅ Both gpu0 and gpu1 Zarr datasets found on S3.")

    # 4. Open Zarr stores and print report
    try:
        print("\n--- Verification Report ---")
        s3_map0 = s3fs.S3Map(root=gpu0_path_s3, s3=s3, check=False)
        data0 = zarr.open(s3_map0, mode='r')

        s3_map1 = s3fs.S3Map(root=gpu1_path_s3, s3=s3, check=False)
        data1 = zarr.open(s3_map1, mode='r')

        print(f"✅ GPU0 Dataset: {data0.shape} samples, dtype: {data0.dtype}")
        print(f"✅ GPU1 Dataset: {data1.shape} samples, dtype: {data1.dtype}")

        total_samples = data0.shape[0] + data1.shape[0]
        print("---------------------------")
        print(f"✅ Total Samples Found: {total_samples}")
        print("---------------------------")
        print("\n🎉 Verification successful! Your S3 data is ready for training.")

    except Exception as e:
        print(f"\n❌ Verification Failed: Could not open Zarr datasets from S3.")
        print(f"   Error: {e}")
        print("   The data might be corrupted, or there could be a permissions issue.")

if __name__ == "__main__":
    verify_s3_data() 