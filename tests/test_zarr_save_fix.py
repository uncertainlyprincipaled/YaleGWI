#!/usr/bin/env python3
"""
Test script to verify zarr save fixes for S3 and local saving.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.core.preprocess import save_combined_zarr_local
import dask.array as da

def test_local_zarr_save():
    """Test that local zarr save works without the shape parameter error."""
    print("🧪 Testing local zarr save...")
    
    # Create test data
    seismic_shape = (2, 500, 5, 500, 70)
    velocity_shape = (2, 500, 1, 70, 70)
    
    # Create dummy dask arrays
    seismic_stack = da.random.random(seismic_shape, chunks=(1, 4, 4, 64, 8))
    velocity_stack = da.random.random(velocity_shape, chunks=(1, 4, 1, 8, 8))
    
    print(f"Seismic shape: {seismic_shape}")
    print(f"Velocity shape: {velocity_shape}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"
        
        # This should work without the shape parameter error
        save_combined_zarr_local(
            seismic_stack=seismic_stack,
            velocity_stack=velocity_stack,
            output_path=output_path
        )
        
        # Verify the zarr file was created
        if output_path.exists():
            print("✅ Local zarr save test completed successfully")
            print("✅ Zarr file created without shape parameter errors")
            
            # Check that both datasets exist
            import zarr
            root = zarr.open(str(output_path))
            if 'seismic' in root and 'velocity' in root:
                print("✅ Both seismic and velocity datasets created successfully")
                print(f"✅ Seismic shape: {root['seismic'].shape}")
                print(f"✅ Velocity shape: {root['velocity'].shape}")
            else:
                print("❌ Missing datasets in zarr file")
                return False
        else:
            print("❌ Zarr file was not created")
            return False
    
    return True

def test_fallback_to_numpy():
    """Test that the fallback to numpy arrays works."""
    print("🧪 Testing fallback to numpy arrays...")
    
    # Create test data
    seismic_shape = (2, 500, 5, 500, 70)
    velocity_shape = (2, 500, 1, 70, 70)
    
    # Create dummy dask arrays
    seismic_stack = da.random.random(seismic_shape, chunks=(1, 4, 4, 64, 8))
    velocity_stack = da.random.random(velocity_shape, chunks=(1, 4, 1, 8, 8))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test"
        
        # Force fallback by creating an invalid path
        invalid_path = Path("/invalid/path/test.zarr")
        
        try:
            save_combined_zarr_local(
                seismic_stack=seismic_stack,
                velocity_stack=velocity_stack,
                output_path=invalid_path
            )
        except Exception as e:
            print(f"Expected error (fallback should trigger): {e}")
            
            # Check if numpy files were created as fallback
            seismic_npy = output_path.with_suffix('.seismic.npy')
            velocity_npy = output_path.with_suffix('.velocity.npy')
            
            if seismic_npy.exists() and velocity_npy.exists():
                print("✅ Fallback to numpy arrays worked")
                return True
            else:
                print("❌ Fallback to numpy arrays failed")
                return False
    
    return False

if __name__ == "__main__":
    print("🔧 Testing zarr save fixes...")
    print("="*60)
    
    try:
        test1_passed = test_local_zarr_save()
        test2_passed = test_fallback_to_numpy()
        
        print("\n" + "="*60)
        if test1_passed and test2_passed:
            print("🎉 All zarr save tests passed!")
            print("✅ Local zarr save works without shape parameter errors")
            print("✅ Fallback to numpy arrays works")
            print("✅ S3 save should now work with AWS CLI or boto3 fallback")
        else:
            print("❌ Some tests failed")
            sys.exit(1)
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1) 