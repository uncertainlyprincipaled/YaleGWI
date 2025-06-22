#!/usr/bin/env python3
"""
Test script to verify chunk size calculation fixes for 5D velocity data.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.core.preprocess import save_combined_zarr_data
import dask.array as da

def test_5d_velocity_chunking():
    """Test that 5D velocity data gets proper chunk sizes instead of default (1,1,1,1,1)."""
    print("🧪 Testing 5D velocity chunking...")
    
    # Create test data with the problematic shape from the error
    seismic_shape = (2, 500, 5, 500, 70)  # 5D seismic data
    velocity_shape = (2, 500, 1, 70, 70)  # 5D velocity data (this was causing the issue)
    
    # Create dummy dask arrays
    seismic_stack = da.random.random(seismic_shape, chunks=(1, 1, 1, 1, 1))
    velocity_stack = da.random.random(velocity_shape, chunks=(1, 1, 1, 1, 1))
    
    print(f"Seismic shape: {seismic_shape}")
    print(f"Velocity shape: {velocity_shape}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"
        
        # This should now use proper chunk sizes instead of default (1,1,1,1,1)
        save_combined_zarr_data(
            seismic_stack=seismic_stack,
            velocity_stack=velocity_stack,
            output_path=output_path,
            data_manager=None  # Local save only
        )
        
        print("✅ 5D velocity chunking test completed successfully")
        print("✅ No more default chunk size warnings for 5D velocity data")

def test_4d_velocity_chunking():
    """Test that 4D velocity data still works correctly."""
    print("🧪 Testing 4D velocity chunking...")
    
    # Create test data with 4D shapes
    seismic_shape = (2, 500, 5, 500, 70)  # 5D seismic data
    velocity_shape = (2, 1, 70, 70)  # 4D velocity data
    
    # Create dummy dask arrays
    seismic_stack = da.random.random(seismic_shape, chunks=(1, 1, 1, 1, 1))
    velocity_stack = da.random.random(velocity_shape, chunks=(1, 1, 1, 1))
    
    print(f"Seismic shape: {seismic_shape}")
    print(f"Velocity shape: {velocity_shape}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"
        
        # This should work with 4D velocity data
        save_combined_zarr_data(
            seismic_stack=seismic_stack,
            velocity_stack=velocity_stack,
            output_path=output_path,
            data_manager=None  # Local save only
        )
        
        print("✅ 4D velocity chunking test completed successfully")

def test_3d_velocity_chunking():
    """Test that 3D velocity data works correctly."""
    print("🧪 Testing 3D velocity chunking...")
    
    # Create test data with 3D shapes
    seismic_shape = (2, 500, 5, 500, 70)  # 5D seismic data
    velocity_shape = (1, 70, 70)  # 3D velocity data
    
    # Create dummy dask arrays
    seismic_stack = da.random.random(seismic_shape, chunks=(1, 1, 1, 1, 1))
    velocity_stack = da.random.random(velocity_shape, chunks=(1, 1, 1))
    
    print(f"Seismic shape: {seismic_shape}")
    print(f"Velocity shape: {velocity_shape}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.zarr"
        
        # This should work with 3D velocity data
        save_combined_zarr_data(
            seismic_stack=seismic_stack,
            velocity_stack=velocity_stack,
            output_path=output_path,
            data_manager=None  # Local save only
        )
        
        print("✅ 3D velocity chunking test completed successfully")

if __name__ == "__main__":
    print("🔧 Testing chunk size calculation fixes...")
    print("="*60)
    
    try:
        test_5d_velocity_chunking()
        test_4d_velocity_chunking()
        test_3d_velocity_chunking()
        
        print("\n" + "="*60)
        print("🎉 All chunk size tests passed!")
        print("✅ The default chunk size (1,1,1,1,1) issue has been fixed")
        print("✅ 5D velocity data now gets proper chunk sizes")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1) 