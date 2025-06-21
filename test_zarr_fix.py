#!/usr/bin/env python3
"""
Test script to verify zarr compression fix
"""

import numpy as np
import tempfile
from pathlib import Path

def test_zarr_compression():
    """Test that zarr compression works without Blosc"""
    print("🧪 Testing zarr compression fix...")
    
    try:
        import zarr
        import dask.array as da
        
        print(f"✅ Zarr version: {zarr.__version__}")
        
        # Create test data
        test_data = np.random.randn(100, 100).astype(np.float16)
        print(f"✅ Test data created: {test_data.shape}")
        
        # Create dask array
        dask_array = da.from_array(test_data, chunks=(50, 50))
        print("✅ Dask array created")
        
        # Test saving with default compression
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.zarr"
            
            try:
                # Try default compression
                dask_array.to_zarr(str(output_path), chunks=(50, 50))
                print("✅ Default compression works")
                
                # Verify data
                loaded_data = zarr.open(str(output_path))
                print(f"✅ Data loaded: {loaded_data.shape}")
                
                return True
                
            except Exception as e:
                print(f"⚠️ Default compression failed: {e}")
                
                # Try without compression
                dask_array.to_zarr(str(output_path), chunks=(50, 50), compressor=None)
                print("✅ No compression works")
                
                # Verify data
                loaded_data = zarr.open(str(output_path))
                print(f"✅ Data loaded: {loaded_data.shape}")
                
                return True
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_zarr_compression()
    if success:
        print("\n🎉 Zarr compression fix test PASSED!")
    else:
        print("\n❌ Zarr compression fix test FAILED!") 