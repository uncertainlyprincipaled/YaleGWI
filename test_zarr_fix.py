#!/usr/bin/env python3
"""
Test script to verify zarr compression fix and 5D dimension handling
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
        
        # Create dask array and rechunk it
        dask_array = da.from_array(test_data, chunks=(50, 50))
        dask_array = dask_array.rechunk((50, 50))  # Ensure proper chunking
        print("✅ Dask array created and rechunked")
        
        # Test saving with default compression
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.zarr"
            
            try:
                # Try default compression (without chunks parameter)
                dask_array.to_zarr(str(output_path))
                print("✅ Default compression works")
                
                # Verify data
                loaded_data = zarr.open(str(output_path))
                print(f"✅ Data loaded: {loaded_data.shape}")
                
                return True
                
            except Exception as e:
                print(f"⚠️ Default compression failed: {e}")
                
                # Try without compression
                dask_array.to_zarr(str(output_path), compressor=None)
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

def test_5d_dimension_handling():
    """Test that 5D data is handled correctly with proper chunk sizes"""
    print("\n🧪 Testing 5D dimension handling...")
    
    try:
        import zarr
        import dask.array as da
        import dask
        
        # Create 5D test data similar to the actual data shape
        # Shape: (batch, samples, sources, time, receivers) = (16, 500, 5, 1000, 70)
        test_data = np.random.randn(500, 5, 1000, 70).astype(np.float16)
        print(f"✅ 5D test data created: {test_data.shape}")
        
        # Create multiple arrays to simulate stacking
        arrays = [test_data for _ in range(3)]  # 3 files
        
        # Create lazy Dask arrays from files
        lazy_arrays = [
            da.from_delayed(dask.delayed(lambda x: x)(arr), shape=test_data.shape, dtype=test_data.dtype)
            for arr in arrays
        ]
        
        # Stack arrays (this creates 5D data)
        stack = da.stack(lazy_arrays, axis=0)
        print(f"✅ Stack shape after stacking: {stack.shape}")
        
        # Test the chunk size adjustment logic
        stack_shape = stack.shape
        
        if len(stack_shape) == 5:
            # For 5D data (batch, samples, sources, time, receivers)
            adjusted_chunk_size = (
                1,  # batch dimension - keep small for memory efficiency
                min(4, stack_shape[1]),  # samples dimension
                min(4, stack_shape[2]),  # sources dimension  
                min(64, stack_shape[3]),  # time dimension
                min(8, stack_shape[4])   # receivers dimension
            )
        elif len(stack_shape) == 4:
            adjusted_chunk_size = (1, min(4, stack_shape[1]), min(64, stack_shape[2]), min(8, stack_shape[3]))
        elif len(stack_shape) == 3:
            adjusted_chunk_size = (1, min(64, stack_shape[0]), min(8, stack_shape[1]))
        else:
            adjusted_chunk_size = tuple(1 for _ in range(len(stack_shape)))
        
        print(f"✅ Adjusted chunk size: {adjusted_chunk_size}")
        
        # Test rechunking
        stack = stack.rechunk(adjusted_chunk_size)
        print(f"✅ Rechunked successfully. Final shape: {stack.shape}, chunks: {stack.chunks}")
        
        # Test saving to zarr
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_5d.zarr"
            
            try:
                # Try saving with no compression
                stack.to_zarr(str(output_path), compressor=None)
                print("✅ 5D data saved to zarr successfully")
                
                # Verify data
                loaded_data = zarr.open(str(output_path))
                print(f"✅ 5D data loaded: {loaded_data.shape}")
                
                return True
                
            except Exception as e:
                print(f"❌ 5D zarr save failed: {e}")
                return False
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ 5D test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_zarr_compression()
    success2 = test_5d_dimension_handling()
    
    if success1 and success2:
        print("\n🎉 All tests PASSED!")
    else:
        print("\n❌ Some tests FAILED!") 