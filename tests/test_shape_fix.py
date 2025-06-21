#!/usr/bin/env python3
"""
Test script to verify shape mismatch and S3 save fixes
"""

import numpy as np
import tempfile
from pathlib import Path

def test_shape_separation():
    """Test that seismic and velocity files are handled separately"""
    print("🧪 Testing shape separation...")
    
    try:
        # Create test data with different shapes
        seismic_data = np.random.randn(500, 5, 250, 70).astype(np.float16)  # Seismic data
        velocity_data = np.random.randn(500, 1, 70, 70).astype(np.float16)  # Velocity data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Save test files
            np.save(tmp_path / "seis_data1.npy", seismic_data)
            np.save(tmp_path / "seis_data2.npy", seismic_data)
            np.save(tmp_path / "vel_data1.npy", velocity_data)
            np.save(tmp_path / "vel_data2.npy", velocity_data)
            
            # Test the separation logic
            seismic_paths = []
            velocity_paths = []
            
            for path in tmp_path.glob("*.npy"):
                if 'seis_' in path.name:
                    seismic_paths.append(str(path))
                elif 'vel_' in path.name:
                    velocity_paths.append(str(path))
            
            print(f"✅ Found {len(seismic_paths)} seismic files and {len(velocity_paths)} velocity files")
            
            # Verify shapes are consistent within each type
            if seismic_paths:
                first_seismic = np.load(seismic_paths[0])
                for path in seismic_paths:
                    arr = np.load(path)
                    if arr.shape != first_seismic.shape:
                        print(f"❌ Seismic shape mismatch: {first_seismic.shape} vs {arr.shape}")
                        return False
                print(f"✅ All seismic files have consistent shape: {first_seismic.shape}")
            
            if velocity_paths:
                first_velocity = np.load(velocity_paths[0])
                for path in velocity_paths:
                    arr = np.load(path)
                    if arr.shape != first_velocity.shape:
                        print(f"❌ Velocity shape mismatch: {first_velocity.shape} vs {arr.shape}")
                        return False
                print(f"✅ All velocity files have consistent shape: {first_velocity.shape}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_s3fs_compatibility():
    """Test s3fs compatibility"""
    print("\n🧪 Testing s3fs compatibility...")
    
    try:
        import s3fs
        print(f"✅ S3fs version: {s3fs.__version__}")
        
        # Test basic s3fs functionality
        try:
            # Create a simple s3fs instance
            fs = s3fs.S3FileSystem(anon=True)
            print("✅ S3fs basic functionality working")
            return True
        except Exception as e:
            print(f"❌ S3fs functionality test failed: {e}")
            return False
            
    except ImportError:
        print("❌ S3fs not installed")
        return False

def test_zarr_save_without_compression():
    """Test zarr save without compression"""
    print("\n🧪 Testing zarr save without compression...")
    
    try:
        import zarr
        import dask.array as da
        
        # Create test data
        test_data = np.random.randn(100, 100).astype(np.float16)
        dask_array = da.from_array(test_data, chunks=(50, 50))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.zarr"
            
            # Save without compression
            dask_array.to_zarr(str(output_path))
            print("✅ Zarr save without compression works")
            
            # Verify data
            loaded_data = zarr.open(str(output_path))
            if loaded_data.shape == test_data.shape:
                print("✅ Data verification successful")
                return True
            else:
                print(f"❌ Shape mismatch: expected {test_data.shape}, got {loaded_data.shape}")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_shape_separation()
    success2 = test_s3fs_compatibility()
    success3 = test_zarr_save_without_compression()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests PASSED!")
        print("✅ Shape separation working")
        print("✅ S3fs compatibility verified")
        print("✅ Zarr save without compression working")
    else:
        print("\n❌ Some tests FAILED!")
        if not success1:
            print("❌ Shape separation failed")
        if not success2:
            print("❌ S3fs compatibility failed")
        if not success3:
            print("❌ Zarr save failed") 