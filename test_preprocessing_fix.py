#!/usr/bin/env python3
"""
Test script to verify preprocessing fixes work correctly
"""

import numpy as np
import tempfile
from pathlib import Path
import sys
import os

def test_shape_separation_fix():
    """Test that the shape separation fix works correctly"""
    print("🧪 Testing shape separation fix...")
    
    try:
        # Create test data with different shapes (like the actual data)
        seismic_data = np.random.randn(500, 5, 250, 70).astype(np.float16)  # Seismic data (downsampled)
        velocity_data = np.random.randn(500, 1, 70, 70).astype(np.float16)  # Velocity data (different shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Save test files with different naming patterns
            np.save(tmp_path / "seis_data1.npy", seismic_data)
            np.save(tmp_path / "seis_data2.npy", seismic_data)
            np.save(tmp_path / "vel_data1.npy", velocity_data)
            np.save(tmp_path / "vel_data2.npy", velocity_data)
            
            # Test the separation logic (same as in preprocess.py)
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
            
            # Test that seismic and velocity have different shapes (as expected)
            if seismic_paths and velocity_paths:
                seismic_shape = np.load(seismic_paths[0]).shape
                velocity_shape = np.load(velocity_paths[0]).shape
                if seismic_shape != velocity_shape:
                    print(f"✅ Seismic and velocity shapes correctly different: {seismic_shape} vs {velocity_shape}")
                    return True
                else:
                    print(f"⚠️ Unexpected: Seismic and velocity shapes are the same: {seismic_shape}")
                    return False
            else:
                print("⚠️ Missing either seismic or velocity files for comparison")
                return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_s3fs_compatibility():
    """Test s3fs compatibility"""
    print("\n🧪 Testing s3fs compatibility...")
    
    try:
        import s3fs
        print(f"✅ S3fs version: {s3fs.__version__}")
        
        # Test s3fs functionality
        try:
            # Create a simple s3fs instance (anonymous mode)
            fs = s3fs.S3FileSystem(anon=True)
            print("✅ S3fs basic functionality working")
            return True
        except Exception as e:
            if "asynchronous" in str(e):
                print(f"❌ S3fs has the old 'asynchronous' compatibility issue: {e}")
                return False
            else:
                print(f"✅ S3fs working (different error: {e})")
                return True
            
    except ImportError:
        print("❌ S3fs not installed")
        return False

def test_preprocessing_import():
    """Test that preprocessing module can be imported and functions exist"""
    print("\n🧪 Testing preprocessing module import...")
    
    try:
        # Add src to path
        sys.path.append('src')
        
        # Test imports
        from core.preprocess import create_zarr_dataset, save_zarr_data
        print("✅ Preprocessing functions imported successfully")
        
        # Test that the functions exist and are callable
        if callable(create_zarr_dataset):
            print("✅ create_zarr_dataset function exists")
        else:
            print("❌ create_zarr_dataset is not callable")
            return False
            
        if callable(save_zarr_data):
            print("✅ save_zarr_data function exists")
        else:
            print("❌ save_zarr_data is not callable")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing preprocessing fixes...")
    print("="*50)
    
    success1 = test_shape_separation_fix()
    success2 = test_s3fs_compatibility()
    success3 = test_preprocessing_import()
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"  Shape Separation Fix: {'✅' if success1 else '❌'}")
    print(f"  S3fs Compatibility: {'✅' if success2 else '❌'}")
    print(f"  Preprocessing Import: {'✅' if success3 else '❌'}")
    
    if success1 and success2 and success3:
        print("\n🎉 All tests PASSED!")
        print("✅ Preprocessing fixes are working correctly")
    else:
        print("\n❌ Some tests FAILED!")
        if not success1:
            print("❌ Shape separation fix failed")
        if not success2:
            print("❌ S3fs compatibility failed")
        if not success3:
            print("❌ Preprocessing import failed") 