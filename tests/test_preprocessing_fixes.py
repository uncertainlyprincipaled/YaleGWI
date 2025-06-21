#!/usr/bin/env python3
"""
Comprehensive test script to verify preprocessing fixes work correctly.
This tests the s3fs version fix, test path updates, and preprocessing functionality.
"""

import sys
import os
import tempfile
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append('/content/YaleGWI/src')

def test_s3fs_version_fix():
    """Test that s3fs version detection and update works correctly."""
    print("🧪 Testing s3fs version fix...")
    
    try:
        from src.utils.colab_setup import check_and_fix_s3fs_installation
        
        # Test the s3fs check function
        result = check_and_fix_s3fs_installation()
        
        if result:
            print("  ✅ S3fs version fix working correctly")
            return True
        else:
            print("  ❌ S3fs version fix failed")
            return False
            
    except Exception as e:
        print(f"  ❌ S3fs version fix test failed: {e}")
        return False

def test_preprocessing_functionality():
    """Test that preprocessing functions work correctly."""
    print("🧪 Testing preprocessing functionality...")
    
    try:
        from src.core.preprocess import preprocess_one, validate_nyquist, PreprocessingFeedback
        
        # Test with mock data
        seis_4d = np.random.randn(500, 5, 2000, 70).astype(np.float32)
        feedback = PreprocessingFeedback()
        
        result = preprocess_one(seis_4d, dt_decimate=4, is_seismic=True, feedback=feedback)
        
        # Check that downsampling worked correctly
        if result.shape[2] == 500:  # Should be downsampled from 2000 to 500
            print("  ✅ Preprocessing downsampling working correctly")
        else:
            print(f"  ❌ Preprocessing shape mismatch: expected time dim 500, got {result.shape[2]}")
            return False
        
        # Check that data type conversion worked
        if result.dtype == np.float16:
            print("  ✅ Preprocessing dtype conversion working correctly")
        else:
            print(f"  ❌ Preprocessing dtype mismatch: expected float16, got {result.dtype}")
            return False
        
        # Test Nyquist validation
        nyquist_valid = validate_nyquist(seis_4d, dt_decimate=4, feedback=feedback)
        if nyquist_valid is not None:  # Should return True or False
            print("  ✅ Nyquist validation working correctly")
        else:
            print("  ❌ Nyquist validation failed")
            return False
        
        print("  ✅ Preprocessing functionality tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Preprocessing functionality test failed: {e}")
        return False

def test_shape_separation():
    """Test that seismic and velocity files are handled separately."""
    print("🧪 Testing shape separation...")
    
    try:
        from src.core.preprocess import create_zarr_dataset
        
        # Create test data with different shapes
        seismic_data = np.random.randn(500, 5, 250, 70).astype(np.float16)  # Seismic data (downsampled)
        velocity_data = np.random.randn(500, 1, 70, 70).astype(np.float16)  # Velocity data (different shape)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Save test files with the actual naming pattern
            np.save(tmp_path / "seis_data1.npy", seismic_data)
            np.save(tmp_path / "seis_data2.npy", seismic_data)
            np.save(tmp_path / "vel_data1.npy", velocity_data)
            np.save(tmp_path / "vel_data2.npy", velocity_data)
            
            # Get file paths
            processed_paths = [str(p) for p in tmp_path.glob("*.npy")]
            
            # Test create_zarr_dataset
            output_path = tmp_path / "test.zarr"
            
            # This should work without errors now
            create_zarr_dataset(
                processed_paths=processed_paths,
                output_path=output_path,
                chunk_size=(1, 4, 64, 8),
                data_manager=None  # Local mode
            )
            
            print("  ✅ Shape separation working correctly")
            return True
            
    except Exception as e:
        print(f"  ❌ Shape separation test failed: {e}")
        return False

def test_test_paths():
    """Test that test files are in the correct locations."""
    print("🧪 Testing test file paths...")
    
    try:
        # Check that test files are in the tests directory
        tests_dir = Path('/content/YaleGWI/tests')
        
        expected_test_files = [
            'test_colab_validation.py',
            'test_zarr_fix.py',
            'test_preprocessing_fix.py',
            'test_shape_fix.py',
            'test_final_fix.py'
        ]
        
        missing_files = []
        for test_file in expected_test_files:
            test_path = tests_dir / test_file
            if not test_path.exists():
                missing_files.append(test_file)
        
        if missing_files:
            print(f"  ❌ Missing test files: {missing_files}")
            return False
        else:
            print("  ✅ All test files in correct location")
            return True
            
    except Exception as e:
        print(f"  ❌ Test path check failed: {e}")
        return False

def test_colab_setup_imports():
    """Test that colab_setup can import all required modules."""
    print("🧪 Testing colab_setup imports...")
    
    try:
        from src.utils.colab_setup import (
            setup_colab_environment,
            check_and_fix_s3fs_installation,
            check_and_fix_zarr_installation,
            run_preprocessing,
            complete_colab_setup
        )
        
        print("  ✅ All colab_setup functions imported successfully")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline with mock data."""
    print("🧪 Testing complete preprocessing pipeline...")
    
    try:
        from src.core.preprocess import load_data
        
        # Create mock data structure
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create mock family directories
            families = ['FlatVel_A', 'CurveVel_A', 'Style_A']
            for family in families:
                family_dir = tmp_path / family
                family_dir.mkdir(parents=True, exist_ok=True)
                
                # Create mock seismic and velocity data
                seismic_data = np.random.randn(500, 5, 2000, 70).astype(np.float32)
                velocity_data = np.random.randn(500, 1, 70, 70).astype(np.float32)
                
                np.save(family_dir / "seis_data1.npy", seismic_data)
                np.save(family_dir / "vel_data1.npy", velocity_data)
            
            # Create output directory
            output_dir = tmp_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run preprocessing pipeline
            feedback = load_data(
                input_root=str(tmp_path),
                output_root=str(output_dir),
                use_s3=False  # Local mode for testing
            )
            
            # Check that preprocessing completed
            if feedback and len(feedback) > 0:
                print("  ✅ Preprocessing pipeline completed successfully")
                return True
            else:
                print("  ❌ Preprocessing pipeline failed")
                return False
                
    except Exception as e:
        print(f"  ❌ Preprocessing pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running comprehensive preprocessing fix tests...")
    print("="*60)
    
    results = {
        's3fs_version_fix': test_s3fs_version_fix(),
        'preprocessing_functionality': test_preprocessing_functionality(),
        'shape_separation': test_shape_separation(),
        'test_paths': test_test_paths(),
        'colab_setup_imports': test_colab_setup_imports(),
        'preprocessing_pipeline': test_preprocessing_pipeline()
    }
    
    print("\n" + "="*60)
    print("📊 Test Results Summary:")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n💡 All preprocessing fixes are working correctly!")
        print("   Ready to run the full Colab setup.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("   Fix the failing tests before proceeding with setup.")
    
    return all_passed

if __name__ == "__main__":
    main() 