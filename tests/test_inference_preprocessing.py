#!/usr/bin/env python3
"""
Test script to verify inference-optimized preprocessing functions.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tempfile
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.core.preprocess import (
    preprocess_one, preprocess_one_cached, preprocess_for_inference,
    preprocess_test_data_batch, create_inference_optimized_dataset,
    PreprocessingCache
)

def test_preprocessing_consistency():
    """Test that inference preprocessing matches training preprocessing exactly."""
    print("🧪 Testing preprocessing consistency...")
    
    # Create test data
    test_data = np.random.randn(500, 5, 2000, 70).astype(np.float32)
    
    # Test original preprocessing
    result_original = preprocess_one(test_data, dt_decimate=4, is_seismic=True)
    
    # Test cached preprocessing
    result_cached = preprocess_one_cached(test_data, dt_decimate=4, is_seismic=True, use_cache=True)
    
    # Test inference preprocessing
    result_inference = preprocess_for_inference(test_data, dt_decimate=4, use_cache=True)
    
    # Verify all results are identical
    assert np.allclose(result_original, result_cached, rtol=1e-5), "Cached result differs from original"
    assert np.allclose(result_original, result_inference, rtol=1e-5), "Inference result differs from original"
    
    print("✅ Preprocessing consistency verified")
    print(f"  Original shape: {result_original.shape}, dtype: {result_original.dtype}")
    print(f"  Cached shape: {result_cached.shape}, dtype: {result_cached.dtype}")
    print(f"  Inference shape: {result_inference.shape}, dtype: {result_inference.dtype}")
    
    return True

def test_caching_functionality():
    """Test that preprocessing caching works correctly."""
    print("🧪 Testing caching functionality...")
    
    # Create test data
    test_data = np.random.randn(100, 5, 2000, 70).astype(np.float32)
    
    # First call - should process and cache
    start_time = time.time()
    result1 = preprocess_one_cached(test_data, dt_decimate=4, is_seismic=True, use_cache=True)
    time1 = time.time() - start_time
    
    # Second call - should use cache
    start_time = time.time()
    result2 = preprocess_one_cached(test_data, dt_decimate=4, is_seismic=True, use_cache=True)
    time2 = time.time() - start_time
    
    # Verify results are identical
    assert np.allclose(result1, result2, rtol=1e-5), "Cached results differ"
    
    # Verify caching is faster (should be significantly faster)
    print(f"✅ Caching test completed")
    print(f"  First call time: {time1:.3f}s")
    print(f"  Cached call time: {time2:.3f}s")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    return True

def test_batch_preprocessing():
    """Test batch preprocessing functionality."""
    print("🧪 Testing batch preprocessing...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create mock test files
        test_files = []
        for i in range(10):  # Small number for testing
            file_path = tmp_path / f"test_{i}.npy"
            test_data = np.random.randn(500, 5, 2000, 70).astype(np.float32)
            np.save(file_path, test_data)
            test_files.append(file_path)
        
        # Test batch preprocessing
        output_dir = tmp_path / "output"
        processed_files = preprocess_test_data_batch(
            test_files=test_files,
            output_dir=output_dir,
            dt_decimate=4,
            batch_size=3,  # Small batch size for testing
            num_workers=2,
            use_cache=True
        )
        
        # Verify results
        assert len(processed_files) == len(test_files), "Not all files were processed"
        
        # Check that processed files exist and have correct shape
        for file_path in processed_files:
            assert file_path.exists(), f"Processed file {file_path} does not exist"
            data = np.load(file_path)
            assert data.shape[2] == 500, f"Expected time dim 500, got {data.shape[2]}"
            assert data.dtype == np.float16, f"Expected float16, got {data.dtype}"
        
        print("✅ Batch preprocessing test completed")
        print(f"  Input files: {len(test_files)}")
        print(f"  Output files: {len(processed_files)}")
        
        return True

def test_inference_dataset_creation():
    """Test inference-optimized dataset creation."""
    print("🧪 Testing inference dataset creation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create mock test files
        test_files = []
        for i in range(5):  # Small number for testing
            file_path = tmp_path / f"test_{i}.npy"
            test_data = np.random.randn(500, 5, 2000, 70).astype(np.float32)
            np.save(file_path, test_data)
            test_files.append(file_path)
        
        # Test dataset creation
        output_dir = tmp_path / "output"
        zarr_path = create_inference_optimized_dataset(
            test_files=test_files,
            output_dir=output_dir,
            use_cache=True
        )
        
        # Verify zarr dataset was created
        assert zarr_path.exists(), f"Zarr dataset {zarr_path} was not created"
        
        # Load and verify zarr dataset
        import zarr
        test_data = zarr.open(str(zarr_path))
        
        # Check dataset properties
        expected_shape = (len(test_files), 500, 5, 500, 70)  # After downsampling
        assert test_data.shape == expected_shape, f"Expected shape {expected_shape}, got {test_data.shape}"
        
        print("✅ Inference dataset creation test completed")
        print(f"  Zarr dataset: {zarr_path}")
        print(f"  Dataset shape: {test_data.shape}")
        print(f"  Dataset dtype: {test_data.dtype}")
        
        return True

def test_memory_efficiency():
    """Test memory efficiency of new preprocessing functions."""
    print("🧪 Testing memory efficiency...")
    
    # Create larger test data
    test_data = np.random.randn(1000, 5, 2000, 70).astype(np.float32)
    
    # Test memory usage
    import psutil
    process = psutil.Process()
    
    # Memory before
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # Process data
    result = preprocess_for_inference(test_data, dt_decimate=4, use_cache=True)
    
    # Memory after
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    
    print("✅ Memory efficiency test completed")
    print(f"  Input data size: {test_data.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Output data size: {result.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Memory used: {mem_used:.1f} MB")
    print(f"  Compression ratio: {test_data.nbytes / result.nbytes:.1f}x")
    
    return True

def test_performance_comparison():
    """Test performance comparison between original and optimized preprocessing."""
    print("🧪 Testing performance comparison...")
    
    # Create test data
    test_data = np.random.randn(100, 5, 2000, 70).astype(np.float32)
    
    # Test original preprocessing
    start_time = time.time()
    result_original = preprocess_one(test_data, dt_decimate=4, is_seismic=True)
    time_original = time.time() - start_time
    
    # Test optimized preprocessing
    start_time = time.time()
    result_optimized = preprocess_for_inference(test_data, dt_decimate=4, use_cache=True)
    time_optimized = time.time() - start_time
    
    # Verify results are identical
    assert np.allclose(result_original, result_optimized, rtol=1e-5), "Results differ"
    
    print("✅ Performance comparison test completed")
    print(f"  Original time: {time_original:.3f}s")
    print(f"  Optimized time: {time_optimized:.3f}s")
    print(f"  Performance ratio: {time_original/time_optimized:.1f}x")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Running inference preprocessing tests...")
    print("="*60)
    
    tests = [
        test_preprocessing_consistency,
        test_caching_functionality,
        test_batch_preprocessing,
        test_inference_dataset_creation,
        test_memory_efficiency,
        test_performance_comparison
    ]
    
    results = {}
    for test in tests:
        try:
            results[test.__name__] = test()
            print()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            results[test.__name__] = False
            print()
    
    # Summary
    print("="*60)
    print("📊 Test Results Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Inference preprocessing is ready.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 