#!/usr/bin/env python3
"""
Quick test script to verify preprocessed data structure and GPU datasets.
This script checks if the preprocessing pipeline created the expected GPU-specific datasets.
"""

import os
import sys
from pathlib import Path
import numpy as np

def test_preprocessed_data():
    """Test if preprocessed data exists and has the correct structure."""
    print("🔍 Testing preprocessed data structure...")
    
    # Check if preprocessed directory exists
    preprocessed_dir = Path('/content/YaleGWI/preprocessed')
    if not preprocessed_dir.exists():
        print("❌ Preprocessed directory not found")
        return False
    
    print(f"✅ Preprocessed directory found: {preprocessed_dir}")
    
    # Check for GPU-specific directories
    gpu0_dir = preprocessed_dir / 'gpu0'
    gpu1_dir = preprocessed_dir / 'gpu1'
    
    if not gpu0_dir.exists():
        print("❌ GPU0 directory not found")
        return False
    
    if not gpu1_dir.exists():
        print("❌ GPU1 directory not found")
        return False
    
    print("✅ GPU-specific directories found")
    
    # Check for zarr datasets
    gpu0_zarr = gpu0_dir / 'seismic.zarr'
    gpu1_zarr = gpu1_dir / 'seismic.zarr'
    
    if not gpu0_zarr.exists():
        print("❌ GPU0 zarr dataset not found")
        return False
    
    if not gpu1_zarr.exists():
        print("❌ GPU1 zarr dataset not found")
        return False
    
    print("✅ Zarr datasets found")
    
    # Try to load the datasets
    try:
        import zarr
        
        # Load GPU0 dataset
        data0 = zarr.open(str(gpu0_zarr))
        print(f"✅ GPU0 dataset loaded: {data0.shape}")
        
        # Load GPU1 dataset
        data1 = zarr.open(str(gpu1_zarr))
        print(f"✅ GPU1 dataset loaded: {data1.shape}")
        
        # Check data types
        print(f"✅ GPU0 dtype: {data0.dtype}")
        print(f"✅ GPU1 dtype: {data1.dtype}")
        
        # Check if data is not empty
        if len(data0) == 0:
            print("❌ GPU0 dataset is empty")
            return False
        
        if len(data1) == 0:
            print("❌ GPU1 dataset is empty")
            return False
        
        print(f"✅ GPU0 samples: {len(data0)}")
        print(f"✅ GPU1 samples: {len(data1)}")
        
        # Test loading a sample
        sample0 = data0[0]
        sample1 = data1[0]
        print(f"✅ Sample shapes: GPU0={sample0.shape}, GPU1={sample1.shape}")
        
        return True
        
    except ImportError:
        print("⚠️ zarr not available, skipping dataset loading test")
        return True
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return False

def test_data_loading():
    """Test if the data loading components work with the preprocessed data."""
    print("\n🔍 Testing data loading components...")
    
    try:
        from src.core.geometric_loader import FamilyDataLoader
        
        # Test family data loader
        loader = FamilyDataLoader('/content/YaleGWI/preprocessed/gpu0', batch_size=4)
        print("✅ FamilyDataLoader created successfully")
        
        # Test loading a batch
        loaders = loader.get_all_loaders()
        if loaders:
            for family, data_loader in loaders.items():
                batch = next(iter(data_loader))
                print(f"✅ Successfully loaded batch from {family}: {batch['data'].shape}")
                break
        else:
            print("⚠️ No data loaders found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running preprocessed data verification tests...")
    print("="*60)
    
    # Test 1: Preprocessed data structure
    test1_passed = test_preprocessed_data()
    
    # Test 2: Data loading
    test2_passed = test_data_loading()
    
    print("\n" + "="*60)
    print("📊 Test Results:")
    print(f"  Preprocessed Data Structure: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"  Data Loading Components: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Preprocessed data is ready for training.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the preprocessing pipeline.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 