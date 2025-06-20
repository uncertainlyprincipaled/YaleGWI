"""
Test script to verify preprocessing fixes work with actual S3 data structure.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.core.preprocess import preprocess_one, validate_nyquist, PreprocessingFeedback
from src.core.data_manager import DataManager

def test_preprocessing_fixes():
    """Test the preprocessing fixes with mock data that mimics S3 structure."""
    
    print("🧪 Testing preprocessing fixes...")
    
    # Test 1: 4D seismic data (batch, sources, time, receivers)
    print("\n1. Testing 4D seismic data...")
    seis_4d = np.random.randn(500, 5, 2000, 70).astype(np.float32)
    feedback = PreprocessingFeedback()
    
    try:
        result = preprocess_one(seis_4d, dt_decimate=4, is_seismic=True, feedback=feedback)
        print(f"✅ 4D seismic preprocessing successful: {result.shape}, {result.dtype}")
    except Exception as e:
        print(f"❌ 4D seismic preprocessing failed: {e}")
    
    # Test 2: 3D seismic data (sources, time, receivers)
    print("\n2. Testing 3D seismic data...")
    seis_3d = np.random.randn(5, 2000, 70).astype(np.float32)
    
    try:
        result = preprocess_one(seis_3d, dt_decimate=4, is_seismic=True, feedback=feedback)
        print(f"✅ 3D seismic preprocessing successful: {result.shape}, {result.dtype}")
    except Exception as e:
        print(f"❌ 3D seismic preprocessing failed: {e}")
    
    # Test 3: Velocity model data
    print("\n3. Testing velocity model data...")
    vel_data = np.random.randn(500, 1, 70, 70).astype(np.float32)
    
    try:
        result = preprocess_one(vel_data, is_seismic=False, feedback=feedback)
        print(f"✅ Velocity preprocessing successful: {result.shape}, {result.dtype}")
    except Exception as e:
        print(f"❌ Velocity preprocessing failed: {e}")
    
    # Test 4: Nyquist validation
    print("\n4. Testing Nyquist validation...")
    try:
        valid = validate_nyquist(seis_4d, dt_decimate=4, feedback=feedback)
        print(f"✅ Nyquist validation successful: {valid}")
    except Exception as e:
        print(f"❌ Nyquist validation failed: {e}")
    
    # Test 5: Edge case - different data shapes
    print("\n5. Testing edge cases...")
    
    # Test with time as first dimension
    seis_time_first = np.random.randn(2000, 70, 5).astype(np.float32)
    try:
        result = preprocess_one(seis_time_first, dt_decimate=4, is_seismic=True, feedback=feedback)
        print(f"✅ Time-first preprocessing successful: {result.shape}")
    except Exception as e:
        print(f"❌ Time-first preprocessing failed: {e}")
    
    print(f"\n📊 Feedback summary:")
    print(f"  Arrays processed: {feedback.arrays_processed}")
    print(f"  Nyquist warnings: {feedback.nyquist_warnings}")
    print(f"  Warning percentage: {feedback.warning_percentage:.2f}%")

def test_s3_data_structure():
    """Test with actual S3 data structure if available."""
    print("\n🔍 Testing S3 data structure...")
    
    try:
        data_manager = DataManager(use_s3=True)
        
        # List a few files to understand the structure
        families = ['CurveFault_A', 'FlatVel_A']
        
        for family in families:
            print(f"\n  Family: {family}")
            try:
                seis_files, vel_files, family_type = data_manager.list_family_files(family)
                print(f"    Seismic files: {len(seis_files)}")
                print(f"    Velocity files: {len(vel_files)}")
                
                if seis_files:
                    # Try to load first file to check shape
                    sample_data = data_manager.stream_from_s3(seis_files[0])
                    print(f"    Sample seismic shape: {sample_data.shape}")
                    
            except Exception as e:
                print(f"    ❌ Error accessing {family}: {e}")
                
    except Exception as e:
        print(f"❌ S3 test failed: {e}")

if __name__ == "__main__":
    test_preprocessing_fixes()
    test_s3_data_structure()
    print("\n🎉 Preprocessing fix tests completed!") 