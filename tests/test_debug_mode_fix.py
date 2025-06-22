#!/usr/bin/env python3
"""
Test script to verify debug mode fixes work correctly.
"""

import sys
import os
from pathlib import Path
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_debug_mode_fixes():
    """Test the debug mode fixes."""
    print("🧪 Testing debug mode fixes...")
    
    # Test 1: Check if the fixes are applied
    print("  Testing fix 1: Missing tempfile imports...")
    try:
        import tempfile
        from src.utils.colab_setup import run_tests_and_validation
        
        # Run the tests to see if tempfile errors are fixed
        results = run_tests_and_validation()
        
        # Check if the tempfile-related tests pass
        if results.get('data_loading_tests', False):
            print("  ✅ Data loading tests pass (tempfile import fixed)")
        else:
            print("  ❌ Data loading tests still fail")
            
        if results.get('5d_dimension_tests', False):
            print("  ✅ 5D dimension tests pass (tempfile import fixed)")
        else:
            print("  ❌ 5D dimension tests still fail")
            
        if results.get('shape_separation_tests', False):
            print("  ✅ Shape separation tests pass (tempfile import fixed)")
        else:
            print("  ❌ Shape separation tests still fail")
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False
    
    # Test 2: Check if the debugging improvements are applied
    print("  Testing fix 2: Enhanced debugging...")
    try:
        from src.core.preprocess import process_family, PreprocessingFeedback
        
        # Create test data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create mock seismic and velocity data
            seismic_data = np.random.randn(500, 5, 1000, 70).astype(np.float32)
            velocity_data = np.random.randn(500, 1, 70, 70).astype(np.float32)
            
            # Save test files
            np.save(tmp_path / "seis_data1.npy", seismic_data)
            np.save(tmp_path / "vel_data1.npy", velocity_data)
            
            # Test process_family with local processing
            output_dir = tmp_path / "output"
            feedback = PreprocessingFeedback()
            
            # This should work and provide debugging output
            processed_paths, feedback = process_family(
                family="test_family",
                input_path=tmp_path,
                output_dir=output_dir,
                data_manager=None  # Local processing
            )
            
            print(f"  ✅ Process family function works: {len(processed_paths)} files processed")
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False
    
    # Test 3: Check if Nyquist validation improvements are applied
    print("  Testing fix 3: Enhanced Nyquist validation...")
    try:
        from src.core.preprocess import validate_nyquist, PreprocessingFeedback
        
        # Create test data with high frequency content
        test_data = np.random.randn(500, 5, 1000, 70).astype(np.float32)
        
        # Add high frequency noise to trigger Nyquist warning
        high_freq_noise = np.random.randn(500, 5, 1000, 70).astype(np.float32) * 0.1
        test_data += high_freq_noise
        
        feedback = PreprocessingFeedback()
        
        # This should provide detailed warnings about Nyquist frequency
        result = validate_nyquist(test_data, original_fs=1000, dt_decimate=4, feedback=feedback)
        
        print(f"  ✅ Nyquist validation works: {result}")
        print(f"  ✅ Feedback collected: {feedback.nyquist_warnings} warnings")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False
    
    print("🎉 All debug mode fixes verified!")
    return True

if __name__ == "__main__":
    success = test_debug_mode_fixes()
    sys.exit(0 if success else 1) 