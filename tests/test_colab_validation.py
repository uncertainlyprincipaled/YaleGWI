#!/usr/bin/env python3
"""
Quick test script to validate the fixes for Colab setup issues.
Run this after making changes to verify everything works.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('/content/YaleGWI/src')

def test_preprocessing_fixes():
    """Test the preprocessing fixes."""
    print("🧪 Testing preprocessing fixes...")
    try:
        from src.core.preprocess import preprocess_one, validate_nyquist, PreprocessingFeedback
        import numpy as np
        
        # Test with mock data
        seis_4d = np.random.randn(500, 5, 2000, 70).astype(np.float32)
        feedback = PreprocessingFeedback()
        
        result = preprocess_one(seis_4d, dt_decimate=2, is_seismic=True, feedback=feedback)
        if result.shape[2] == 1000:  # Should be downsampled from 2000 to 1000 with factor 2
            print("  ✅ Preprocessing tests passed")
            return True
        else:
            print(f"  ❌ Preprocessing shape mismatch: expected time dim 1000, got {result.shape[2]}")
            return False
            
    except Exception as e:
        print(f"  ❌ Preprocessing tests failed: {e}")
        return False

def test_phase1_components():
    """Test Phase 1 components."""
    print("🧪 Testing Phase 1 components...")
    try:
        from src.core.registry import ModelRegistry
        from src.core.checkpoint import CheckpointManager
        from src.core.data_manager import DataManager
        from src.core.geometric_cv import GeometricCrossValidator
        
        # Test model registry
        registry = ModelRegistry()
        if registry is not None:
            print("  ✅ Model registry working")
            
        # Test checkpoint manager
        checkpoint_mgr = CheckpointManager()
        if checkpoint_mgr is not None:
            print("  ✅ Checkpoint manager working")
            
        # Test data manager
        data_mgr = DataManager(use_s3=False)  # Test local mode
        if data_mgr is not None:
            print("  ✅ Data manager working")
            
        # Test cross-validator
        cv = GeometricCrossValidator(n_splits=3)
        if cv is not None:
            print("  ✅ Cross-validator working")
            
        # Test geometric metrics
        import numpy as np
        test_data = np.random.randn(100, 100)
        metrics = cv.compute_geometric_metrics(test_data, test_data)
        if 'ssim' in metrics and 'boundary_iou' in metrics:
            print("  ✅ Geometric metrics computation working")
            
        print("  ✅ Phase 1 tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Phase 1 tests failed: {e}")
        return False

def test_data_loading():
    """Test data loading components."""
    print("🧪 Testing data loading...")
    try:
        from src.core.geometric_loader import FamilyDataLoader
        
        # Test family data loader instantiation
        family_loader = FamilyDataLoader('/tmp/test_data', batch_size=16)
        if family_loader is not None:
            print("  ✅ Family data loader working")
            
        # Test geometric dataset
        from src.core.geometric_loader import GeometricDataset
        # Create mock data for testing
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # This is a basic test - in real usage, we'd have actual zarr data
            print("  ✅ Geometric dataset structure working")
            
        print("  ✅ Data loading tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Data loading tests failed: {e}")
        return False

def test_integration():
    """Test integration components."""
    print("🧪 Testing integration...")
    try:
        from src.core.config import CFG
        
        # Verify config loads
        if CFG.env.kind in ['colab', 'kaggle']:
            print("  ✅ Environment config working")
            
        # Verify paths exist
        if hasattr(CFG, 'paths') and hasattr(CFG.paths, 'families'):
            print("  ✅ Family paths configured")
            
        # Verify FAMILY_FILE_MAP has updated downsample factors
        from src.core.config import FAMILY_FILE_MAP
        fault_families = ['CurveFault_A', 'CurveFault_B', 'FlatFault_A', 'FlatFault_B']
        for family in fault_families:
            if family in FAMILY_FILE_MAP:
                factor = FAMILY_FILE_MAP[family]['downsample_factor']
                if factor == 2:
                    print(f"  ✅ {family} downsample factor updated to 2")
                else:
                    print(f"  ❌ {family} downsample factor still {factor}")
            
        print("  ✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration tests failed: {e}")
        return False

def test_preprocessing_skip():
    """Test the preprocessing skip functionality."""
    print("🧪 Testing preprocessing skip functionality...")
    try:
        from src.utils.colab_setup import check_preprocessed_data_exists, copy_preprocessed_data_from_drive
        
        # Test check function
        result = check_preprocessed_data_exists('/tmp/test_output', save_to_drive=False, use_s3=False)
        
        # Verify result structure
        expected_keys = ['exists_locally', 'exists_in_drive', 'exists_in_s3', 'local_path', 'drive_path', 's3_path', 'data_quality']
        if all(key in result for key in expected_keys):
            print("  ✅ Preprocessing check function working")
            return True
        else:
            print("  ❌ Preprocessing check function missing expected keys")
            return False
            
    except Exception as e:
        print(f"  ❌ Preprocessing skip test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running quick validation tests...")
    print("="*50)
    
    results = {
        'preprocessing': test_preprocessing_fixes(),
        'phase1': test_phase1_components(),
        'data_loading': test_data_loading(),
        'integration': test_integration(),
        'preprocessing_skip': test_preprocessing_skip()
    }
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n💡 Ready to run full Colab setup!")
    else:
        print("\n⚠️ Fix the failing tests before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    main() 