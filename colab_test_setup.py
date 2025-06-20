"""
Colab Test Setup Script

This script can be run in Google Colab to test the preprocessing fixes
and Phase 1 components without running the full setup.

Usage in Colab:
!python colab_test_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('/content/YaleGWI/src')

def test_preprocessing_fixes():
    """Test the preprocessing fixes with mock data."""
    print("🧪 Testing preprocessing fixes...")
    
    try:
        from src.core.preprocess import preprocess_one, validate_nyquist, PreprocessingFeedback
        import numpy as np
        
        # Test 1: 4D seismic data
        print("  Testing 4D seismic data...")
        seis_4d = np.random.randn(500, 5, 2000, 70).astype(np.float32)
        feedback = PreprocessingFeedback()
        
        result = preprocess_one(seis_4d, dt_decimate=4, is_seismic=True, feedback=feedback)
        if result.shape[2] == 500:  # Should be downsampled from 2000 to 500
            print("  ✅ 4D seismic preprocessing successful")
        else:
            print(f"  ❌ 4D seismic preprocessing failed: expected time dim 500, got {result.shape[2]}")
            return False
            
        # Test 2: 3D seismic data
        print("  Testing 3D seismic data...")
        seis_3d = np.random.randn(5, 2000, 70).astype(np.float32)
        result = preprocess_one(seis_3d, dt_decimate=4, is_seismic=True, feedback=feedback)
        if result.shape[1] == 500:  # Should be downsampled from 2000 to 500
            print("  ✅ 3D seismic preprocessing successful")
        else:
            print(f"  ❌ 3D seismic preprocessing failed: expected time dim 500, got {result.shape[1]}")
            return False
            
        # Test 3: Velocity data
        print("  Testing velocity data...")
        vel_data = np.random.randn(500, 1, 70, 70).astype(np.float32)
        result = preprocess_one(vel_data, is_seismic=False, feedback=feedback)
        print("  ✅ Velocity preprocessing successful")
        
        # Test 4: Nyquist validation
        print("  Testing Nyquist validation...")
        valid = validate_nyquist(seis_4d, dt_decimate=4, feedback=feedback)
        print(f"  ✅ Nyquist validation successful: {valid}")
        
        print(f"  📊 Feedback: {feedback.arrays_processed} arrays processed, {feedback.nyquist_warnings} warnings")
        return True
        
    except Exception as e:
        print(f"  ❌ Preprocessing test failed: {e}")
        return False

def test_phase1_components():
    """Test Phase 1 components."""
    print("\n🧪 Testing Phase 1 components...")
    
    try:
        # Test model registry
        print("  Testing model registry...")
        from src.core.registry import ModelRegistry
        registry = ModelRegistry()
        print("  ✅ Model registry working")
        
        # Test checkpoint manager
        print("  Testing checkpoint manager...")
        from src.core.checkpoint import CheckpointManager
        checkpoint_mgr = CheckpointManager()
        print("  ✅ Checkpoint manager working")
        
        # Test data manager
        print("  Testing data manager...")
        from src.core.data_manager import DataManager
        data_mgr = DataManager(use_s3=False)  # Test local mode
        print("  ✅ Data manager working")
        
        # Test cross-validation
        print("  Testing cross-validation...")
        from src.core.geometric_cv import GeometricCrossValidator
        cv = GeometricCrossValidator()
        print("  ✅ Cross-validation working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Phase 1 test failed: {e}")
        return False

def test_integration():
    """Test integration components."""
    print("\n🧪 Testing integration...")
    
    try:
        # Test config
        print("  Testing configuration...")
        from src.core.config import CFG
        print(f"  ✅ Environment: {CFG.env.kind}")
        print(f"  ✅ Device: {CFG.env.device}")
        
        # Test family configuration
        if hasattr(CFG, 'paths') and hasattr(CFG.paths, 'families'):
            print(f"  ✅ Family paths configured: {len(CFG.paths.families)} families")
        
        # Test FAMILY_FILE_MAP
        from src.core.config import FAMILY_FILE_MAP
        print(f"  ✅ FAMILY_FILE_MAP configured: {len(FAMILY_FILE_MAP)} families")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 YaleGWI Test Setup")
    print("="*50)
    
    # Test preprocessing fixes
    preprocessing_ok = test_preprocessing_fixes()
    
    # Test Phase 1 components
    phase1_ok = test_phase1_components()
    
    # Test integration
    integration_ok = test_integration()
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary")
    print("="*50)
    print(f"  Preprocessing Fixes: {'✅ PASSED' if preprocessing_ok else '❌ FAILED'}")
    print(f"  Phase 1 Components: {'✅ PASSED' if phase1_ok else '❌ FAILED'}")
    print(f"  Integration: {'✅ PASSED' if integration_ok else '❌ FAILED'}")
    
    if preprocessing_ok and phase1_ok and integration_ok:
        print("\n🎉 All tests passed! Phase 1 is ready for use.")
        print("\n💡 Next steps:")
        print("  1. Run complete setup: complete_colab_setup(use_s3=True, run_tests=True)")
        print("  2. Start training with Phase 1 components")
        print("  3. Integrate Phase 2 & 3 models")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Recommendations:")
        if not preprocessing_ok:
            print("  - Check preprocessing pipeline fixes")
        if not phase1_ok:
            print("  - Verify Phase 1 component implementations")
        if not integration_ok:
            print("  - Check configuration and imports")

if __name__ == "__main__":
    main() 