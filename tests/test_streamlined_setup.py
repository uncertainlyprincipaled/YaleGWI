#!/usr/bin/env python3
"""
Test script for streamlined Colab setup with missing dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_streamlined_setup():
    """Test the streamlined setup function."""
    print("🧪 Testing streamlined Colab setup...")
    
    try:
        from src.utils.colab_setup import quick_colab_setup_streamlined
        
        # Test with local mode (no S3)
        print("\n📋 Testing local mode...")
        results = quick_colab_setup_streamlined(
            use_s3=False,
            mount_drive=False,
            run_tests=False,
            debug_mode=True,
            debug_family='FlatVel_A'
        )
        
        print("✅ Streamlined setup test completed successfully!")
        print(f"📊 Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlined setup test failed: {e}")
        return False

def test_config_import():
    """Test that config can be imported without boto3."""
    print("\n🧪 Testing config import...")
    
    try:
        from src.core.config import CFG
        print("✅ Config imported successfully")
        print(f"📊 Environment: {CFG.env.kind}")
        print(f"📊 Device: {CFG.env.device}")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_data_manager():
    """Test DataManager without S3."""
    print("\n🧪 Testing DataManager...")
    
    try:
        from src.core.data_manager import DataManager
        
        # Test local mode
        dm = DataManager(use_s3=False)
        print("✅ DataManager created successfully (local mode)")
        
        # Test S3 mode (should fail gracefully)
        try:
            dm_s3 = DataManager(use_s3=True)
            print("⚠️ S3 DataManager created (boto3 might be available)")
        except Exception as e:
            print(f"✅ S3 DataManager failed gracefully as expected: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataManager test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting streamlined setup tests...")
    
    tests = [
        test_config_import,
        test_data_manager,
        test_streamlined_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 