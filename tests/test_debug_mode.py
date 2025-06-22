#!/usr/bin/env python3
"""
Test script for debug mode functionality.

This script demonstrates how to use the new debug mode to process only one family
for quick S3 I/O testing and debugging.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('/content/YaleGWI/src')

def test_debug_mode_local():
    """Test debug mode with local processing."""
    print("🧪 Testing debug mode with local processing...")
    
    try:
        from src.utils.colab_setup import quick_colab_setup
        
        # Test with local processing (no S3)
        results = quick_colab_setup(
            use_s3=False,
            mount_drive=False,
            run_tests=False,
            debug_mode=True,
            debug_family='FlatVel_A'
        )
        
        if results.get('preprocessing', {}).get('success', False):
            print("✅ Debug mode local test passed")
            return True
        else:
            print("❌ Debug mode local test failed")
            return False
            
    except Exception as e:
        print(f"❌ Debug mode local test failed with error: {e}")
        return False

def test_debug_mode_s3():
    """Test debug mode with S3 processing."""
    print("🧪 Testing debug mode with S3 processing...")
    
    try:
        from src.utils.colab_setup import quick_colab_setup
        
        # Test with S3 processing
        results = quick_colab_setup(
            use_s3=True,
            mount_drive=False,
            run_tests=False,
            debug_mode=True,
            debug_family='FlatVel_A'
        )
        
        if results.get('preprocessing', {}).get('success', False):
            print("✅ Debug mode S3 test passed")
            return True
        else:
            print("❌ Debug mode S3 test failed")
            return False
            
    except Exception as e:
        print(f"❌ Debug mode S3 test failed with error: {e}")
        return False

def test_different_families():
    """Test debug mode with different families."""
    print("🧪 Testing debug mode with different families...")
    
    families_to_test = ['FlatVel_A', 'CurveVel_A', 'Style_A']
    
    try:
        from src.utils.colab_setup import quick_colab_setup
        
        for family in families_to_test:
            print(f"  Testing family: {family}")
            
            results = quick_colab_setup(
                use_s3=False,  # Use local for faster testing
                mount_drive=False,
                run_tests=False,
                debug_mode=True,
                debug_family=family
            )
            
            if results.get('preprocessing', {}).get('success', False):
                print(f"  ✅ Family {family} test passed")
            else:
                print(f"  ❌ Family {family} test failed")
                return False
        
        print("✅ All family tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Family tests failed with error: {e}")
        return False

def test_s3_debugging():
    """Test S3 structure debugging functionality."""
    print("🔍 Testing S3 structure debugging...")
    
    try:
        from src.utils.colab_setup import debug_s3_structure
        from src.core.data_manager import DataManager
        
        # Test with S3 enabled
        data_manager = DataManager(use_s3=True)
        
        # Debug S3 structure for FlatVel_A
        debug_s3_structure(data_manager, 'FlatVel_A')
        
        print("✅ S3 debugging test completed")
        return True
        
    except Exception as e:
        print(f"❌ S3 debugging test failed: {e}")
        return False

def test_s3_download():
    """Test S3 download functionality."""
    print("📋 Testing S3 download functionality...")
    
    try:
        from src.utils.colab_setup import download_preprocessed_data_from_s3
        from src.core.data_manager import DataManager
        import tempfile
        from pathlib import Path
        
        # Test with S3 enabled
        data_manager = DataManager(use_s3=True)
        
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_download"
            
            # Try to download preprocessed data
            success = download_preprocessed_data_from_s3("preprocessed", str(tmp_path), data_manager)
            
            if success:
                print(f"✅ S3 download test successful - downloaded to {tmp_path}")
            else:
                print("⚠️ S3 download test - no files found (this is expected if no preprocessed data exists)")
            
            return True
        
    except Exception as e:
        print(f"❌ S3 download test failed: {e}")
        return False

def main():
    """Run all debug mode tests."""
    print("🧪 Debug Mode Testing Suite")
    print("="*50)
    
    tests = [
        ("Local Processing", test_debug_mode_local),
        ("S3 Processing", test_debug_mode_s3),
        ("Different Families", test_different_families),
        ("S3 Debugging", test_s3_debugging),
        ("S3 Download", test_s3_download)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary")
    print("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} | {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 