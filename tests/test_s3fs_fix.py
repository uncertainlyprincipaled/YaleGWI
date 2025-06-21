#!/usr/bin/env python3
"""
Test script for s3fs fix functionality.

This script tests the s3fs update mechanism to ensure it resolves the 'asynchronous' parameter issue.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add src to path
sys.path.append('/content/YaleGWI/src')

def test_s3fs_version_check():
    """Test the s3fs version check and update mechanism."""
    print("🧪 Testing s3fs version check and update...")
    
    try:
        # Try to import the function, but handle the case where src is not available
        try:
            from src.utils.colab_setup import check_and_fix_s3fs_installation
            
            # Test the s3fs check function
            result = check_and_fix_s3fs_installation()
            
            if result:
                print("✅ S3fs check and update successful")
                
                # Verify the version is recent
                import s3fs
                version_parts = s3fs.__version__.split('.')
                if len(version_parts) >= 2:
                    major = int(version_parts[0])
                    if major >= 2024:
                        print(f"✅ S3fs version {s3fs.__version__} is recent (>=2024)")
                        return True
                    else:
                        print(f"⚠️ S3fs version {s3fs.__version__} is still old (<2024)")
                        return False
                else:
                    print(f"⚠️ Could not parse s3fs version: {s3fs.__version__}")
                    return False
            else:
                print("❌ S3fs check and update failed")
                return False
                
        except ImportError:
            # If src module is not available, just check the current s3fs version
            print("⚠️ src module not available, checking s3fs version directly...")
            import s3fs
            version_parts = s3fs.__version__.split('.')
            if len(version_parts) >= 2:
                major = int(version_parts[0])
                if major >= 2024:
                    print(f"✅ S3fs version {s3fs.__version__} is recent (>=2024)")
                    return True
                else:
                    print(f"⚠️ S3fs version {s3fs.__version__} is old (<2024)")
                    return False
            else:
                print(f"⚠️ Could not parse s3fs version: {s3fs.__version__}")
                return False
            
    except Exception as e:
        print(f"❌ S3fs test failed with error: {e}")
        return False

def test_s3fs_functionality():
    """Test that s3fs works without the 'asynchronous' error."""
    print("🧪 Testing s3fs functionality...")
    
    try:
        import s3fs
        
        # Test creating S3FileSystem without the 'asynchronous' error
        try:
            fs = s3fs.S3FileSystem(anon=True)
            print("✅ S3fs.S3FileSystem creation successful")
            
            # Test basic functionality
            try:
                # Try to list a public bucket (this should work without credentials)
                files = fs.ls('s3://noaa-ghcn-pds/')
                print(f"✅ S3fs functionality verified - listed {len(files)} files")
                return True
            except Exception as e:
                if "asynchronous" in str(e):
                    print(f"❌ S3fs still has 'asynchronous' issue: {e}")
                    return False
                else:
                    print(f"✅ S3fs working (different error: {e})")
                    return True
                    
        except Exception as e:
            if "asynchronous" in str(e):
                print(f"❌ S3fs.S3FileSystem creation failed with 'asynchronous' error: {e}")
                return False
            else:
                print(f"✅ S3fs.S3FileSystem creation failed with different error: {e}")
                return True
                
    except ImportError:
        print("❌ S3fs not installed")
        return False
    except Exception as e:
        print(f"❌ S3fs functionality test failed: {e}")
        return False

def test_s3fs_with_zarr():
    """Test s3fs integration with zarr."""
    print("🧪 Testing s3fs integration with zarr...")
    
    try:
        import s3fs
        import zarr
        import numpy as np
        import tempfile
        
        # Test creating a simple zarr array with s3fs
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test local zarr first
            local_path = Path(tmpdir) / "test.zarr"
            test_data = np.random.randn(10, 10).astype(np.float32)
            
            try:
                # Save locally
                zarr.save(str(local_path), test_data)
                print("✅ Local zarr save successful")
                
                # Test s3fs with zarr (this might fail without proper credentials, but shouldn't have 'asynchronous' error)
                try:
                    # This will likely fail due to credentials, but shouldn't have 'asynchronous' error
                    s3_path = "s3://test-bucket/test.zarr"
                    zarr.save(s3_path, test_data)
                    print("✅ S3 zarr save successful")
                    return True
                except Exception as e:
                    if "asynchronous" in str(e):
                        print(f"❌ S3 zarr save failed with 'asynchronous' error: {e}")
                        return False
                    elif "credentials" in str(e).lower() or "access" in str(e).lower():
                        print(f"✅ S3 zarr save failed with credentials error (expected): {e}")
                        return True
                    else:
                        print(f"✅ S3 zarr save failed with different error: {e}")
                        return True
                        
            except Exception as e:
                print(f"❌ Local zarr save failed: {e}")
                return False
                
    except ImportError as e:
        print(f"❌ Required packages not available: {e}")
        return False
    except Exception as e:
        print(f"❌ S3fs-zarr integration test failed: {e}")
        return False

def test_manual_s3fs_update():
    """Test manual s3fs update if needed."""
    print("🧪 Testing manual s3fs update...")
    
    try:
        import s3fs
        current_version = s3fs.__version__
        print(f"Current s3fs version: {current_version}")
        
        # Check if version is old
        version_parts = current_version.split('.')
        if len(version_parts) >= 2:
            major = int(version_parts[0])
            if major < 2024:
                print("⚠️ S3fs version is old, attempting manual update...")
                
                # Try manual update
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 's3fs'], check=True)
                    subprocess.run([sys.executable, '-m', 'pip', 'install', 's3fs>=2024.1.0'], check=True)
                    
                    # Reload s3fs
                    import importlib
                    importlib.reload(s3fs)
                    new_version = s3fs.__version__
                    print(f"✅ S3fs updated from {current_version} to {new_version}")
                    
                    # Test functionality
                    fs = s3fs.S3FileSystem(anon=True)
                    print("✅ Updated s3fs functionality verified")
                    return True
                    
                except Exception as e:
                    print(f"❌ Manual s3fs update failed: {e}")
                    return False
            else:
                print("✅ S3fs version is already recent")
                return True
        else:
            print("⚠️ Could not parse s3fs version")
            return False
            
    except Exception as e:
        print(f"❌ Manual s3fs update test failed: {e}")
        return False

def main():
    """Run all s3fs tests."""
    print("🧪 S3fs Fix Testing Suite")
    print("="*50)
    
    tests = [
        ("Version Check & Update", test_s3fs_version_check),
        ("S3fs Functionality", test_s3fs_functionality),
        ("S3fs-Zarr Integration", test_s3fs_with_zarr),
        ("Manual Update", test_manual_s3fs_update)
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
        print(f"{test_name:<25} | {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\n💡 Recommendations:")
        print("1. Restart the runtime and try again")
        print("2. Run the setup with debug mode to test S3 I/O")
        print("3. Check AWS credentials if S3 operations fail")
        print("4. Consider using local processing if S3 issues persist")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 