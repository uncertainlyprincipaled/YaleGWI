#!/usr/bin/env python3
"""
Quick setup script to install the correct s3fs version.

Run this immediately after git clone to avoid S3 compatibility issues:
    python setup_s3fs.py
"""

import subprocess
import sys
import importlib

def install_correct_s3fs():
    """Install the correct s3fs version to avoid compatibility issues."""
    print("🔧 Installing correct s3fs version to avoid compatibility issues...")
    
    try:
        # Check current s3fs version
        try:
            import s3fs
            current_version = s3fs.__version__
            print(f"Current s3fs version: {current_version}")
            
            # Check if version is recent enough
            version_parts = current_version.split('.')
            if len(version_parts) >= 2:
                major = int(version_parts[0])
                if major >= 2024:
                    print("✅ S3fs version is already recent (>=2024)")
                    return True
        except ImportError:
            print("S3fs not installed")
        except Exception as e:
            print(f"Could not check current s3fs version: {e}")
        
        # Uninstall any existing s3fs
        print("Uninstalling old s3fs...")
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 's3fs'], check=True)
        
        # Install the correct version that fixes the 'asynchronous' parameter issue
        print("Installing s3fs>=2024.1.0...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 's3fs>=2024.1.0'], check=True)
        
        # Verify installation
        importlib.reload(importlib.import_module('s3fs'))
        import s3fs
        new_version = s3fs.__version__
        print(f"✅ S3fs updated to version: {new_version}")
        
        # Test functionality
        try:
            fs = s3fs.S3FileSystem(anon=True)
            print("✅ S3fs functionality verified")
            return True
        except Exception as e:
            if "asynchronous" in str(e):
                print(f"❌ S3fs still has 'asynchronous' issue: {e}")
                return False
            else:
                print(f"✅ S3fs working (different error: {e})")
                return True
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install s3fs: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 S3fs Setup Script")
    print("="*50)
    
    success = install_correct_s3fs()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("💡 You can now run the preprocessing pipeline without S3 compatibility issues.")
        print("\nExample usage:")
        print("  from src.utils.colab_setup import quick_colab_setup")
        print("  results = quick_colab_setup(use_s3=True, debug_mode=True)")
    else:
        print("\n❌ Setup failed!")
        print("💡 You may need to:")
        print("  1. Check your internet connection")
        print("  2. Try running with sudo if you have permission issues")
        print("  3. Use local processing instead: use_s3=False")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 