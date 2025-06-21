#!/usr/bin/env python3
"""
Test script for Colab setup functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_basic_imports():
    """Test that basic packages can be imported"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ torch imported")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    return True

def test_config_import():
    """Test that the config module can be imported"""
    print("\nTesting config import...")
    
    try:
        from src.core.config import CFG
        print(f"✅ Config imported - Environment: {CFG.env.kind}")
        return True
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_colab_setup_import():
    """Test that the colab setup module can be imported"""
    print("\nTesting colab setup import...")
    
    try:
        from src.utils.colab_setup import setup_colab_environment
        print("✅ Colab setup module imported")
        return True
    except ImportError as e:
        print(f"❌ Colab setup import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Colab Setup Components")
    print("="*50)
    
    tests = [
        test_basic_imports,
        test_config_import,
        test_colab_setup_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Setup should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 