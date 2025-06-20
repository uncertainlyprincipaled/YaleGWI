#!/usr/bin/env python3
"""
Test runner for YaleGWI project.
Runs all tests in the tests directory.
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test_file(test_file: Path) -> bool:
    """
    Run a single test file.
    
    Args:
        test_file: Path to test file
        
    Returns:
        True if test passed, False otherwise
    """
    try:
        logger.info(f"Running {test_file.name}...")
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {test_file.name} passed")
            return True
        else:
            logger.error(f"❌ {test_file.name} failed")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running {test_file.name}: {e}")
        return False

def main():
    """Run all tests in the tests directory."""
    tests_dir = Path(__file__).parent
    test_files = [
        "test_phase1_integration.py",
        "test_phase2_3.py",
        "test_hybrid_forward.py",
        "test_spectral.py",
        "test_iunet.py"
    ]
    
    logger.info("Starting YaleGWI test suite...")
    
    results = {}
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            results[test_file] = run_test_file(test_path)
        else:
            logger.warning(f"Test file {test_file} not found")
            results[test_file] = False
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info(f"\nTest Results Summary:")
    logger.info(f"Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 