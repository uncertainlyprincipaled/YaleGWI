"""
Common imports and dependency management for the project.
This module provides both common imports used across the codebase and functions to manage dependencies.
"""
# Standard library imports
from __future__ import annotations
import os
import sys
import json
import time
import signal
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Literal, NamedTuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core dependencies with versions
CORE_DEPS = [
    'numpy>=1.21.0',
    'torch>=1.9.0',
    'scipy>=1.7.0',
    'pandas>=1.3.0',
    'scikit-learn>=0.24.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'tqdm>=4.62.0',
    'zarr>=2.10.0',
    'dask>=2021.9.0',
    'boto3>=1.18.0',
    'botocore>=1.21.0',
    'timm>=0.6.0'
]

# Optional dependencies for specific environments
ENV_DEPS = {
    'kaggle': [
        'kaggle>=1.5.0',
        'kaggle-secrets>=0.1.0',
    ],
    'colab': [
        'google-colab>=0.0.1a1',
    ],
    'sagemaker': [
        'sagemaker>=2.0.0',
    ]
}

def install_dependencies(env: str = None):
    """
    Install all required dependencies for the project.
    
    Args:
        env: Optional environment name ('kaggle', 'colab', 'sagemaker')
             to install environment-specific dependencies.
    """
    try:
        # Install core dependencies
        logger.info("Installing core dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + CORE_DEPS)
        
        # Install environment-specific dependencies
        if env and env in ENV_DEPS:
            logger.info(f"Installing {env} specific dependencies...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + ENV_DEPS[env])
            
        logger.info("All dependencies installed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def setup_environment(env: str = None):
    """
    Set up the complete environment including dependencies and imports.
    
    Args:
        env: Optional environment name ('kaggle', 'colab', 'sagemaker')
             to install environment-specific dependencies.
    """
    # Install dependencies
    install_dependencies(env)
    
    # Import core dependencies
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import scipy
    import pandas as pd
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import zarr
    import dask
    import boto3
    from botocore.exceptions import ClientError
    import timm
    # kagglehub is imported optionally at the top level
    
    # Import project modules
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    from src.core.config import CFG
    from src.core.data_manager import DataManager
    
    # Set up environment-specific configurations
    if env == 'kaggle':
        os.environ['KAGGLE_CONFIG_DIR'] = '/kaggle/input'
    elif env == 'colab':
        from google.colab import drive
        drive.mount('/content/drive')
    elif env == 'sagemaker':
        os.environ['SAGEMAKER_MODE'] = '1'
    
    logger.info(f"Environment setup complete for {env if env else 'default'} mode")
    
    return {
        'np': np,
        'torch': torch,
        'nn': nn,
        'F': F,
        'scipy': scipy,
        'pd': pd,
        'metrics': metrics,
        'plt': plt,
        'sns': sns,
        'tqdm': tqdm,
        'zarr': zarr,
        'dask': dask,
        'boto3': boto3,
        'ClientError': ClientError,
        'timm': timm,
        'kagglehub': kagglehub,
        'CFG': CFG,
        'DataManager': DataManager
    }

# Common imports used across the codebase
# These are imported here for convenience and to ensure they're available
# when importing from this module
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import timm
try:
    import kagglehub
except ImportError:
    kagglehub = None

if __name__ == "__main__":
    # Example usage
    deps = setup_environment('sagemaker')  # or 'colab' or 'sagemaker'
    print("Environment setup complete!")