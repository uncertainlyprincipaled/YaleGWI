import os
from pathlib import Path
import re
from typing import List
import logging
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.config import CFG

def extract_code_blocks(content: str, source_path: str = None) -> List[tuple[str, str]]:
    """Extract code blocks from Python file."""
    blocks = []
    current_block = []
    
    for line in content.split('\n'):
        if line.startswith('# %%'):
            if current_block:
                blocks.append(('\n'.join(current_block), source_path))
                current_block = []
        else:
            current_block.append(line)
    
    if current_block:
        blocks.append(('\n'.join(current_block), source_path))
    
    return blocks

def create_notebook_block(content: str, source_path: str = None) -> str:
    """Create a notebook code block."""
    if source_path:
        return f'# %%\n# Source: {source_path}\n{content}\n\n'
    return f'# %%\n{content}\n\n'

def update_kaggle_notebook():
    """Update kaggle_notebook.py with content from individual files."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # List of files to include in the notebook, in order of execution
    FILES_TO_INCLUDE = [
        'src/core/config.py',
        'src/core/preprocess.py',  # Preprocessing with geometric validation
        'src/core/registry.py',    # Model registry with geometric metadata
        'src/core/checkpoint.py',  # Checkpoint management
        'src/core/geometric_loader.py',  # Family-specific data loading
        'src/core/geometric_cv.py',  # Cross-validation framework
        'src/core/data_manager.py',
        'src/core/model.py',
        'src/core/train.py',
        'src/core/inference.py'
    ]
    
    # Read all files
    all_blocks = []
    
    # Add header comment
    all_blocks.append(("# SpecProj-UNet for Seismic Waveform Inversion\n"
                     "# This notebook implements a physics-guided neural network for seismic waveform inversion\n"
                     "# using spectral projectors and UNet architecture.", "header"))
    
    # Add imports block first
    all_blocks.append(("""
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

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import timm
import kagglehub  # Optional import

# Local imports
from src.core.config import CFG
    """, "imports"))
    
    # Process each file
    for file in FILES_TO_INCLUDE:
        file_path = project_root / file
        if not file_path.exists():
            print(f"Warning: {file} not found, skipping...")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            # Use just the filename as the source path
            blocks = extract_code_blocks(content, file)
            all_blocks.extend(blocks)
    
    # Create notebook content
    notebook_content = []
    for block, source_path in all_blocks:
        notebook_content.append(create_notebook_block(block, source_path))
    
    # Write to kaggle_notebook.py
    notebook_path = project_root / 'kaggle_notebook.py'
    with open(notebook_path, 'w') as f:
        f.write('\n'.join(notebook_content))
    
    print(f"Successfully updated {notebook_path}")

if __name__ == '__main__':
    update_kaggle_notebook() 