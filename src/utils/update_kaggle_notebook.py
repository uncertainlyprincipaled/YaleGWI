import os
import shutil
from pathlib import Path
import re
from typing import List
import logging
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.config import CFG

# Files to include in the Kaggle notebook
FILES_TO_INCLUDE = [
    'src/core/config.py',
    'src/core/preprocess.py',
    'src/core/registry.py',
    'src/core/checkpoint.py',
    'src/core/geometric_loader.py',
    'src/core/geometric_cv.py',
    'src/core/data_manager.py',
    'src/core/model.py',
    'src/utils/update_kaggle_notebook.py',
    'requirements.txt'
]

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

def create_notebook():
    """Create/update the Kaggle notebook Python file."""
    # Create the main notebook file
    notebook_content = []
    
    # Add header
    notebook_content.append('''"""
# Seismic Waveform Inversion - Preprocessing Pipeline

This notebook implements the preprocessing pipeline for seismic waveform inversion, including:
- Geometric-aware preprocessing with Nyquist validation
- Family-specific data loading
- Cross-validation framework
- Model registry and checkpoint management
"""

# Install dependencies
# !pip install -r requirements.txt

''')
    
    # Process each file
    for file_path in FILES_TO_INCLUDE:
        src_path = Path(file_path)
        if src_path.exists():
            with open(src_path, 'r') as f:
                content = f.read()
                blocks = extract_code_blocks(content, str(src_path))
                for block, source in blocks:
                    notebook_content.append(create_notebook_block(block, source))
        else:
            print(f"Warning: {file_path} not found")
    
    # Write to the root kaggle_notebook.py
    with open(project_root / 'kaggle_notebook.py', 'w') as f:
        f.write('\n'.join(notebook_content))
    
    print("Notebook updated successfully!")

if __name__ == "__main__":
    create_notebook() 