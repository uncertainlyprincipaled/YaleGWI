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
    """Create a Kaggle notebook with the required files."""
    # Create notebook directory
    notebook_dir = Path('kaggle_notebook')
    notebook_dir.mkdir(exist_ok=True)
    
    # Create __init__.py files
    for path in ['src', 'src/core', 'src/utils']:
        init_dir = notebook_dir / path
        init_dir.mkdir(parents=True, exist_ok=True)
        init_file = init_dir / '__init__.py'
        init_file.touch(exist_ok=True)
    
    # Copy files to notebook directory
    for file_path in FILES_TO_INCLUDE:
        src_path = Path(file_path)
        if src_path.exists():
            dest_path = notebook_dir / src_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Warning: {file_path} not found")
    
    # Create main notebook file
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Seismic Waveform Inversion - Preprocessing Pipeline\n",
    "\n",
    "This notebook implements the preprocessing pipeline for seismic waveform inversion, including:\n",
    "- Geometric-aware preprocessing with Nyquist validation\n",
    "- Family-specific data loading\n",
    "- Cross-validation framework\n",
    "- Model registry and checkpoint management"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Install dependencies\n",
    "!pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import required modules\n",
    "from src.core.preprocess import main as preprocess_main\n",
    "\n",
    "# Run preprocessing pipeline\n",
    "preprocess_main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    with open(notebook_dir / 'seismic_preprocessing.ipynb', 'w') as f:
        f.write(notebook_content)
    
    print("Notebook created successfully!")

if __name__ == "__main__":
    create_notebook() 