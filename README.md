# SpecProj-UNet for Seismic Waveform Inversion

This project implements a physics-guided neural network for seismic waveform inversion using spectral projectors and UNet architecture.

## Project Structure

```
.
├── src/
│   ├── core/                 # Core implementation files
│   │   ├── config.py        # Configuration and paths
│   │   ├── data_utils.py    # Data loading and preprocessing
│   │   ├── proj_mask.py     # Spectral projector implementation
│   │   ├── specproj_unet.py # UNet architecture
│   │   ├── losses.py        # Loss functions
│   │   ├── train.py         # Training script
│   │   └── infer.py         # Inference script
│   └── utils/
│       ├── update_kaggle_notebook.py  # Script to update Kaggle notebook
│       └── watch_and_update.py       # File watcher for automatic updates
├── kaggle_notebook.py       # Generated Kaggle notebook
└── requirements.txt         # Project dependencies
```

## Environment-Specific Setup

### Local Development
- Clone the repository using git:
  ```bash
  git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
  cd YaleGWI
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Download the dataset manually or using `kagglehub` as needed.

### Google Colab
**Note:** The repository cloning logic has been removed from `setup.py`. You must run the following code block in your first Colab cell to set up the environment:

```python
# Install required packages
!pip install kagglehub

# Clone the repository
!git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
%cd YaleGWI

# Import the project
from src.core.setup import setup_environment
from config import CFG

# Setup the environment
setup_environment()
```

This will install dependencies, clone the repository, and set up the environment for Colab. The dataset will be downloaded automatically by the setup script.

### AWS SageMaker
- Clone the repository and ensure it is available in the appropriate directory (e.g., `/opt/ml/code/YaleGWI`).
- The setup script will configure paths and expect the dataset in `/opt/ml/input/data`.

## Automatic Notebook Updates

The project includes an automatic update system that keeps the Kaggle notebook (`kaggle_notebook.py`) in sync with changes to the source files. There are two ways to use this:

1. **Manual Update**: Run the update script directly:
   ```bash
   python src/utils/update_kaggle_notebook.py
   ```

2. **Automatic Updates**: Start the file watcher to automatically update the notebook when source files change:
   ```bash
   python src/utils/watch_and_update.py
   ```

The file watcher will monitor all Python files in the `src` directory and its subdirectories. When a file is modified, it will automatically run the update script to regenerate the Kaggle notebook.

## Development Workflow

1. Make changes to the source files in `src/core/`
2. The changes will be automatically reflected in `kaggle_notebook.py` if the file watcher is running
3. If the file watcher is not running, manually run the update script
4. Test your changes in the Kaggle notebook

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```
