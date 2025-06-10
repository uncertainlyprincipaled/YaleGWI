# SpecProj-UNet for Seismic Waveform Inversion

## Data IO Policy (IMPORTANT)
**All data loading, streaming, and batching must go through `DataManager` in `src/core/data_manager.py`.**
- Do NOT load data directly in any other file.
- This ensures memory efficiency and prevents RAM spikes in Kaggle.
- All source files must import and use DataManager for any data access.

## Overview
A deep learning solution for seismic waveform inversion using a HGNet/ConvNeXt backbone with:
- EMA (Exponential Moving Average) for stable training
- AMP (Automatic Mixed Precision) for faster training
- Distributed Data Parallel (DDP) for multi-GPU training
- Test Time Augmentation (TTA) for improved inference

## Quick Start

### Local Development Setup
1. Clone the repository:
```bash
git clone https://github.com/uncertainlyprincipaled/YaleGWI.git
cd YaleGWI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Development Workflow
1. **All data IO must use DataManager.**
2. Develop and test your code locally in `kaggle_notebook.py`
3. Copy relevant code chunks to Kaggle notebook cells as needed
4. Test functionality in Kaggle environment
5. Once everything works, copy the entire `kaggle_notebook.py` into a single Kaggle cell for final submission

### Kaggle Setup
1. Create a new notebook in the [Waveform Inversion competition](https://www.kaggle.com/competitions/waveform-inversion)

2. **Important**: Add the required datasets to your notebook first:
   - Click on the 'Data' tab
   - Click 'Add Data'
   - Search for and add:
     1. 'Waveform Inversion' competition dataset
     2. 'openfwi-preprocessed-72x72' dataset (contains preprocessed data and pretrained models)

3. Copy code chunks from `kaggle_notebook.py` into separate cells in your Kaggle notebook for testing

4. For final submission, copy the entire contents of `kaggle_notebook.py` into a single cell

## Code Structure
- `kaggle_notebook.py`: Main development file containing all code
- `src/core/data_manager.py`: **Single source of truth for all data IO**
- `requirements.txt`: Project dependencies

## TODO
- [ ] Add proper error handling for Kaggle environment detection
- [ ] Add data validation checks
- [ ] Add model checkpointing
- [ ] Add logging functionality
- [ ] Add visualization utilities
- [ ] Add test suite

## Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```

## Features

- **Physics-Guided Architecture**: Uses spectral projectors to inject wave-equation priors
- **Mixed Precision Training**: Supports both fp16 and bfloat16 for efficient training
- **Memory Optimization**: Implements efficient attention and memory management
- **Family-Aware Training**: Stratified sampling by geological families
- **Robust Training**: Includes gradient clipping, early stopping, and learning rate scheduling
- **Comprehensive Checkpointing**: Saves full training state for easy resumption

## Project Structure

```
.
├── src/
│   ├── core/
│   │   ├── config.py      # Configuration management
│   │   ├── train.py       # Training loop implementation
│   │   ├── model.py       # Model architecture
│   │   └── losses.py      # Loss functions
│   └── utils/
│       ├── data.py        # Data loading utilities
│       └── metrics.py     # Evaluation metrics
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
└── outputs/              # Training outputs and checkpoints
```

## Usage

### Training

```python
from src.core.train import train

# Start training with default settings
train(fp16=True)  # Enable mixed precision training
```

### Inference

```python
from src.core.model import get_model
import torch

# Load model and weights
model = get_model()
model.load_state_dict(torch.load('outputs/best.pth'))

# Run inference
predictions = model(input_data)
```

## Key Components

### Model Architecture
- SpecProj-UNet: Combines UNet with spectral projectors for physics-guided learning
- EMA: Exponential Moving Average for model weights
- Mixed Precision: Automatic mixed precision training support

### Training Features
- Gradient Clipping: Prevents exploding gradients
- Early Stopping: Stops training when validation performance plateaus
- Learning Rate Scheduling: Adaptive learning rate adjustment
- Memory Management: Efficient GPU memory usage
- Comprehensive Checkpointing: Saves full training state

### Loss Functions
- L1 Loss: Basic reconstruction loss
- PDE Residual: Physics-based consistency term
- Joint Loss: Combines multiple loss components

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
We use black for code formatting and flake8 for linting:
```bash
black src/
flake8 src/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Kaggle competition "Seismic Waveform Inversion"
- Inspired by various physics-guided deep learning approaches
- Uses PyTorch for deep learning implementation


