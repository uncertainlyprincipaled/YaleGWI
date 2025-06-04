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


