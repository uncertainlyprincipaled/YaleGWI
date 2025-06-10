# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

waveform_inversion_path = kagglehub.competition_download('waveform-inversion')
sharifi76_offile_openfwi_path = kagglehub.notebook_output_download('sharifi76/offile-openfwi')

print('Data source import complete.')

# SETUP

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import torch
import pandas as pd
from torchvision.transforms import Compose
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List
import logging
import csv
import torch.nn.functional as F
import csv

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

# PATHS
BASE_DIR = '/kaggle/input/waveform-inversion'
TRAIN_DIR = os.path.join(BASE_DIR, 'train_samples')
TEST_DIR = os.path.join(BASE_DIR, 'test')

print("Train Folders:", os.listdir(TRAIN_DIR))

# DEVICE/MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexConvNet().to(device)
print("Using device:", device)


# DATA / MODEL

def load_dataset_config(config_path, dataset_name):
    """Loads normalization parameters from dataset_config.json."""
    try:
        with open(config_path) as f:
            ctx = json.load(f)[dataset_name]
        print(f"Loaded config for dataset: {dataset_name}")
        return ctx
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Dataset '{dataset_name}' not found in {config_path}.")
        sys.exit(1)

def get_transforms(ctx, k):
    """Gets the transformations for data and label based on test.py."""
    log_data_min = T.log_transform(ctx['data_min'], k=k)
    log_data_max = T.log_transform(ctx['data_max'], k=k)
    transform_data = Compose([
        T.LogTransform(k=k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    return transform_data
    
# ================================================================
# Flexible Exploration for Any Family
# Auto-Skip empty folders
# ================================================================

def explore_family(folder_name):
    folder_path = os.path.join(TRAIN_DIR, folder_name)
    print(f"\nExploring {folder_name} Dataset")
    print("Available Files:", os.listdir(folder_path))

    seis_files = sorted([f for f in os.listdir(folder_path) if f.startswith('seis')])
    vel_files = sorted([f for f in os.listdir(folder_path) if f.startswith('vel')])

    print(f"Found {len(seis_files)} Seismic files")
    print(f"Found {len(vel_files)} Velocity files")

    # Check before loading
    if seis_files and vel_files:
        example_seis = load_npy(os.path.join(folder_path, seis_files[0]))
        example_vel = load_npy(os.path.join(folder_path, vel_files[0]))
        example_vel = np.squeeze(example_vel)

        print("Seismic Shape:", example_seis.shape)
        print("Velocity Shape:", example_vel.shape)
    else:
        print("Skipping... No seismic or velocity files found.")

# ================================================================
# Helper to Load Numpy file
# ================================================================
def load_npy(file_path):
    return np.load(file_path)
    
# =============================================================================
# 1. Data Preparation
# =============================================================================
def collect_input_files(data_dir: str) -> list:
    """
    Recursively search for .npy files in data_dir that contain 'seis' or 'data' in their filename.
    """
    return [f for f in Path(data_dir).rglob("*.npy") if ("seis" in f.stem) or ("data" in f.stem)]

def map_input_to_output(input_files: list) -> list:
    """
    Map each input file to its corresponding output file by replacing keywords.
    """
    return [Path(str(f).replace("seis", "vel").replace("data", "model")) for f in input_files]

# Define training sample directory
TRAIN_DIR = "/kaggle/input/waveform-inversion/train_samples"
inputs_all = collect_input_files(TRAIN_DIR)
outputs_all = map_input_to_output(inputs_all)

# Check all output files exist
assert all(f.exists() for f in outputs_all)

# Split dataset into training and validation based on sampling frequency
train_inputs = [inputs_all[i] for i in range(0, len(inputs_all), 2)]
valid_inputs = [f for f in inputs_all if f not in train_inputs]
train_outputs = map_input_to_output(train_inputs)
valid_outputs = map_input_to_output(valid_inputs)

# =============================================================================
# 2. Dataset Definition
# =============================================================================
class SeismicDataset(Dataset):
    """
    Dataset handling seismic files with multiple examples per file.
    """
    def __init__(self, in_files: list, out_files: list, examples_per_file: int = 500):
        assert len(in_files) == len(out_files)
        self.in_files = in_files
        self.out_files = out_files
        self.examples_per_file = examples_per_file

    def __len__(self):
        return len(self.in_files) * self.examples_per_file

    def __getitem__(self, idx: int):
        file_index = idx // self.examples_per_file
        sample_index = idx % self.examples_per_file

        # Memory map the file to reduce memory usage
        x_data = np.load(self.in_files[file_index], mmap_mode="r")
        y_data = np.load(self.out_files[file_index], mmap_mode="r")
        try:
            return x_data[sample_index].copy(), y_data[sample_index].copy()
        finally:
            del x_data, y_data

# Create DataLoaders for training and validation
train_dataset = SeismicDataset(train_inputs, train_outputs, examples_per_file=500)
valid_dataset = SeismicDataset(valid_inputs, valid_outputs, examples_per_file=500)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, pin_memory=True,
    drop_last=True, num_workers=4, persistent_workers=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=64, shuffle=False, pin_memory=True,
    drop_last=False, num_workers=4, persistent_workers=True
)

# =============================================================================
# 3. Model Architecture: SmartConvNet
# =============================================================================
class SmartConvNet(nn.Module):
    """A convolutional network with adaptive pooling and dense layers."""
    def __init__(self, input_channels: int = 5, output_size: int = 70 * 70):
        super().__init__()
        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # spatial reduction
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # further reduction
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Pool output to a fixed size
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feat = self.feature_extractor(x)
        pooled = self.pool(feat)
        flat = pooled.view(batch_size, -1)
        out = self.fc(flat)
        # Reshape output to (batch_size, 1, 70, 70) and apply scaling and bias
        return out.view(batch_size, 1, 70, 70) * 1000 + 1500

# -----------------------------------------------------------------------------
# Residual Block with Squeeze-and-Excitation (SE) Module
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    A residual block that optionally uses a squeeze-and-excitation (SE) module to
    adaptively weight the channels.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.use_se = use_se
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If there's a change in dimensions, adjust the shortcut (residual path)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Squeeze-and-Excitation module for channel attention
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)

# -----------------------------------------------------------------------------
# ComplexConvNet: Enhanced and Deeper Model Architecture
# -----------------------------------------------------------------------------
class ComplexConvNet(nn.Module):
    """
    A more complex deep convolutional network that uses a stem followed by a series of
    residual blocks with SE modules. The network employs global pooling and dense layers
    to produce an output of shape (batch_size, 1, 70, 70) with the desired scaling.
    """
    def __init__(self, input_channels=5, output_size=70*70):
        super().__init__()
        # Stem: initial feature extractor
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual layers with increasing feature channels, each using SE
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, use_se=True),
            ResidualBlock(64, 64, stride=1, use_se=True)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_se=True),
            ResidualBlock(128, 128, stride=1, use_se=True)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_se=True),
            ResidualBlock(256, 256, stride=1, use_se=True)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, use_se=True),
            ResidualBlock(512, 512, stride=1, use_se=True)
        )
        
        # Global Pooling to get fixed spatial dimensions regardless of input size
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dense (fully connected) layers for final processing
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 2048),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_size)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.fc(x)
        
        # Reshape output to (batch_size, 1, 70, 70) and apply scaling and offset
        return x.view(batch_size, 1, 70, 70) * 1000 + 1500

class TestDataset(Dataset):
    """
    Dataset for test files that returns the test data and its identifier.
    """
    def __init__(self, files: list):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        return np.load(file_path), file_path.stem


def load_dataset_config(config_path, dataset_name):
    """Loads normalization parameters from dataset_config.json."""
    try:
        with open(config_path) as f:
            ctx = json.load(f)[dataset_name]
        print(f"Loaded config for dataset: {dataset_name}")
        return ctx
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    except KeyError:
        print(f"Error: Dataset '{dataset_name}' not found in {config_path}.")
        sys.exit(1)

def get_transforms(ctx, k):
    """Gets the transformations for data and label based on test.py."""
    log_data_min = T.log_transform(ctx['data_min'], k=k)
    log_data_max = T.log_transform(ctx['data_max'], k=k)
    transform_data = Compose([
        T.LogTransform(k=k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    return transform_data
    
# ================================================================
# Flexible Exploration for Any Family
# Auto-Skip empty folders
# ================================================================

def explore_family(folder_name):
    folder_path = os.path.join(TRAIN_DIR, folder_name)
    print(f"\nExploring {folder_name} Dataset")
    print("Available Files:", os.listdir(folder_path))

    seis_files = sorted([f for f in os.listdir(folder_path) if f.startswith('seis')])
    vel_files = sorted([f for f in os.listdir(folder_path) if f.startswith('vel')])

    print(f"Found {len(seis_files)} Seismic files")
    print(f"Found {len(vel_files)} Velocity files")

    # Check before loading
    if seis_files and vel_files:
        example_seis = load_npy(os.path.join(folder_path, seis_files[0]))
        example_vel = load_npy(os.path.join(folder_path, vel_files[0]))
        example_vel = np.squeeze(example_vel)

        print("Seismic Shape:", example_seis.shape)
        print("Velocity Shape:", example_vel.shape)
    else:
        print("Skipping... No seismic or velocity files found.")

# ================================================================
# Helper to Load Numpy file
# ================================================================
def load_npy(file_path):
    return np.load(file_path)
    
# =============================================================================
# 1. Data Preparation
# =============================================================================
def collect_input_files(data_dir: str) -> list:
    """
    Recursively search for .npy files in data_dir that contain 'seis' or 'data' in their filename.
    """
    return [f for f in Path(data_dir).rglob("*.npy") if ("seis" in f.stem) or ("data" in f.stem)]

def map_input_to_output(input_files: list) -> list:
    """
    Map each input file to its corresponding output file by replacing keywords.
    """
    return [Path(str(f).replace("seis", "vel").replace("data", "model")) for f in input_files]

# Define training sample directory
TRAIN_DIR = "/kaggle/input/waveform-inversion/train_samples"
inputs_all = collect_input_files(TRAIN_DIR)
outputs_all = map_input_to_output(inputs_all)

# Check all output files exist
assert all(f.exists() for f in outputs_all)

# Split dataset into training and validation based on sampling frequency
train_inputs = [inputs_all[i] for i in range(0, len(inputs_all), 2)]
valid_inputs = [f for f in inputs_all if f not in train_inputs]
train_outputs = map_input_to_output(train_inputs)
valid_outputs = map_input_to_output(valid_inputs)

# =============================================================================
# 2. Dataset Definition
# =============================================================================
class SeismicDataset(Dataset):
    """
    Dataset handling seismic files with multiple examples per file.
    """
    def __init__(self, in_files: list, out_files: list, examples_per_file: int = 500):
        assert len(in_files) == len(out_files)
        self.in_files = in_files
        self.out_files = out_files
        self.examples_per_file = examples_per_file

    def __len__(self):
        return len(self.in_files) * self.examples_per_file

    def __getitem__(self, idx: int):
        file_index = idx // self.examples_per_file
        sample_index = idx % self.examples_per_file

        # Memory map the file to reduce memory usage
        x_data = np.load(self.in_files[file_index], mmap_mode="r")
        y_data = np.load(self.out_files[file_index], mmap_mode="r")
        try:
            return x_data[sample_index].copy(), y_data[sample_index].copy()
        finally:
            del x_data, y_data

# Create DataLoaders for training and validation
train_dataset = SeismicDataset(train_inputs, train_outputs, examples_per_file=500)
valid_dataset = SeismicDataset(valid_inputs, valid_outputs, examples_per_file=500)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, pin_memory=True,
    drop_last=True, num_workers=4, persistent_workers=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=64, shuffle=False, pin_memory=True,
    drop_last=False, num_workers=4, persistent_workers=True
)

# =============================================================================
# 3. Model Architecture: SmartConvNet
# =============================================================================
class SmartConvNet(nn.Module):
    """A convolutional network with adaptive pooling and dense layers."""
    def __init__(self, input_channels: int = 5, output_size: int = 70 * 70):
        super().__init__()
        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # spatial reduction
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # further reduction
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Pool output to a fixed size
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        feat = self.feature_extractor(x)
        pooled = self.pool(feat)
        flat = pooled.view(batch_size, -1)
        out = self.fc(flat)
        # Reshape output to (batch_size, 1, 70, 70) and apply scaling and bias
        return out.view(batch_size, 1, 70, 70) * 1000 + 1500

# -----------------------------------------------------------------------------
# Residual Block with Squeeze-and-Excitation (SE) Module
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    A residual block that optionally uses a squeeze-and-excitation (SE) module to
    adaptively weight the channels.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.use_se = use_se
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If there's a change in dimensions, adjust the shortcut (residual path)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Squeeze-and-Excitation module for channel attention
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)

# -----------------------------------------------------------------------------
# ComplexConvNet: Enhanced and Deeper Model Architecture
# -----------------------------------------------------------------------------
class ComplexConvNet(nn.Module):
    """
    A more complex deep convolutional network that uses a stem followed by a series of
    residual blocks with SE modules. The network employs global pooling and dense layers
    to produce an output of shape (batch_size, 1, 70, 70) with the desired scaling.
    """
    def __init__(self, input_channels=5, output_size=70*70):
        super().__init__()
        # Stem: initial feature extractor
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual layers with increasing feature channels, each using SE
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, use_se=True),
            ResidualBlock(64, 64, stride=1, use_se=True)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_se=True),
            ResidualBlock(128, 128, stride=1, use_se=True)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_se=True),
            ResidualBlock(256, 256, stride=1, use_se=True)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, use_se=True),
            ResidualBlock(512, 512, stride=1, use_se=True)
        )
        
        # Global Pooling to get fixed spatial dimensions regardless of input size
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dense (fully connected) layers for final processing
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 2048),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_size)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.fc(x)
        
        # Reshape output to (batch_size, 1, 70, 70) and apply scaling and offset
        return x.view(batch_size, 1, 70, 70) * 1000 + 1500

class TestDataset(Dataset):
    """
    Dataset for test files that returns the test data and its identifier.
    """
    def __init__(self, files: list):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        return np.load(file_path), file_path.stem

# CURVEFAULT_A

curve_fault_a_path = os.path.join(TRAIN_DIR, 'CurveFault_A')
print("Files in CurveFault_A:", os.listdir(curve_fault_a_path))
seis_file = os.path.join(curve_fault_a_path, 'seis2_1_0.npy')
vel_file = os.path.join(curve_fault_a_path, 'vel2_1_0.npy')

seis = load_npy(seis_file)
vel = load_npy(vel_file)

print("Seismic Data shape:", seis.shape)  
print("Velocity Data shape:", vel.shape)

vel = np.squeeze(vel)  

print("Velocity Shape after squeeze:", vel.shape)

sample_id = 0

# VELOCITY PLOT

plt.figure(figsize=(8, 6))
plt.title(f"Velocity Map (Ground Truth) - Sample {sample_id}")
sns.heatmap(vel[sample_id], cmap='viridis')
plt.show()

# SEISMIC DATA

plt.figure(figsize=(10, 6))
plt.title(f"Seismic Data - Batch 0, Source 0")
plt.imshow(seis[0, 0], aspect='auto', cmap='seismic')
plt.colorbar(label="Amplitude")
plt.xlabel("Receivers")
plt.ylabel("Timesteps")
plt.show()

# EXPLORE ALL FAMILIES

families = os.listdir(TRAIN_DIR)

for fam in families:
    explore_family(fam)

# TEST SET EXPLORATION

test_files = sorted(os.listdir(TEST_DIR))

print("\nSample Test File:", test_files[0])
sample_test = load_npy(os.path.join(TEST_DIR, test_files[0]))
print("Test Sample Shape:", sample_test.shape)

plt.figure(figsize=(10, 6))
plt.title("Test Seismic Sample Visualization")
# Showing full 2D seismic data → shape (Receivers, Timesteps)
plt.imshow(sample_test[0], aspect='auto', cmap='seismic')  
plt.colorbar()
plt.xlabel("Timesteps")
plt.ylabel("Receivers")
plt.show()

# TRAINING

# =============================================================================
# 5. Training Loop
# =============================================================================
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
n_epochs = 50

training_history = []

for epoch in range(1, n_epochs + 1):
    print(f"[{epoch:02d}] Starting training")
    
    # ----- Training Phase -----
    model.train()
    epoch_train_losses = []
    for batch_inputs, batch_targets in tqdm(train_loader, desc="Training", leave=False):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        
        epoch_train_losses.append(loss.item())
    avg_train_loss = np.mean(epoch_train_losses)
    print("Train loss: {:.5f}".format(avg_train_loss))

    # ----- Validation Phase -----
    model.eval()
    epoch_valid_losses = []
    for batch_inputs, batch_targets in tqdm(valid_loader, desc="Validation", leave=False):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        with torch.inference_mode():
            predictions = model(batch_inputs)
        loss = criterion(predictions, batch_targets)
        epoch_valid_losses.append(loss.item())
    avg_valid_loss = np.mean(epoch_valid_losses)
    print("Valid loss: {:.5f}".format(avg_valid_loss))
    
    training_history.append({
        "train": avg_train_loss,
        "valid": avg_valid_loss
    })

    # ----- Plot Example Outputs Every 4 Epochs -----
    if epoch % 4 == 0:
        sample_true = batch_targets[0, 0].detach().cpu()
        sample_pred = predictions[0, 0].detach().cpu()
        fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
        fig.suptitle(f"Epoch {epoch} | Valid: {avg_valid_loss:.5f}")
        axs[0].imshow(sample_true)
        axs[0].set_title("Ground Truth")
        axs[1].imshow(sample_pred)
        axs[1].set_title("Prediction")
        plt.show()

# Plot training history
pd.DataFrame(training_history).plot(title="Training and Validation Loss History");

# =============================================================================
# 6. Test Set Inference and Submission File
# =============================================================================
# Collect test files
test_dir = "/kaggle/input/waveform-inversion/test"
test_files = list(Path(test_dir).glob("*.npy"))
print("Test files:", len(test_files))

# Define CSV header details
x_cols = [f"x_{i}" for i in range(1, 70, 2)]
csv_fields = ["oid_ypos"] + x_cols

# Create test DataLoader
test_dataset = TestDataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=4, pin_memory=True)

# Switch model to evaluation mode
model.eval()

# Generate submission CSV
submission_path = "submission.csv"
with open(submission_path, "wt", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()
    for batch_inputs, batch_ids in tqdm(test_loader, desc="Test Inference"):
        batch_inputs = batch_inputs.to(device)
        with torch.inference_mode():
            outputs = model(batch_inputs)
        # Bring predictions back to CPU as numpy array
        predictions = outputs[:, 0].cpu().numpy()
        for pred_map, file_id in zip(predictions, batch_ids):
            # For each y position, extract every second element from x positions
            for y_index in range(70):
                row_dict = {
                    "oid_ypos": f"{file_id}_y_{y_index}",
                    **{x_cols[i]: pred_map[y_index, (i * 2) + 1] for i in range(len(x_cols))}
                }
                writer.writerow(row_dict)

print(f"Submission file generated: {submission_path}")

