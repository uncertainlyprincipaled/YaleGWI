# 5 depth U net with residual

# -*- coding: utf-8 -*-
"""
Script for training a U-Net model (with Residual Blocks) for Full Waveform
Inversion, using data sourced solely from Kaggle input directories, with
data augmentation. Corrected generate_sample function.
"""

# %% Imports
# Standard Library Imports
import csv
import gc
import glob
import os
import random
import shutil
import sys
from pathlib import Path

# Third-party Imports
# Install webdataset if not present (useful in notebook environments)
try:
    import webdataset as wds
except ImportError:
    print("Installing webdataset...")
    !pip install webdataset
    import webdataset as wds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF  # For Augmentation
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# %% Configuration
class cfg:
    """Configuration parameters for the workflow."""

    # --- Paths ---
    kaggle_train_dir = "/kaggle/input/waveform-inversion/train_samples"
    kaggle_test_dir = "/kaggle/input/waveform-inversion/test"
    shard_output_dir = "/kaggle/working/sharded_data"
    working_dir = "/kaggle/working/"
    submission_file = os.path.join(working_dir, "submission.csv")

    # --- Dataset Params ---
    dataset_name = "fwi_kaggle_only_augmented_resnet" # Updated name

    # --- Sharding Params ---
    maxsize = 1e9  # Approx 1 GB
    force_shard_creation = False

    # --- Splitting & Loading Params ---
    num_used_shards = None  # Use all available
    test_size = 0.1  # Proportion for validation split
    batch_size = 16
    num_workers = 2

    # --- Augmentation Params ---
    apply_augmentation = True
    aug_hflip_prob = 0.5  # Probability of horizontal flip
    aug_seis_noise_std = 0.01  # Std dev of Gaussian noise added to seismic

    # --- Model params (U-Net with Residual Blocks) ---
    unet_in_channels = 5
    unet_out_channels = 1
    unet_init_features = 32
    unet_depth = 5  # Number of downsampling stages
    unet_bilinear = True # Upsampling method

    # --- Training params ---
    n_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    plot_every_n_epochs = 5

    # --- Misc ---
    seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    autocast_dtype = torch.float16 if use_cuda else torch.bfloat16


# %% Helper Functions
def set_seed(seed=42):
    """Sets seed for reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure reproducibility if desired, may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def find_best_model(model_dir=cfg.working_dir, model_prefix="unet_best_model"):
    """
    Finds the best model file based on filename pattern (lowest loss).
    Falls back to most recently created/modified if pattern fails or doesn't exist.
    """
    best_loss = float("inf")
    best_model_path = None
    pattern = os.path.join(model_dir, f"{model_prefix}_epoch_*_loss_*.pth")
    model_files = glob.glob(pattern)

    if not model_files:
        # Fallback 1: No pattern match -> find latest created .pth
        print(f"W: No models matching pattern '{pattern}'. Looking for *.pth")
        all_pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if all_pth_files:
            best_model_path = max(all_pth_files, key=os.path.getctime, default=None)
            if best_model_path:
                print(
                    f"Using most recently created: {os.path.basename(best_model_path)}"
                )
            else:
                print("W: No .pth models found.")
                return None
        else:
            print("W: No .pth models found in model directory.")
            return None

    elif "loss" in os.path.basename(pattern):
        # Try parsing loss from filename
        parsed_models = []
        for f in model_files:
            try:
                loss_str = f.split("_loss_")[-1].split(".pth")[0]
                loss = float(loss_str)
                parsed_models.append((loss, f))
            except (ValueError, IndexError, AttributeError):
                print(f"W: Couldn't parse loss from filename: {os.path.basename(f)}")

        if parsed_models:
            # Found models with parseable loss, sort by loss
            parsed_models.sort(key=lambda x: x[0])
            best_loss, best_model_path = parsed_models[0]
            print(
                f"Found best model by loss: {os.path.basename(best_model_path)} (Loss: {best_loss:.4f})"
            )
        elif model_files:
            # Pattern matched, but loss couldn't be parsed from any filename
            print(
                "W: Pattern matched but no losses parsed. Selecting most recently created."
            )
            best_model_path = max(model_files, key=os.path.getctime, default=None)
            if best_model_path:
                print(f"Using most recent creation time: {os.path.basename(best_model_path)}")

    else:
        # Pattern matched but doesn't contain "loss" part (unexpected)
        if model_files:
            print(
                f"W: Pattern matched but no loss info expected. Selecting most recently created."
            )
            best_model_path = max(model_files, key=os.path.getctime, default=None)
            if best_model_path:
                print(
                    f"Using most recent creation time match: {os.path.basename(best_model_path)}"
                )

    if not best_model_path:
        # Final fallback: If no model found yet, use the latest modified .pth file
        all_pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if all_pth_files:
            print("W: Fallback: Selecting most recently modified .pth file.")
            best_model_path = max(all_pth_files, key=os.path.getmtime, default=None)
            if best_model_path:
                print(
                    f"Using most recent modification time: {os.path.basename(best_model_path)}"
                )

    return best_model_path


# %% WebDataset Preprocessing Functions
def search_data_path(target_dirs, root_dir, shuffle=True, seed=42):
    """Finds input/output .npy file pairs within subdirectories of a root directory."""
    files = []
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"W: Root directory not found: {root_path}")
        return []

    print(f"Searching for data families {target_dirs} in root: {root_path}")
    total_pairs_found = 0
    for target_dir in target_dirs:
        data_dir = root_path / target_dir
        if not data_dir.is_dir():
            # print(f"W: Target directory {target_dir} not found in {root_path}")
            continue

        in_files, out_files = [], []
        data_subdir = data_dir / "data"
        model_subdir = data_dir / "model"

        # Check for HF structure first, then Kaggle structure
        if data_subdir.is_dir() and model_subdir.is_dir():
            in_files = sorted(data_subdir.glob("*.npy"))
            out_files = sorted(model_subdir.glob("*.npy"))
            # print(f"Found {len(in_files)}/{len(out_files)} files (HF style) in {target_dir}")
        else:
            in_files = sorted(data_dir.glob("seis*.npy"))
            out_files = sorted(data_dir.glob("vel*.npy"))
            # print(f"Found {len(in_files)}/{len(out_files)} files (Kaggle style) in {target_dir}")

        if not in_files or len(in_files) != len(out_files):
            if in_files or out_files:  # Only warn if some files were found
                print(
                    f"W: Mismatch or missing files in {data_dir} (in:{len(in_files)}, out:{len(out_files)}). Skipping."
                )
            continue

        current_pairs = list(zip(in_files, out_files))
        files.extend(current_pairs)
        total_pairs_found += len(current_pairs)

    print(f"Found {len(files)} total valid pairs across specified families.")
    if shuffle and files:
        print(f"Shuffling {len(files)} pairs (seed={seed}).")
        rng = np.random.default_rng(seed)
        rng.shuffle(files)

    return files


# ==================================================
# CORRECTED generate_sample function (No finally block)
# ==================================================
def generate_sample(in_file, out_file=None, base_dir=None):
    """
    Loads data from .npy files, prepares dicts for WebDataset, converts to float16.
    Handles errors during loading gracefully.
    """
    data = []
    seis = None  # Initialize to ensure variable exists for potential del
    vel = None
    try:
        if out_file is None:
            # Logic for test data sharding (if needed later) - not implemented here
            print("W: generate_sample called without out_file (test mode?), not implemented.")
            return []
        else:
            # --- Load Train/Validation data ---
            try:
                # Use mmap_mode='r' for memory efficiency if files are large
                seis = np.load(in_file, mmap_mode="r")
            except Exception as e:
                print(f"E: Load fail for input {in_file.name}: {e}")
                return []  # Exit early if input fails

            try:
                vel = np.load(out_file, mmap_mode="r")
            except Exception as e:
                print(f"E: Load fail for output {out_file.name}: {e}")
                # Clean up the already loaded seis if vel loading fails
                if seis is not None:
                    del seis
                return []  # Exit early if output fails

            # --- Validate shapes and determine number of samples ---
            n_samples = 0
            if seis.ndim == 4 and vel.ndim == 4:  # Batch of samples (N, C, H, W)
                if seis.shape[0] != vel.shape[0]:
                    print(
                        f"W: Batch size mismatch in {in_file.name} ({seis.shape[0]}) vs {out_file.name} ({vel.shape[0]})"
                    )
                    del seis, vel
                    return []
                n_samples = seis.shape[0]
            elif seis.ndim == 3 and vel.ndim == 3:  # Single sample (C, H, W)
                n_samples = 1
            else:
                # Raise error for unexpected dimensions
                raise ValueError(
                    f"Unexpected dims: seis {seis.shape}, vel {vel.shape} in {in_file.name}"
                )

            if n_samples == 0:
                print(f"W: Found 0 samples in pair: {in_file.name}, {out_file.name}")
                del seis, vel
                return []

            # --- Generate unique key based on file path relative to base_dir ---
            common_part = f"{in_file.parent.name}_{in_file.stem}"  # Default key
            if base_dir:
                try:
                    # Create key from relative path parts, removing .npy suffix
                    relative_path = in_file.relative_to(base_dir)
                    common_part = "_".join(relative_path.parts).replace(".npy", "")
                    # Ensure compatibility across OS path separators
                    common_part = common_part.replace(os.sep, "_").replace("\\", "_")
                except ValueError:
                    # If relative_to fails (e.g., different drives), use the default key
                    pass

            # --- Process and append each sample ---
            for i in range(n_samples):
                key = f"{common_part}_{i}"
                # Extract sample, explicitly copy, and convert to float16
                s_sample = (
                    seis[i].copy().astype(np.float16)
                    if seis.ndim == 4
                    else seis.copy().astype(np.float16)
                )
                v_sample = (
                    vel[i].copy().astype(np.float16)
                    if vel.ndim == 4
                    else vel.copy().astype(np.float16)
                )
                data.append(
                    {
                        "__key__": key,
                        "sample_id.txt": key,  # Store key as text too
                        "seis.npy": s_sample,
                        "vel.npy": v_sample,
                    }
                )

            # --- Explicitly delete mmap objects after copying data ---
            # This is important to release file handles, especially with mmap
            del seis
            del vel

    except Exception as e:
        # Catch other errors (ValueError from dim check, key gen errors, etc.)
        print(f"E: Error during sample generation for {in_file.name}: {e}")
        # Explicitly try deleting here, in case they were loaded before the error
        if seis is not None:
            try:
                del seis
            except NameError:  # Should not happen if assigned None initially
                pass
        if vel is not None:
            try:
                del vel
            except NameError:
                pass
        return []  # Return empty list on any error during processing

    # No finally block needed as del is handled within try/except scopes
    return data


# ==================================================


# %% WebDataset Loading Functions
def get_shard_paths(
    root_dir, dataset_name, stage, num_shards=None, test_size=0.2, seed=42
):
    """Gets list of shard paths, optionally selects subset, optionally splits train/val."""
    source_dir_name = f"train_{dataset_name}"
    dataset_dir = Path(root_dir) / source_dir_name
    print(f"Looking for shards for stage '{stage}' in: {dataset_dir}")

    if not dataset_dir.is_dir():
        print(f"W: Shard directory not found: {dataset_dir}")
        return (None, None) if stage == "train" else None

    shard_paths = sorted([str(p) for p in dataset_dir.glob("*.tar")])

    if not shard_paths:
        print(f"W: No .tar shards found in {dataset_dir}.")
        return (None, None) if stage == "train" else None

    print(f"Found {len(shard_paths)} total shards.")

    # --- Shard Selection Logic ---
    selected_paths = shard_paths
    available_count = len(shard_paths)
    if num_shards is not None:
        if 0 < num_shards < available_count:
            print(f"Selecting {num_shards} shards randomly (seed={seed}).")
            rng = np.random.default_rng(seed)
            indices = rng.choice(available_count, size=num_shards, replace=False)
            selected_paths = sorted([shard_paths[i] for i in indices])
        elif num_shards >= available_count:
            print(
                f"Requested {num_shards} or more shards, using all {available_count} available."
            )
        else:  # num_shards <= 0
            print(
                f"W: Invalid num_shards ({num_shards}). Using all {available_count} shards."
            )
    print(f"Using {len(selected_paths)} selected shards for stage '{stage}'.")

    # --- Train/Validation Split Logic ---
    if stage == "train":
        count = len(selected_paths)
        print(f"Splitting {count} selected shards (test_size={test_size}, seed={seed})")
        try:
            if not (0 <= test_size < 1):
                raise ValueError("test_size must be in [0, 1)")
            if count <= 1 or test_size == 0:
                reason = "only 1 shard" if count <= 1 else "test_size is 0"
                print(f"W: Cannot split for validation ({reason}). Assigning all to train.")
                return sorted(selected_paths), []
            else:
                trn_paths, val_paths = train_test_split(
                    selected_paths, test_size=test_size, random_state=seed, shuffle=True
                )
                trn_paths.sort()
                val_paths.sort()
                print(f"# Train shards: {len(trn_paths)}, # Val shards: {len(val_paths)}")
                return trn_paths, val_paths
        except Exception as e:
            print(f"E: Failed to split shards: {e}")
            return None, None
    else:  # Not 'train' stage (e.g., 'val' direct loading or 'test')
        print(f"# Shards returned for stage '{stage}': {len(selected_paths)}")
        return sorted(selected_paths)


def get_dataset(paths, stage, seed=42):
    """Creates WebDataset object. Applies augmentations if stage=='train'."""
    if not paths:
        print(f"W: No shard paths provided for stage '{stage}'. Cannot create dataset.")
        return None

    print(f"Creating WebDataset for stage '{stage}' from {len(paths)} shards.")
    is_train = stage == "train"
    # Continue pipeline even if some samples fail decoding/mapping
    map_handler = wds.warn_and_continue

    try:
        dataset = wds.WebDataset(
            paths, nodesplitter=wds.split_by_node, shardshuffle=is_train, seed=seed
        )
        # Decode standard types (.npy, .txt, etc.)
        dataset = dataset.decode(handler=map_handler)

        def map_train_val(sample):
            """Inner function to process decoded samples and apply augmentations."""
            key_info = sample.get("__key__", "N/A")  # For error reporting
            try:
                required = ["sample_id.txt", "seis.npy", "vel.npy"]
                if not all(k in sample for k in required):
                    raise KeyError(f"Missing required keys in sample {key_info}")

                sid = sample["sample_id.txt"]
                # Ensure numpy arrays and convert to float32 tensors
                s_np = np.asarray(sample["seis.npy"])
                v_np = np.asarray(sample["vel.npy"])
                seis_tensor = torch.from_numpy(s_np).float()
                vel_tensor = torch.from_numpy(v_np).float()

                # --- Augmentation Block ---
                if is_train and cfg.apply_augmentation:
                    # 1. Horizontal Flip
                    if torch.rand(1).item() < cfg.aug_hflip_prob:
                        seis_tensor = TF.hflip(seis_tensor)
                        vel_tensor = TF.hflip(vel_tensor)
                    # 2. Add Gaussian Noise to Seismic Data
                    if cfg.aug_seis_noise_std > 0:
                        noise = torch.randn_like(seis_tensor) * cfg.aug_seis_noise_std
                        seis_tensor.add_(noise)  # In-place addition

                return {"sample_id": sid, "seis": seis_tensor, "vel": vel_tensor}

            except Exception as map_e:
                print(f"E: Map function failed for sample {key_info}: {map_e}")
                # Let the handler decide whether to skip or raise
                raise map_e

        # Apply the mapping function to train/val stages
        if stage in ["train", "val"]:
            dataset = dataset.map(map_train_val, handler=map_handler)

        # Shuffle buffer for training data
        if is_train:
            dataset = dataset.shuffle(1000)  # Buffer size for shuffling

        return dataset

    except Exception as e:
        print(f"E: Error creating WebDataset pipeline for stage '{stage}': {e}")
        return None


# %% Kaggle TestSet Loading (Directly from .npy)
class KaggleTestDataset(Dataset):
    """Loads the final Kaggle test set directly from individual .npy files."""

    def __init__(self, test_files_dir):
        self.test_files_dir = Path(test_files_dir)
        self.test_files = []
        try:
            if not self.test_files_dir.is_dir():
                raise FileNotFoundError(
                    f"Kaggle test directory missing: {self.test_files_dir}"
                )
            self.test_files = sorted(list(self.test_files_dir.glob("*.npy")))
            print(
                f"Found {len(self.test_files)} '.npy' files in Kaggle test dir: {self.test_files_dir}"
            )
            if not self.test_files:
                print(f"W: No .npy files found in {self.test_files_dir}.")
        except Exception as e:
            print(
                f"E: Error accessing Kaggle test directory {self.test_files_dir}: {e}"
            )

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        if not self.test_files or index >= len(self.test_files):
            raise IndexError(
                f"Index {index} out of bounds for KaggleTestDataset ({len(self.test_files)} files)."
            )
        test_file_path = self.test_files[index]
        try:
            # Load numpy array and convert to float32 tensor
            data = torch.from_numpy(np.load(test_file_path).astype(np.float32))
            # Get the original ID (filename without extension)
            original_id = test_file_path.stem
            return data, original_id
        except Exception as e:
            # Raise a more informative error if loading fails
            raise IOError(f"Error loading Kaggle test file: {test_file_path}") from e


# %% U-Net Model Definition (Formatted for Readability with Residual Blocks)

class ResidualDoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2 + Residual Connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution layer
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle potential channel mismatch
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # Projection shortcut: 1x1 conv + BN to match output channels
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # Store the input for the residual connection

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (without final ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply shortcut to the identity path
        identity_mapped = self.shortcut(identity)

        # Add the residual connection
        out += identity_mapped

        # Apply final ReLU
        out = self.relu(out)
        return out


class Up(nn.Module):
    """Upscaling then ResidualDoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            # Input to ResidualDoubleConv = channels from upsampled layer below + channels from skip connection
            # Output of ResidualDoubleConv = desired output channels for this decoder stage
            self.conv = ResidualDoubleConv(in_channels + out_channels, out_channels) # Use ResidualDoubleConv

        else: # Using ConvTranspose2d
            # ConvTranspose halves the channels: in_channels -> in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Input channels to ResidualDoubleConv
            conv_in_channels = in_channels // 2 # Channels after ConvTranspose
            skip_channels = out_channels       # Channels from skip connection
            total_in_channels = conv_in_channels + skip_channels
            self.conv = ResidualDoubleConv(total_in_channels, out_channels) # Use ResidualDoubleConv

    def forward(self, x1, x2):
        # x1 is the feature map from the layer below (needs upsampling)
        # x2 is the skip connection from the corresponding encoder layer
        x1 = self.up(x1)

        # Pad x1 if its dimensions don't match x2 after upsampling
        # Input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        # Pad format: (padding_left, padding_right, padding_top, padding_bottom)
        x1 = F.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 Convolution for the output layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture implementation with Residual Blocks"""

    def __init__(
        self,
        n_channels=cfg.unet_in_channels,
        n_classes=cfg.unet_out_channels,
        init_features=cfg.unet_init_features,
        depth=cfg.unet_depth, # number of pooling layers
        bilinear=cfg.unet_bilinear,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        self.initial_pool = nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1))

        # --- Encoder ---
        self.encoder_convs = nn.ModuleList() # Store conv blocks
        self.encoder_pools = nn.ModuleList() # Store pool layers

        # Initial conv block (no pooling before it)
        # Use ResidualDoubleConv for the initial convolution block
        self.inc = ResidualDoubleConv(n_channels, init_features)
        self.encoder_convs.append(self.inc)

        current_features = init_features
        for _ in range(depth):
            # Define convolution block for this stage
            conv = ResidualDoubleConv(current_features, current_features * 2)
            # Define pooling layer for this stage
            pool = nn.MaxPool2d(2)
            self.encoder_convs.append(conv)
            self.encoder_pools.append(pool)
            current_features *= 2

        # --- Bottleneck ---
        # Use ResidualDoubleConv for the bottleneck
        self.bottleneck = ResidualDoubleConv(current_features, current_features)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Input features start from bottleneck output features
        # Output features at each stage are halved
        for _ in range(depth):
            # Up block uses ResidualDoubleConv internally and handles channels
            up_block = Up(current_features, current_features // 2, bilinear)
            self.decoder_blocks.append(up_block)
            current_features //= 2 # Halve features for next Up block input

        # --- Output Layer ---
        # Input features are the output features of the last Up block
        self.outc = OutConv(current_features, n_classes)

    def _pad_or_crop(self, x, target_h=70, target_w=70):
        """Pads or crops input tensor x to target height and width."""
        _, _, h, w = x.shape
        # Pad Height if needed
        if h < target_h:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            x = F.pad(x, (0, 0, pad_top, pad_bottom))  # Pad height only
            h = target_h
        # Pad Width if needed
        if w < target_w:
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0))  # Pad width only
            w = target_w
        # Crop Height if needed
        if h > target_h:
            crop_top = (h - target_h) // 2
            # Use slicing to crop
            x = x[:, :, crop_top : crop_top + target_h, :]
            h = target_h
        # Crop Width if needed
        if w > target_w:
            crop_left = (w - target_w) // 2
            x = x[:, :, :, crop_left : crop_left + target_w]
            w = target_w
        return x

    def forward(self, x):
        # Initial pooling and resizing
        x_pooled = self.initial_pool(x)
        x_resized = self._pad_or_crop(x_pooled, target_h=70, target_w=70)

        # --- Encoder Path ---
        skip_connections = []
        xi = x_resized

        # Apply initial conv (inc)
        xi = self.encoder_convs[0](xi)
        skip_connections.append(xi) # Store output of inc

        # Apply subsequent encoder convs and pools
        # self.depth is the number of pooling layers
        for i in range(self.depth):
            # Apply conv block for this stage
            xi = self.encoder_convs[i+1](xi)
            # Store skip connection *before* pooling
            skip_connections.append(xi)
            # Apply pooling layer for this stage
            xi = self.encoder_pools[i](xi)

        # Apply bottleneck conv
        xi = self.bottleneck(xi)

        # --- Decoder Path ---
        xu = xi # Start with bottleneck output
        # Iterate through decoder blocks and corresponding skip connections in reverse
        for i, block in enumerate(self.decoder_blocks):
            # Determine the correct skip connection index from the end
            # Example: depth=5. Skips stored: [inc, enc1, enc2, enc3, enc4] (indices 0-4)
            # Decoder 0 (Up(1024, 512)) needs skip 4 (enc4)
            # Decoder 1 (Up(512, 256)) needs skip 3 (enc3) ...
            # Decoder 4 (Up(64, 32)) needs skip 0 (inc)
            skip_index = self.depth - 1 - i
            skip = skip_connections[skip_index]
            xu = block(xu, skip) # Up block combines xu (from below) and skip

        # --- Final Output ---
        logits = self.outc(xu)
        # Apply scaling and offset specific to the problem's target range
        output = logits * 1000.0 + 1500.0
        return output


# %% Main Execution
print("--- Starting Full Workflow (U-Net with Residual Blocks) ---")
set_seed(cfg.seed)
print(f"Device: {cfg.device}")
print(f"Using PyTorch version: {torch.__version__}")
if cfg.use_cuda:
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# Cleanup Code
# ==============================================================================
print("\n--- Cleaning up previous run artifacts ---")
paths_to_clean = [cfg.shard_output_dir]
# Find previous best model files based on pattern
model_pattern = os.path.join(cfg.working_dir, "unet_best_model_epoch_*_loss_*.pth")
paths_to_clean.extend(glob.glob(model_pattern))
# Add plot and submission files
paths_to_clean.append(os.path.join(cfg.working_dir, "training_history.png"))
paths_to_clean.append(cfg.submission_file)

for path_str in paths_to_clean:
    path_obj = Path(path_str)
    try:
        if path_obj.is_dir():
            print(f"Attempting to remove directory: {path_obj}")
            shutil.rmtree(path_obj, ignore_errors=True)
            print(f"Removed directory (if existed): {path_obj}")
        elif path_obj.is_file():
            print(f"Attempting to remove file: {path_obj}")
            path_obj.unlink(missing_ok=True)  # Ignore error if file doesn't exist
            print(f"Removed file (if existed): {path_obj}")
    except Exception as e:
        print(f"W: Error during cleanup of {path_obj}: {e}")
print("--- Cleanup finished ---")
gc.collect()
# ==============================================================================

# ==============================================================================
# SECTION 0/1: Sharding from Kaggle Data Only
# ==============================================================================
print("\n--- 0/1. Sharding from Kaggle Data Only ---")
# Use updated dataset name in shard path
shard_stage_dir = Path(cfg.shard_output_dir) / f"train_{cfg.dataset_name}"
kaggle_train_root = Path(cfg.kaggle_train_dir)
needs_creation = True
total_samples_written = 0

try:
    # --- Check if shards need creating ---
    if shard_stage_dir.exists() and any(shard_stage_dir.glob("*.tar")):
        if cfg.force_shard_creation:
            print(f"Forcing shard creation. Removing existing shards in {shard_stage_dir}")
            shutil.rmtree(shard_stage_dir)
        else:
            print(f"Found existing shards at: {shard_stage_dir}. Skipping creation.")
            needs_creation = False

    # --- Ensure output directories exist & Check Disk Space ---
    print("\n--- Checking Disk Space Before Directory Creation ---")
    try:
        total, used, free = shutil.disk_usage(cfg.working_dir)
        print(
            f"Disk Usage for {cfg.working_dir}: Total={total / 1e9:.2f}GB, Used={used / 1e9:.2f}GB, Free={free / 1e9:.2f}GB"
        )
    except Exception as du_e:
        print(f"W: Could not check disk usage: {du_e}")

    try:
        Path(cfg.shard_output_dir).mkdir(parents=True, exist_ok=True)
        shard_stage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"E: Critical error creating output directories: {e}")
        raise  # Stop if directories can't be created

    # --- Sharding Process ---
    if needs_creation:
        print(
            f"Starting shard creation from {kaggle_train_root} into {shard_stage_dir}"
        )
        if not kaggle_train_root.is_dir():
            raise FileNotFoundError(f"Kaggle train directory not found: {kaggle_train_root}")

        # Find family subdirectories in the Kaggle train directory
        families = [d.name for d in kaggle_train_root.iterdir() if d.is_dir()]
        print(f"Searching Kaggle data families: {families}")
        if not families:
            raise FileNotFoundError(
                f"No family subdirectories found in {kaggle_train_root}"
            )

        print("Searching for all data pairs in Kaggle source...")
        kaggle_file_pairs = search_data_path(
            families, kaggle_train_root, shuffle=True, seed=cfg.seed
        )
        print(f"Found {len(kaggle_file_pairs)} total valid pairs from Kaggle source.")
        if not kaggle_file_pairs:
            raise RuntimeError(
                "No valid data pairs found in the specified Kaggle directories."
            )

        # --- Write Shards ---
        shard_pattern = str(shard_stage_dir / "%06d.tar")
        print(
            f"Writing shards using pattern {shard_pattern} (max size {cfg.maxsize / 1e9:.2f} GB)"
        )
        with wds.ShardWriter(shard_pattern, maxsize=int(cfg.maxsize)) as writer:
            common_base_dir = kaggle_train_root  # For relative path key generation
            for in_file, out_file in tqdm(
                kaggle_file_pairs, desc="Sharding Kaggle Data", unit="pair"
            ):
                # generate_sample handles potential errors for each pair
                samples_from_pair = generate_sample(
                    Path(in_file), Path(out_file), base_dir=common_base_dir
                )
                if samples_from_pair:
                    for sample_dict in samples_from_pair:
                        writer.write(sample_dict)
                    total_samples_written += len(samples_from_pair)

        print(
            f"Finished writing {total_samples_written} samples from Kaggle source to shards."
        )

    elif not needs_creation:
        existing_shard_count = len(list(shard_stage_dir.glob("*.tar")))
        print(f"Using {existing_shard_count} existing shards.")

except Exception as e:
    print(f"E: Kaggle-only sharding process failed critically: {e}")
    import traceback

    traceback.print_exc()
    raise
# ==============================================================================


# --- 2. Get Train/Val DataLoaders from Created Shards ---
print("\n--- 2. Creating DataLoaders from Shards ---")
dltrain, dlvalid = None, None
val_paths_saved = []  # Keep track of validation paths for potential later use
try:
    # Use updated dataset name for getting paths
    trn_paths, val_paths = get_shard_paths(
        cfg.shard_output_dir,
        cfg.dataset_name, # Use updated name
        "train",  # Request splitting
        num_shards=cfg.num_used_shards,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )
    val_paths_saved = val_paths  # Save the validation paths

    if trn_paths is None:
        # get_shard_paths returns None, None on critical split error
        raise RuntimeError("Failed to get or split shard paths for train/val.")

    # Check if any shards actually exist if paths were returned empty
    # Use updated dataset name in check path
    shard_check_dir = Path(cfg.shard_output_dir) / f"train_{cfg.dataset_name}"
    if not trn_paths and not list(shard_check_dir.glob("*.tar")):
        raise RuntimeError(
            f"No training shards selected AND no .tar files found in {shard_check_dir}."
        )

    # Report shard counts
    if not trn_paths:
        print("W: No shards assigned for training. Training cannot proceed.")
    else:
        print(f"Using {len(trn_paths)} shards for training.")
    if not val_paths:
        print("W: No shards assigned for validation. Validation will be skipped.")
    else:
        print(f"Using {len(val_paths)} shards for validation.")

    # Create WebDatasets (Augmentation applied in get_dataset for 'train')
    trn_ds = get_dataset(trn_paths, "train", seed=cfg.seed) if trn_paths else None
    val_ds = get_dataset(val_paths, "val", seed=cfg.seed + 1) if val_paths else None

    # Check if dataset creation failed unexpectedly
    if trn_ds is None and trn_paths:
        raise RuntimeError("Failed to create train WebDataset pipeline.")
    if val_ds is None and val_paths:
        # Only warn if validation dataset failed, training might still proceed
        print("W: Failed to create validation WebDataset pipeline.")

    # Create DataLoaders
    if trn_ds:
        n_trn_w = min(cfg.num_workers, len(trn_paths)) if trn_paths else 0
        p_trn = n_trn_w > 0  # Use persistent workers only if num_workers > 0
        dltrain = DataLoader(
            trn_ds.batched(cfg.batch_size),
            batch_size=None,  # Already batched by WebDataset
            shuffle=False,  # Shuffling done by WebDataset
            num_workers=n_trn_w,
            pin_memory=cfg.use_cuda,
            persistent_workers=p_trn,
            prefetch_factor=2 if p_trn else None,  # Only relevant if num_workers > 0
        )
        print(f"Train DataLoader created with {n_trn_w} workers.")
    if val_ds:
        n_val_w = min(cfg.num_workers, len(val_paths)) if val_paths else 0
        p_val = n_val_w > 0
        dlvalid = DataLoader(
            val_ds.batched(cfg.batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=n_val_w,
            pin_memory=cfg.use_cuda,
            persistent_workers=p_val,
            prefetch_factor=2 if p_val else None,
        )
        print(f"Validation DataLoader created with {n_val_w} workers.")

    # Final check (can sometimes trigger TypeError: 'IterableDataset' has no len())
    try:
        loaders_exist = bool(dltrain or dlvalid)
        if not loaders_exist and (trn_paths or val_paths):
             # Should not happen if datasets were created but loaders failed
             raise RuntimeError("Loaders missing despite dataset paths existing.")
        print("DataLoader(s) created successfully or skipped appropriately.")
    except TypeError as te:
        # Expected error for IterableDataset without explicit length
        if "has no len()" in str(te):
            print(f"W: Caught expected TypeError '{te}'. Assume DataLoaders are ok.")
        else:
            raise te  # Re-raise unexpected TypeError
except Exception as e:
    print(f"E: DataLoader creation failed critically: {e}")
    import traceback
    traceback.print_exc()
    raise


# --- 3. Initialize Model, Loss, Optimizer ---
print("\n--- 3. Initializing Model, Loss, Optimizer ---")
model = None
try:
    # Instantiate the modified U-Net
    model = UNet().to(cfg.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} (Residual), Trainable Params: {params:,}")
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    print(f"Loss Function: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__} (lr={cfg.learning_rate}, wd={cfg.weight_decay})")
except Exception as e:
    print(f"E: Model initialization failed: {e}")
    raise


# --- 4. Training Loop ---
print("\n--- 4. Starting Training ---")
history = []
best_val_loss = float("inf")

if dltrain is None or model is None:
    print("E: Training cannot proceed. Train DataLoader or Model is missing.")
else:
    try:
        for epoch in range(1, cfg.n_epochs + 1):
            print(f"\n=== Epoch {epoch}/{cfg.n_epochs} ===")
            # --- Training Phase ---
            gc.collect()
            if cfg.use_cuda:
                torch.cuda.empty_cache()
            model.train()
            train_losses = []
            pbar_train = tqdm(dltrain, desc=f"Train E{epoch}", leave=False, unit="batch")
            for i, batch in enumerate(pbar_train):
                if not batch or "seis" not in batch or "vel" not in batch:
                    print(f"W: Skipping invalid train batch {i}")
                    continue
                try:
                    inputs = batch["seis"].to(cfg.device, non_blocking=True).float()
                    targets = batch["vel"].to(cfg.device, non_blocking=True).float()

                    optimizer.zero_grad(set_to_none=True)
                    # Use Automatic Mixed Precision (AMP) if on CUDA
                    with torch.amp.autocast(
                        device_type=cfg.device.type,
                        dtype=cfg.autocast_dtype,
                        enabled=cfg.use_cuda,
                    ):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    # Backward pass and optimization step
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                    # Update progress bar description
                    if i % 100 == 0:
                        pbar_train.set_postfix(loss=f"{np.mean(train_losses):.5f}")

                except Exception as e:
                    print(f"\nE: Training batch {i} failed: {e}")
                    # Stop training on OOM error
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        print("E: CUDA Out of Memory during training. Exiting.")
                        raise e
                    # Continue on other errors if desired, or raise
                    # raise e # Uncomment to stop on any training error

            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            print(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.5f}")

            # --- Validation Phase ---
            if dlvalid is None:
                print("W: Skipping validation phase - no validation DataLoader.")
                history.append(
                    {"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": None}
                )
                continue  # Skip to next epoch

            model.eval()
            val_losses = []
            pbar_val = tqdm(dlvalid, desc=f"Valid E{epoch}", leave=False, unit="batch")
            with torch.no_grad():
                for i, batch in enumerate(pbar_val):
                    if not batch or "seis" not in batch or "vel" not in batch:
                        print(f"W: Skipping invalid validation batch {i}")
                        continue
                    try:
                        inputs = batch["seis"].to(cfg.device, non_blocking=True).float()
                        targets = batch["vel"].to(cfg.device, non_blocking=True).float()
                        with torch.amp.autocast(
                            device_type=cfg.device.type,
                            dtype=cfg.autocast_dtype,
                            enabled=cfg.use_cuda,
                        ):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        val_losses.append(loss.item())

                        # Plotting validation examples periodically
                        if i == 0 and epoch % cfg.plot_every_n_epochs == 0:
                            # Add validation plotting code here if desired
                            pass # Placeholder

                    except Exception as e:
                        print(f"\nE: Validation batch {i} failed: {e}")
                        if isinstance(e, torch.cuda.OutOfMemoryError):
                            print("E: CUDA Out of Memory during validation. Exiting.")
                            raise e
                        # Continue on other errors if desired, or raise
                        # raise e # Uncomment to stop on any validation error

            avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
            print(f"Epoch {epoch} Avg Valid Loss: {avg_val_loss:.5f}")
            history.append(
                {"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": avg_val_loss}
            )

            # --- Save Best Model ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Clean previous best models before saving new one
                del_pattern = os.path.join(
                    cfg.working_dir, f"unet_best_model_epoch_*_loss_*.pth"
                )
                for old_model_path in glob.glob(del_pattern):
                    try:
                        print(f"    Removing old best model: {os.path.basename(old_model_path)}")
                        os.remove(old_model_path)
                    except OSError as e:
                        print(f"W: Could not delete old model {old_model_path}: {e}")

                # Save the new best model
                fname = f"unet_best_model_epoch_{epoch}_loss_{best_val_loss:.4f}.pth"
                fpath = os.path.join(cfg.working_dir, fname)
                print(f"*** New best validation loss: {best_val_loss:.5f}. Saving model: {fname} ***")
                torch.save(model.state_dict(), fpath)

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\nE: Training loop encountered a critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Training Loop Finished ---")


# --- 5. Plot History ---
print("\n--- 5. Plotting Training History ---")
if history:
    try:
        hist_df = pd.DataFrame(history)
        plt.figure(figsize=(12, 6))
        plt.plot(hist_df["epoch"], hist_df["train_loss"], "o-", label="Train Loss")
        # Only plot validation loss if it exists and is not all None/NaN
        if "valid_loss" in hist_df.columns and not hist_df["valid_loss"].isnull().all():
            plt.plot(
                hist_df["epoch"],
                hist_df["valid_loss"],
                "s--",  # Square markers, dashed line
                label="Validation Loss",
            )
        plt.title("Training and Validation Loss vs. Epoch (Residual U-Net)")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss (Mean Absolute Error)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(bottom=0)  # Loss should not be negative
        plt.tight_layout()
        plot_fname = os.path.join(cfg.working_dir, "training_history.png")
        plt.savefig(plot_fname)
        print(f"Saved history plot: {plot_fname}")
        plt.show()  # Display the plot
    except Exception as e:
        print(f"E: Failed plotting training history: {e}")
else:
    print("No training history recorded to plot.")


# --- 6. Error Analysis on Validation Set ---
print("\n--- 6. Error Analysis on Validation Set ---")
# Placeholder for error analysis code - requires dlvalid or val_paths_saved
best_model_path_analysis = find_best_model()
if not best_model_path_analysis:
    print("W: No best model found. Skipping analysis.")
elif dlvalid is None and not val_paths_saved:
    # Need either the original loader or the paths to recreate it
    print("W: Validation loader/paths unavailable. Skipping analysis.")
else:
    print(f"Performing analysis using model: {os.path.basename(best_model_path_analysis)}")
    # Add analysis code block here (e.g., load model, get batch, predict, plot errors)
    # Ensure to handle potential recreation of dlvalid if it was lost
    # Example: Load model, get a batch from dlvalid, predict, compare pred/target
    # model_analysis = UNet().to(cfg.device)
    # model_analysis.load_state_dict(torch.load(best_model_path_analysis, map_location=cfg.device))
    # model_analysis.eval()
    # with torch.no_grad():
    #     # Get a batch (handle if dlvalid needs recreation from val_paths_saved)
    #     # batch = next(iter(dlvalid_or_recreated))
    #     # inputs = batch["seis"].to(cfg.device)... targets = batch["vel"]...
    #     # preds = model_analysis(inputs)
    #     # Plot difference, calculate stats, etc.
    pass


# --- 7. Prediction (Using Kaggle Test Set) ---
print("\n--- 7. Final Prediction on Kaggle Test Set ---")
best_model_final_path = find_best_model()
if not best_model_final_path:
    print("W: No best model found. Skipping final prediction.")
elif not Path(cfg.kaggle_test_dir).is_dir():
    print(f"W: Kaggle test directory '{cfg.kaggle_test_dir}' not found. Skipping prediction.")
else:
    try:
        print(f"Loading model for final prediction: {os.path.basename(best_model_final_path)}")
        # Make sure to instantiate the correct model class (UNet)
        model_pred = UNet().to(cfg.device)
        model_pred.load_state_dict(torch.load(best_model_final_path, map_location=cfg.device))
        model_pred.eval()

        test_ds = KaggleTestDataset(cfg.kaggle_test_dir)
        if len(test_ds) == 0:
            print("W: Kaggle test dataset is empty. No submission generated.")
        else:
            # Setup DataLoader for test set
            t_bs = max(1, cfg.batch_size // 2)
            t_nw = min(
                max(0, cfg.num_workers // 2),
                (os.cpu_count() // 2 if os.cpu_count() else 1),
            )
            dl_test = DataLoader(
                test_ds,
                batch_size=t_bs,
                shuffle=False,
                num_workers=t_nw,
                pin_memory=cfg.use_cuda,
            )
            print(f"Test DataLoader created with bs={t_bs}, workers={t_nw}")
            print(f"Writing submission file to: {cfg.submission_file}")

            rows_written = 0
            with open(cfg.submission_file, "wt", newline="") as csvfile:
                # Define CSV header columns (x_1, x_3, ..., x_69)
                x_cols = [f"x_{i}" for i in range(1, 70, 2)]
                fieldnames = ["oid_ypos"] + x_cols
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                pbar_test = tqdm(dl_test, desc="Generating Submission", unit="batch")
                with torch.no_grad():
                    for inputs, original_ids in pbar_test:
                        # Handle batch size = 1 where original_ids might be a string
                        if isinstance(original_ids, str):
                            original_ids = [original_ids]
                        try:
                            inputs = inputs.to(cfg.device).float()
                            with torch.amp.autocast(
                                device_type=cfg.device.type,
                                dtype=cfg.autocast_dtype,
                                enabled=cfg.use_cuda,
                            ):
                                outputs = model_pred(inputs)
                            # Output shape is (B, 1, H, W), get predictions (B, H, W)
                            preds = outputs[:, 0].cpu().numpy()

                            # Iterate through samples in the batch
                            for y_pred, oid in zip(preds, original_ids): # y_pred is (H, W)
                                # Iterate through y-positions (rows) for this sample
                                for y_pos in range(y_pred.shape[0]): # y_pred.shape[0] should be 70
                                    # Extract values at odd x-indices (1, 3, ..., 69)
                                    vals = y_pred[y_pos, 1::2].astype(np.float32)
                                    # Create row dictionary
                                    row = dict(zip(x_cols, vals))
                                    row["oid_ypos"] = f"{oid}_y_{y_pos}"
                                    writer.writerow(row)
                                    rows_written += 1
                        except Exception as e:
                            # Report error but continue if possible
                            print(
                                f"\nE: Prediction failed for batch (OID: {original_ids[0] if original_ids else '?'}) : {e}"
                            )

            print(f"Submission file created: {cfg.submission_file} ({rows_written} rows).")
            # Sanity check row count
            expected_rows = len(test_ds) * 70  # 70 y-positions per test sample
            if rows_written != expected_rows:
                print(
                    f"W: Row count mismatch! Expected {expected_rows}, but wrote {rows_written}."
                )

    except Exception as e:
        print(f"E: Final prediction process failed critically: {e}")
        import traceback
        traceback.print_exc()

print("\n--- Full Workflow Finished ---")