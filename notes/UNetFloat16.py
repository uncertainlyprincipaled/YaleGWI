# Copy of Gold Level Notebook
# https://www.kaggle.com/code/jdmorgan/gwi-unet-with-float16-dataset/edit

"""
[GWI] Improved UNet pipepline with larger dataset
Initial version
We used UNet model as introduced in 5 depth U net with residual Notebook.
We used part of Full OpenWFI dataset, which was introduced in OpenFWI InversionNet Train with 670G Datasets Notebook.
Present version
We will use openFWI dataset converted from float32 to float16 (openfwi_float16_1, openfwi_float16_2, and openfwi_float16_test). 
Since reading of data from disk is a bottleneck here, the conversion allows us to read data from disk twice faster and to use twice 
more data in training for the same runtime.
"""

# Config Yaml
# %%writefile config.yaml

# data_path: /kaggle/input/waveform-inversion
# model: 
#     name: UNet
#     unet_params:
#         init_features: 32
#         depth: 5
# read_weights: null
# batch_size: 64
# print_freq: 1000
# max_epochs: 15
# es_epochs: 4
# seed: 42
# valid_frac: 16
# train_frac: 2
# optimizer:
#     lr: 0.0001
#     weight_decay: 0.001
# scheduler:
#     params:
#         factor: 0.316227766
#         patience: 1

# Data Processing
# Data

import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]


def get_train_files(data_path):

    all_inputs = [
        f
        for f in
        Path(data_path).rglob('*.npy')
        if ('seis' in f.stem) or ('data' in f.stem)
    ]

    all_outputs = inputs_files_to_output_files(all_inputs)

    assert all(f.exists() for f in all_outputs)

    return all_inputs, all_outputs


class SeismicDataset(Dataset):
    def __init__(self, inputs_files, output_files, n_examples_per_file=500):
        assert len(inputs_files) == len(output_files)
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        # Calculate file offset and sample offset within file
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        X = np.load(self.inputs_files[file_idx], mmap_mode='r')
        y = np.load(self.output_files[file_idx], mmap_mode='r')

        try:
            return X[sample_idx].copy(), y[sample_idx].copy()
        finally:
            del X, y


class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files


    def __len__(self):
        return len(self.test_files)


    def __getitem__(self, i):
        test_file = self.test_files[i]

        return np.load(test_file), test_file.stem
    
# Model
# Model

import torch.nn as nn
import torch.nn.functional as F
import torch

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
        n_channels=5,
        n_classes=1,
        init_features=32,
        depth=5, # number of pooling layers
        bilinear=True,
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
    
# Utils

import datetime
import random
import torch
import numpy as np

def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def seed_everything(
    seed_value: int
) -> None:
    """
    Controlling a unified seed value for Python, NumPy, and PyTorch (CPU, GPU).

    Parameters:
    ----------
    seed_value : int
        The unified random seed value.
    """
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Preparation and training

import os
import sys
import yaml
from torch.utils.data import DataLoader
from pprint import pprint
import torch
import torch.nn as nn
import numpy as np
import time

print(f"GPU: {torch.cuda.get_device_name(0)}")
_, total = torch.cuda.mem_get_info(device=0)
print(f"GPU memory: {total / 1024**3:.2f}GB")

with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)
print()
pprint(config)
if config["data_path"] is None:
    config["data_path"] = os.environ["TMPDIR"]
    print("data_path:", config["data_path"])
print()

seed_everything(config["seed"])

all_inputs, all_outputs = [], []
for x in ["/kaggle/input/open-wfi-1/openfwi_float16_1", "/kaggle/input/open-wfi-2/openfwi_float16_2"]:
    all_inputs1, all_outputs1 = get_train_files(x)
    all_inputs.extend(all_inputs1)
    all_outputs.extend(all_outputs1)
print("Total number of input/output files:", len(all_inputs))

valid_inputs = [all_inputs[i] for i in range(0, len(all_inputs), config["valid_frac"])]
train_inputs = [f for f in all_inputs if not f in valid_inputs]
if config["train_frac"] > 1:
    train_inputs = [train_inputs[i] for i in range(0, len(train_inputs), config["train_frac"])]

print("Number of train files:", len(train_inputs))
print("Number of valid files:", len(valid_inputs))
print()

train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)

dstrain = SeismicDataset(train_inputs, train_outputs)
dltrain = DataLoader(
    dstrain,
    batch_size=config["batch_size"],
    shuffle=True,
    pin_memory=False,
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
)

dsvalid = SeismicDataset(valid_inputs, valid_outputs)
dlvalid = DataLoader(
    dsvalid,
    batch_size=4*config["batch_size"],
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(**config["model"]["unet_params"]).to(device)

if config["read_weights"] is not None:
    print("Reading weights from:", config["read_weights"])
    model.load_state_dict(torch.load(config["read_weights"], weights_only=True))

criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])  # hparams
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **config["scheduler"]["params"])

best_val_loss = 10000.0
epochs_wo_improvement = 0
t0 = time.time()  # Measure staring time

for epoch in range(1, config["max_epochs"] + 1):

    # Train
    model.train()
    train_losses = []
    for step, (inputs, targets) in enumerate(dltrain):

        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if step % config["print_freq"] == config["print_freq"] - 1 or step == len(dltrain) - 1:
            trn_loss = np.mean(train_losses)
            t1 = format_time(time.time() - t0)
            free, total = torch.cuda.mem_get_info(device=0)
            mem_used = (total - free) / 1024**3
            lr = optimizer.param_groups[-1]['lr']
            print(
                f"Epoch: {epoch:02d}  Step {step+1}/{len(dltrain)}  Trn Loss: {trn_loss:.2f}  LR: {lr:.2e}  GPU Usage: {mem_used:.2f}GB  Elapsed Time: {t1}",
                flush=True,
            )

    # Valid
    model.eval()
    valid_losses = []
    for inputs, targets in dlvalid:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)

        loss = criterion(outputs, targets)

        valid_losses.append(loss.item())

    t1 = format_time(time.time() - t0)
    trn_loss = np.mean(train_losses)
    val_loss = np.mean(valid_losses)

    free, total = torch.cuda.mem_get_info(device=0)
    mem_used = (total - free) / 1024**3

    print(
        f"\nEpoch: {epoch:02d}  Trn Loss: {trn_loss:.2f}  Val Loss: {val_loss:.2f}  GPU Usage: {mem_used:.2f}GB  Elapsed Time: {t1}",
        flush=True,
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_wo_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"\nNew best val_loss: {val_loss:.2f}\n", flush=True)
    else:
        epochs_wo_improvement += 1
        print(f"\nEpochs without improvement: {epochs_wo_improvement}\n", flush=True)

    if epochs_wo_improvement == config["es_epochs"]:
        break

    scheduler.step(val_loss)

# Inference

import csv
from pathlib import Path
import os

t0 = time.time()

test_files = list(Path("/kaggle/input/open-wfi-test/test").glob("*.npy"))
x_cols = [f"x_{i}" for i in range(1, 70, 2)]
fieldnames = ["oid_ypos"] + x_cols
ds = TestDataset(test_files)
dl = DataLoader(ds, batch_size=4*config["batch_size"], num_workers=4, pin_memory=False)

model.load_state_dict(torch.load("best_model.pth", weights_only=True))

model.eval()
with open("submission.csv", "wt", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for inputs, oids_test in dl:
        inputs = inputs.to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)

        y_preds = outputs[:, 0].cpu().numpy()

        for y_pred, oid_test in zip(y_preds, oids_test):
            for y_pos in range(70):
                row = dict(zip(x_cols, [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]))
                row["oid_ypos"] = f"{oid_test}_y_{y_pos}"

                writer.writerow(row)

t1 = format_time(time.time() - t0)
print(f"Inference Time: {t1}")