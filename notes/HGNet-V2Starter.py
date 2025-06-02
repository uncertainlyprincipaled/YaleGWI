# HGNet-V2 Starter Notebook
RUN_TRAIN = True
RUN_VALID = True
RUN_TEST  = True

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Requires >= 2 GPUs with CUDA enabled.")

try: 
    import monai
except: 
    !pip install --no-deps monai -q


# This notebook builds on Egor Trushin's great starter notebook [here](https://www.kaggle.com/code/egortrushin/gwi-unet-with-float16-dataset), thanks for sharing. 
# The main purpose of this notebook is to show how to use 2 GPUs during model training, maximizing our weekly GPU quota in the Kaggle environment. 
# In addition, I provide 3x pretrained model checkpoints that were trained for 150 epochs using this setup. Each model achieved a validation MAE of ~60.

# Other additions:                                                                 
# - Flip augmentation
# - Dataset preprocessing
# - EMA (Exponential moving average)
# - Pretrained encoder
# - Monai usample blocks

# Config

from types import SimpleNamespace
import torch

cfg= SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.data_dir = "/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72/"
cfg.local_rank = 0
cfg.seed = 123
cfg.subsample = None

cfg.backbone = "hgnetv2_b2.ssld_stage2_ft_in1k"
cfg.ema = True
cfg.ema_decay = 0.99

cfg.epochs = 10
cfg.batch_size = 512
cfg.batch_size_val = 128

cfg.early_stopping = {"patience": 3, "streak": 0}
cfg.logging_steps = 100

### Preprocess

# Here we reduce the size of each input to (72,72), and then save as fp16. This reduces the size of each input to roughly 4% of the original size.
# This has already been done for every datapoint in the OpenFWI dataset and can be found [here](https://www.kaggle.com/datasets/brendanartley/openfwi-preprocessed-72x72).

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def _preprocess(x):
    x = F.interpolate(x, size=(70, 70), mode='area')
    x = F.pad(x, (1,1,1,1), mode='replicate')
    return x

def _helper(x, ):
    before_shape = x.shape
    before_mem = x.nbytes / 1e6
    x = torch.from_numpy(x).float()

    # Interpolate and pad
    x = _preprocess(x)
    x = x.cpu().numpy().astype(np.float16)

    after_mem = x.nbytes / 1e6
    percent = 100 - 100 * (before_mem - after_mem) / before_mem if before_mem else 0

    # Log
    print("Shape Change")
    print("  {} -> {}".format(before_shape, x.shape))
    print()
    print("Memory Usage")
    print("  {:.1f} MB -> {:.1f} MB".format(before_mem, after_mem))
    print("  ({:.1f}% of original size)".format(percent))
    return x


# New Cell
# Preprocess
x= np.load("/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy")
x = _helper(x)

# Sanity check: Confirm preprocessing matches w/ Dataset
z= np.load("/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72/CurveFault_A/seis2_1_0.npy")
assert np.all(z == x)

del x, z

# Dataset¶
# Here, we introduce a flip augmentation.

# Unlike a normal horizontal flip, we have to reverse the source and receiver dimensions. To match this, we reverse the width dimension of the label as well.

# We use this flip as TTA (test-time augmentation) during inference.


import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        cfg,
        mode = "train", 
    ):
        self.cfg = cfg
        self.mode = mode
        
        self.data, self.labels, self.records = self.load_metadata()

    def load_metadata(self, ):

        # Select rows
        df= pd.read_csv("/kaggle/input/openfwi-preprocessed-72x72/folds.csv")
        if self.cfg.subsample is not None:
            df= df.groupby(["dataset", "fold"]).head(self.cfg.subsample)

        if self.mode == "train":
            df= df[df["fold"] != 0]
        else:
            df= df[df["fold"] == 0]

        
        data = []
        labels = []
        records = []
        mmap_mode = "r" if self.mode == "train" else None

        for idx, row in tqdm(df.iterrows(), total=len(df), disable=self.cfg.local_rank != 0):
            row= row.to_dict()

            # Load
            farr= os.path.join(self.cfg.data_dir, row["data_fpath"])
            flbl= os.path.join(self.cfg.data_dir, row["label_fpath"])
            arr= np.load(farr, mmap_mode=mmap_mode)
            lbl= np.load(flbl, mmap_mode=mmap_mode)

            # Append
            data.append(arr)
            labels.append(lbl)
            records.append(row["dataset"])

        return data, labels, records

    def __getitem__(self, idx):
        row_idx= idx // 500
        col_idx= idx % 500

        d= self.records[row_idx]
        x= self.data[row_idx][col_idx, ...]
        y= self.labels[row_idx][col_idx, ...]

        # Augs 
        if self.mode == "train":
            
            # Temporal flip
            if np.random.random() < 0.5:
                x= x[::-1, :, ::-1]
                y= y[..., ::-1]

        x= x.copy()
        y= y.copy()
        
        return x, y

    def __len__(self, ):
        return len(self.records) * 500
    

# Model

This model includes several modifications beyond a standard U-Net architecture.


### Encoder

The model uses the `HgnetV2` backbone from timm as the encoder. We have to make a few modifications for this to work with the Unet. See more info on this backbone [here](https://huggingface.co/timm/hgnetv2_b2.ssld_stage1_in22k_in1k).


First, we reduce the stride of the stem convolution from (2,2) to (1,1). This increases the size of the feature maps in the backbone. Second, we reduce the stride of the downsample convolution in the deepest block from (2,2) to (1,1). We do this so that upsampling in the decoder can be done without padding.

```python
# Original feature map
[torch.Size([18, 18]), torch.Size([9, 9]), torch.Size([5, 5]), torch.Size([3, 3])]

# Updated stem conv
[torch.Size([36, 36]), torch.Size([18, 18]), torch.Size([9, 9]), torch.Size([5, 5])]

# Updated downsample conv
[torch.Size([36, 36]), torch.Size([18, 18]), torch.Size([9, 9]), torch.Size([9, 9])]
```

### Decoder

# The decoder has a few modifications as well. 

# We remove all BatchNorm2d layers and add intermediate convolutions to the skip connections. I found that removing the normalization layers increased the convergence speed, and the intermediate convolutions improved the model's predictiveness.

# ---

# ### EMA

# We also add an EMA (exponential moving average) class. This is a common strategy used to increase the stability of validation performance between steps/epochs. 

# This implementation is from Tereka [here](https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/475080#2641635).

%%writefile _model.py

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from monai.networks.blocks import UpSample, SubpixelUpsample

####################
## EMA + Ensemble ##
####################

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models).eval()

    def forward(self, x):
        output = None
        
        for m in self.models:
            logits= m(x)
            
            if output is None:
                output = logits
            else:
                output += logits
                
        output /= len(self.models)
        return output
        

###################
## HGNet-V2 Unet ##
###################

class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.conv= nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        self.act= act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1), 
            nn.Sigmoid(),
            )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()

        # Upsample block
        if upsample_mode == "pixelshuffle":
            self.upsample= SubpixelUpsample(
                spatial_dims= 2,
                in_channels= in_channels,
                scale_factor= scale_factor,
            )
        else:
            self.upsample = UpSample(
                spatial_dims= 2,
                in_channels= in_channels,
                out_channels= in_channels,
                scale_factor= scale_factor,
                mode= upsample_mode,
            )

        if intermediate_conv:
            k= 3
            c= skip_channels if skip_channels != 0 else in_channels
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(c, c, k, k//2),
                ConvBnAct2d(c, c, k, k//2),
                )
        else:
            self.intermediate_conv= None

        self.attention1 = Attention2d(
            name= attention_type, 
            in_channels= in_channels + skip_channels,
            )

        self.conv1 = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )
        self.attention2 = Attention2d(
            name= attention_type, 
            in_channels= out_channels,
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                skip = self.intermediate_conv(skip)
            else:
                x = self.intermediate_conv(x)

        if skip is not None:
            # print(x.shape, skip.shape)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder2d(nn.Module):
    """
    Unet decoder.
    Source: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple = (256, 128, 64, 32),
        scale_factors: tuple = (1,2,2,2),
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = True,
        upsample_mode: str = "deconv",
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels= decoder_channels[1:]
        self.decoder_channels= decoder_channels
        
        if skip_channels is None:
            skip_channels= list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()

        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            # print(i, ic, sc, dc)
            self.blocks.append(
                DecoderBlock2d(
                    ic, sc, dc, 
                    norm_layer= norm_layer,
                    attention_type= attention_type,
                    intermediate_conv= intermediate_conv,
                    upsample_mode= upsample_mode,
                    scale_factor= scale_factors[i],
                    )
            )

    def forward(self, feats: list[torch.Tensor]):
        res= [feats[0]]
        feats= feats[1:]

        # Decoder blocks
        for i, b in enumerate(self.blocks):
            skip= feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
                )
            
        return res

class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv= nn.Conv2d(
            in_channels, out_channels, kernel_size= kernel_size,
            padding= kernel_size//2
        )
        self.upsample = UpSample(
            spatial_dims= 2,
            in_channels= out_channels,
            out_channels= out_channels,
            scale_factor= scale_factor,
            mode= mode,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x

class Net(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
    ):
        super().__init__()
        
        # Encoder
        self.backbone= timm.create_model(
            backbone,
            in_chans= 5,
            pretrained= pretrained,
            features_only= True,
            drop_path_rate=0.4,
            )
        ecs= [_["num_chs"] for _ in self.backbone.feature_info][::-1]

        # Decoder
        self.decoder= UnetDecoder2d(
            encoder_channels= ecs,
        )

        self.seg_head= SegmentationHead2d(
            in_channels= self.decoder.decoder_channels[-1],
            out_channels= 1,
            scale_factor= 2,
        )
        self._update_stem(backbone)

    def _update_stem(self, backbone):
        if backbone.startswith("hgnet"):
            self.backbone.stem.stem1.conv.stride=(1,1)
            self.backbone.stages_3.downsample.conv.stride=(1,1)
        
        elif backbone in ["resnet18"]:
            self.backbone.layer4[0].downsample[0].stride= (1,1)
            self.backbone.layer4[0].conv1.stride= (1,1)
            self.backbone.layer3[0].downsample[0].stride= (1,1)
            self.backbone.layer3[0].conv1.stride= (1,1)

        else:
            raise ValueError("Custom striding not implemented.")
        pass

        
    def proc_flip(self, x_in):
        x_in= torch.flip(x_in, dims=[-3, -1])
        x= self.backbone(x_in)
        x= x[::-1]

        # Decoder
        x= self.decoder(x)
        x_seg= self.seg_head(x[-1])
        x_seg= x_seg[..., 1:-1, 1:-1]
        x_seg= torch.flip(x_seg, dims=[-1])
        x_seg= x_seg * 1500 + 3000
        return x_seg

    def forward(self, batch):
        x= batch

        # Encoder
        x_in = x
        x= self.backbone(x)
        # print([_.shape for _ in x])
        x= x[::-1]

        # Decoder
        x= self.decoder(x)
        # print([_.shape for _ in x])
        x_seg= self.seg_head(x[-1])
        x_seg= x_seg[..., 1:-1, 1:-1]
        x_seg= x_seg * 1500 + 3000
    
        if self.training:
            return x_seg
        else:
            p1 = self.proc_flip(x_in)
            x_seg = torch.mean(torch.stack([x_seg, p1]), dim=0)
            return x_seg
        
# Utils
# Same as Egor's.


import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Train
# Here is the main training script.

# By using 2x GPUs, we can use larger batch sizes and speed up model training. No more wasted Quota!

# I won't go into the details of the script as there are already many good resources explaining DDP. Here are a couple of good starting points.

# Run DDP scripts with 2 T 4 by @CPMP
# Getting Started with Distributed Data Parallel
# Distributed Data Parallel Docs

%%writefile _train.py

import os
import time 
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from _cfg import cfg
from _dataset import CustomDataset
from _model import ModelEMA, Net
from _utils import format_time

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return

def main(cfg):

    # ========== Datasets / Dataloaders ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Loading data..")
    train_ds = CustomDataset(cfg=cfg, mode="train")
    sampler= DistributedSampler(
        train_ds, 
        num_replicas=cfg.world_size, 
        rank=cfg.local_rank,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        sampler= sampler,
        batch_size= cfg.batch_size, 
        num_workers= 4,
    )
    
    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    sampler= DistributedSampler(
        valid_ds, 
        num_replicas=cfg.world_size, 
        rank=cfg.local_rank,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, 
        sampler= sampler,
        batch_size= cfg.batch_size_val, 
        num_workers= 4,
    )

    # ========== Model / Optim ==========
    model = Net(backbone=cfg.backbone)
    model= model.to(cfg.local_rank)
    if cfg.ema:
        if cfg.local_rank == 0:
            print("Initializing EMA model..")
        ema_model = ModelEMA(
            model, 
            decay=cfg.ema_decay, 
            device=cfg.local_rank,
        )
    else:
        ema_model = None
    model= DistributedDataParallel(
        model, 
        device_ids=[cfg.local_rank], 
        )
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()


    # ========== Training ==========
    if cfg.local_rank == 0:
        print("="*25)
        print("Give me warp {}, Mr. Sulu.".format(cfg.world_size))
        print("="*25)
    
    best_loss= 1_000_000
    val_loss= 1_000_000

    for epoch in range(0, cfg.epochs+1):
        if epoch != 0:
            tstart= time.time()
            train_dl.sampler.set_epoch(epoch)
    
            # Train loop
            model.train()
            total_loss = []
            for i, (x, y) in enumerate(train_dl):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
        
                with autocast(cfg.device.type):
                    logits = model(x)
                    
                loss = criterion(logits, y)
        
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
        
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
                total_loss.append(loss.item())
                
                if ema_model is not None:
                    ema_model.update(model)
                    
                if cfg.local_rank == 0 and (len(total_loss) >= cfg.logging_steps or i == 0):
                    train_loss = np.mean(total_loss)
                    total_loss = []
                    print("Epoch {}:     Train MAE: {:.2f}     Val MAE: {:.2f}     Time: {}     Step: {}/{}".format(
                        epoch, 
                        train_loss,
                        val_loss,
                        format_time(time.time() - tstart),
                        i+1, 
                        len(train_dl)+1, 
                    ))
    
        # ========== Valid ==========
        model.eval()
        val_logits = []
        val_targets = []
        with torch.no_grad():
            for x, y in tqdm(valid_dl, disable=cfg.local_rank != 0):
                x = x.to(cfg.local_rank)
                y = y.to(cfg.local_rank)
    
                with autocast(cfg.device.type):
                    if ema_model is not None:
                        out = ema_model.module(x)
                    else:
                        out = model(x)

                val_logits.append(out.cpu())
                val_targets.append(y.cpu())

            val_logits= torch.cat(val_logits, dim=0)
            val_targets= torch.cat(val_targets, dim=0)
                
            loss = criterion(val_logits, val_targets).item()

        # Gather loss
        v = torch.tensor([loss], device=cfg.local_rank)
        torch.distributed.all_reduce(v, op=dist.ReduceOp.SUM)
        val_loss = (v[0] / cfg.world_size).item()
    
        # ========== Weights / Early stopping ==========
        stop_train = torch.tensor([0], device=cfg.local_rank)
        if cfg.local_rank == 0:
            es= cfg.early_stopping
            if val_loss < best_loss:
                print("New best: {:.2f} -> {:.2f}".format(best_loss, val_loss))
                print("Saved weights..")
                best_loss = val_loss
                if ema_model is not None:
                    torch.save(ema_model.module.state_dict(), f'best_model_{cfg.seed}.pt')
                else:
                    torch.save(model.state_dict(), f'best_model_{cfg.seed}.pt')
        
                es["streak"] = 0
            else:
                es= cfg.early_stopping
                es["streak"] += 1
                if es["streak"] > es["patience"]:
                    print("Ending training (early_stopping).")
                    stop_train = torch.tensor([1], device=cfg.local_rank)
        
        # Exits training on all ranks
        dist.broadcast(stop_train, src=0)
        if stop_train.item() == 1:
            return

    return
    


if __name__ == "__main__":

    # GPU Specs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _, total = torch.cuda.mem_get_info(device=rank)

    # Init
    setup(rank, world_size)
    time.sleep(rank)
    print(f"Rank: {rank}, World size: {world_size}, GPU memory: {total / 1024**3:.2f}GB", flush=True)
    time.sleep(world_size - rank)

    # Seed
    set_seed(cfg.seed+rank)

    # Run
    cfg.local_rank= rank
    cfg.world_size= world_size
    main(cfg)
    cleanup()

if RUN_TRAIN:
    print("Starting training..")
    !OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 _train.py

# Pretrained Models
# Next, we load in 3x pretrained models. These models were trained with with an effective batch_size of 512 (256 per GPU) and use the B4 variant of the HgnetV2 backbone.

import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from _cfg import cfg
from _model import Net, EnsembleModel

if RUN_VALID or RUN_TEST:

    # Load pretrained models
    models = []
    for f in sorted(glob.glob("/kaggle/input/openfwi-preprocessed-72x72/models/*.pt")):
        print("Loading: ", f)
        m = Net(
            backbone="hgnetv2_b4.ssld_stage2_ft_in1k",
            pretrained=False,
        )
        state_dict= torch.load(f, map_location=cfg.device, weights_only=True)
        m.load_state_dict(state_dict)
        models.append(m)
    
    # Combine
    model = EnsembleModel(models)
    model = model.to(cfg.device)
    model = model.eval()
    print("n_models: {:_}".format(len(models)))


# Valid
# Next, we score the ensemble on the validation set.

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast

from _dataset import CustomDataset


if RUN_VALID:

    # Dataset / Dataloader
    valid_ds = CustomDataset(cfg=cfg, mode="valid")
    sampler = torch.utils.data.SequentialSampler(valid_ds)
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, 
        sampler= sampler,
        batch_size= cfg.batch_size_val, 
        num_workers= 4,
    )

    # Valid loop
    criterion = nn.L1Loss()
    val_logits = []
    val_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(valid_dl):
            x = x.to(cfg.device)
            y = y.to(cfg.device)
    
            with autocast(cfg.device.type):
                out = model(x)
    
            val_logits.append(out.cpu())
            val_targets.append(y.cpu())
    
        val_logits= torch.cat(val_logits, dim=0)
        val_targets= torch.cat(val_targets, dim=0)
    
        total_loss= criterion(val_logits, val_targets).item()
    
    # Dataset Scores
    ds_idxs= np.array([valid_ds.records])
    ds_idxs= np.repeat(ds_idxs, repeats=500)
    
    print("="*25)
    with torch.no_grad():    
        for idx in sorted(np.unique(ds_idxs)):
    
            # Mask
            mask = ds_idxs == idx
            logits_ds = val_logits[mask]
            targets_ds = val_targets[mask]
    
            # Score predictions
            loss = criterion(val_logits[mask], val_targets[mask]).item()
            print("{:15} {:.2f}".format(idx, loss))
    print("="*25)
    print("Val MAE: {:.2f}".format(total_loss))
    print("="*25)

# Test
# Finally, we make predictions on the test data.

import torch

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_files):
        self.test_files = test_files

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, i):
        test_file = self.test_files[i]
        test_stem = test_file.split("/")[-1].split(".")[0]
        return np.load(test_file), test_stem

import csv
import time
import glob
from tqdm import tqdm
import numpy as np

from _utils import format_time


if RUN_TEST:
    row_count = 0
    t0 = time.time()
    
    test_files = sorted(glob.glob("/kaggle/input/open-wfi-test/test/*.npy"))
    x_cols = [f"x_{i}" for i in range(1, 70, 2)]
    fieldnames = ["oid_ypos"] + x_cols
    
    test_ds = TestDataset(test_files)
    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        sampler=torch.utils.data.SequentialSampler(test_ds),
        batch_size=cfg.batch_size_val, 
        num_workers=4,
    )
    
    with open("submission.csv", "wt", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with torch.inference_mode():
            with torch.autocast(cfg.device.type):
                for inputs, oids_test in tqdm(test_dl, total=len(test_dl)):
                    inputs = inputs.to(cfg.device)
            
                    inputs = _preprocess(inputs)
                    outputs = model(inputs)
                            
                    y_preds = outputs[:, 0].cpu().numpy()
                    
                    for y_pred, oid_test in zip(y_preds, oids_test):
                        for y_pos in range(70):
                            row = dict(zip(x_cols, [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]))
                            row["oid_ypos"] = f"{oid_test}_y_{y_pos}"
            
                            writer.writerow(row)
                            row_count += 1

                            # Clear buffer
                            if row_count % 100_000 == 0:
                                csvfile.flush()
    
    t1 = format_time(time.time() - t0)
    print(f"Inference Time: {t1}")

# We can also view a few samples to make sure things look reasonable.

import matplotlib.pyplot as plt 

if RUN_TEST:
    # Plot a few samples
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    axes= axes.flatten()

    n = min(len(outputs), len(axes))
    
    for i in range(n):
        img= outputs[0, 0, ...].cpu().numpy()
        img = outputs[i, 0].cpu().numpy()
        idx= oids_test[i]
    
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(idx)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()