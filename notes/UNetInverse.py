# Github Source: https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI

# | Model Name | Dataset | LB | Local cv | Version |
# | :----: | :----: | :----: | :----: | :----: |
# | UNetInverseModel 33M | Kaggle Dataset | 190 | 186.4 | v1 |

!pip install -q iunets

import transforms as T

import os
import sys
import time
import datetime
import json

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision.transforms import Compose

from iunets import iUNet

from tqdm.notebook import tqdm
from tqdm import tqdm

import iunet_network
import utils

import random
import numpy as np

from pathlib import Path

from torchvision.models import vgg16
import torchvision.models.vgg as vgg

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class CFG:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fil_size = 500 # number of samples in each npy file

    # Path related
    output_path = '/kaggle/working/Invnet_models'
    plot_directory = 'visualisation'
    cfg_path = '/kaggle/input/unet-configs'

    # Model realted
    model = 'UNetInverseModel'
    latent_dim = 70
    skip = 1 # [0, 1] Unet skip connections 0:False, and 1:True
    up_mode = None # upsampling layer mode such as "nearest", "bicubic", etc.
    sample_spatial = 1.0 
    sample_temporal = 1
    optimizer = 'Adam'
    lr_scheduler = 'StepLR'
    unet_depth = 2
    unet_repeat_blocks = 2

    # Training related
    batch_size = 64
    lr = 0.001
    lr_milestones = []
    momentum = 0.9
    weight_decay = 13-4
    lr_gamma = 0.1
    lr_warmup_epochs = 0
    epoch_block = 20
    num_block = 5
    workers = 4
    k = 1
    print_freq = 50
    resume = '/kaggle/input/unet-inverse-training-inference/Invnet_models/model_60.pth'
    start_epoch = 0
    plot_interval = 5
    num_images = 3
    rm_direct_arrival=1 # 'Remove direct arrival from amplitude data.'
    velocity_transform = 'min_max'
    amplitude_transform = 'normalize'
    mask_factor = 0.0

    # Loss realted
    lambda_g1v = 1.0
    lambda_g2v = 1.0
    lambda_vel = 1.0
    lambda_vgg_vel = 0.1
    vgg_layer_output = 2 # VGG16 pretrained model layer output for perceptual loss calculation.
    lambda_reg = 0.1
    lambda_recons = 0.0

args = CFG()

args.epochs = args.epoch_block * args.num_block
args.skip = True if args.skip==1 else False



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1234)

# Util

joint_model_list = [iunet_network.IUnetModel, iunet_network.JointModel, iunet_network.Decouple_IUnetModel]
inverse_model_list = [
                        iunet_network.InversionNet, 
                        iunet_network.IUnetInverseModel, 
                        iunet_network.UNetInverseModel,
                        iunet_network.IUnetInverseModel_Legacy, 
                        iunet_network.UNetInverseModel_Legacy,
                      ]
rainbow_cmap = ListedColormap(np.load('/kaggle/input/unet-configs/rainbow256.npy'))
def get_optimizer(args, model, lr):
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=args.weight_decay) #lr = 1.0 default
    elif args.optimizer == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay) # deafult lr = 0.002
    else:
        print("Optimizer not found")
    return optimizer

def get_lr_scheduler(args, optimizer):
    # LR-Schedulers
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs-args.start_epoch)//3, gamma=0.5)
    elif args.lr_scheduler == "LinearLR":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                         start_factor=1,
                                                         end_factor = 1e-4, 
                                                         total_iters=args.epochs-args.start_epoch)
    elif args.lr_scheduler == "ExponentialLR":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.lr_scheduler == "CosineAnneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.start_epoch)
    elif args.lr_scheduler == "None":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.epochs-args.start_epoch), gamma=1.0) #Fake LR Scheduler
    else:
        print("LR Scheduler not found")
    return lr_scheduler

def get_transforms(args, ctx):
    # label transforms
    if args.velocity_transform == "min_max":
        transform_label = T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    elif args.velocity_transform == "normalize":
        transform_label = T.Normalize(ctx['label_mean'], ctx['label_std'])
    elif args.velocity_transform == "quantile":
        transform_label = T.QuantileTransform(n_quantiles=100)
    else:
        transform_label = None

    # data transforms
    if args.amplitude_transform == "min_max":
        transform_data = T.MinMaxNormalize(ctx['data_min'], ctx['data_max'])
    elif args.amplitude_transform == "normalize":
        transform_data = T.Normalize(ctx['data_mean'], ctx['data_std'])
    elif args.amplitude_transform == "quantile":
        transform_data = T.QuantileTransform(n_quantiles=100)      
    else:
        transform_data=None
    return transform_data, transform_label


def count_parameters(model, verbose=True):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params >= 1e6:
        num_params /= 1e6
        suffix = "M"
    elif num_params >= 1e3:
        num_params /= 1e3
        suffix = "K"
    else:
        suffix = ""
    if verbose:
        print(f"Number of trainable parameters: {num_params:.2f}{suffix}")
    return num_params


class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        vgg_model = vgg.vgg16(pretrained=True)
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x, vgg_layer_output=2):
        assert vgg_layer_output <= len(self.layer_name_mapping)
        
        count = 0
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                if count == vgg_layer_output:
                    return x
                count += 1
        return None

def plot_images(num_images, dataset, model, epoch, vis_folder, device, transform_data, transform_label, plot=True, save_key="results_epoch"):
    items = np.random.choice(len(dataset), num_images)

    # _, amp_true, vel_true = dataset[items]

    samples = [dataset[i] for i in items]

    _, amp_true, vel_true = zip(*samples)
    amp_true, vel_true = torch.tensor(amp_true).to(device), torch.tensor(vel_true).to(device)

    model = model.to(device)

    if np.any([isinstance(model, item) for item in joint_model_list]):
        # if isinstance(model, iunet_network.JointModel) and isinstance(model.forward_model, forward_network.FNO2d):
        #     amp_true = torch.einsum("ijkl->iklj", amp_true)
        #     vel_true = torch.einsum("ijkl->iklj", vel_true)
        
        amp_pred = model.forward(vel_true).detach()
        if transform_data is not None:
            amp_true_np = transform_data.inverse_transform(amp_true.detach().cpu().numpy())
            amp_pred_np = transform_data.inverse_transform(amp_pred.detach().cpu().numpy())

        vel_pred = model.inverse(amp_true).detach()
        if transform_label is not None:
            vel_true_np = transform_label.inverse_transform(vel_true.detach().cpu().numpy())
            vel_pred_np = transform_label.inverse_transform(vel_pred.detach().cpu().numpy())
        
        fig, axes = plt.subplots(num_images, 6, figsize=(20, int(3*num_images)), dpi=150)
        for i in range(num_images):
            vel = np.concatenate([vel_pred_np[i, 0], vel_true_np[i, 0]], axis=1)
            amp = np.concatenate([amp_pred_np[i, 0], amp_true_np[i, 0]], axis=1)
            
            min_vel, max_vel = vel.min(), vel.max()
            
            diff_vel = vel_pred_np[i, 0] - vel_true_np[i, 0]
            diff_amp = amp_pred_np[i, 0] - amp_true_np[i, 0]
            
            ax = axes[i, 0]
            img = ax.imshow(vel_true_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 1]
            img = ax.imshow(vel_pred_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 2]
            img = ax.imshow(diff_vel, aspect='auto', cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 3]
            img = ax.imshow(amp_true_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 4]
            img = ax.imshow(amp_pred_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 5]
            img = ax.imshow(diff_amp, aspect='auto', cmap="seismic", vmin=-1, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

    # plotting for inverse problem only
    elif np.any([isinstance(model, item) for item in inverse_model_list]):
        vel_pred = model(amp_true).detach()
        
        if transform_label is not None:
            vel_true_np = transform_label.inverse_transform(vel_true.detach().cpu().numpy())
            vel_pred_np = transform_label.inverse_transform(vel_pred.detach().cpu().numpy())
        
        fig, axes = plt.subplots(num_images, 3, figsize=(10, int(3*num_images)), dpi=150)
        for i in range(num_images):
            vel = np.concatenate([vel_pred_np[i, 0], vel_true_np[i, 0]], axis=1)
            
            min_vel, max_vel = vel.min(), vel.max()
            
            diff_vel = vel_pred_np[i, 0] - vel_true_np[i, 0]
            
            ax = axes[i, 0]
            img = ax.imshow(vel_true_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 1]
            img = ax.imshow(vel_pred_np[i, 0], aspect='auto', vmin=min_vel, vmax=max_vel, cmap=rainbow_cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 2]
            img = ax.imshow(diff_vel, aspect='auto', cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Velocity Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
     

    # plotting for forward problem only
    elif np.any([isinstance(model, item) for item in forward_model_list]):
#         if isinstance(model, forward_network.FNO2d):
#             amp_true = torch.einsum("ijkl->iklj", amp_true)
#             vel_true = torch.einsum("ijkl->iklj", vel_true)
        amp_pred = model(vel_true).detach()
        if transform_data is not None:
            amp_true_np = transform_data.inverse_transform(amp_true.detach().cpu().numpy())
            amp_pred_np = transform_data.inverse_transform(amp_pred.detach().cpu().numpy())

        fig, axes = plt.subplots(num_images, 3, figsize=(10, int(3*num_images)), dpi=150)
        for i in range(num_images):
            amp = np.concatenate([amp_pred_np[i, 0], amp_true_np[i, 0]], axis=1)
            diff_amp = amp_pred_np[i, 0] - amp_true_np[i, 0]
            
            ax = axes[i, 0]
            img = ax.imshow(amp_true_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform GT {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 1]
            img = ax.imshow(amp_pred_np[i, 0], aspect='auto', vmin=-1, vmax=1, cmap="seismic")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Predicted {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax = axes[i, 2]
            img = ax.imshow(diff_amp, aspect='auto', cmap="seismic", vmin=-1, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(img, cax=cax)
            ax.set_title(f"Waveform Difference {i}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(vis_folder, f"{save_key}_{epoch}.pdf"))
    if plot:
        plt.show()
    plt.close()

# Ctx (What does this mean?)
ctx_dict = {
    "flatvel-a": {
        "data_min": -26.95,
        "data_max": 52.77,
        "data_mean": -3.3804e-05,
        "data_std": 1.4797,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 2782.0442,
        "label_std": 786.1557,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvevel-a": {
        "data_min": -27.11,
        "data_max": 55.10,
        "data_mean": -5.0961e-05,
        "data_std": 1.4870,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "label_mean": 2788.1562,
        "label_std": 794.2432,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatvel-b": {
        "data_min": -27.17,
        "data_max": 56.05,
        "data_mean": -0.0002,
        "data_std": 1.7832,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 3001.3389,
        "label_std": 866.6636,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvevel-b": {
        "data_min": -29.04,
        "data_max": 57.03,
        "data_mean": -0.0002,
        "data_std": 1.7648,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 3000.5669,
        "label_std": 865.4404,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
	"flatfault-a": {
        "data_min": -26.10,
        "data_max": 50.86,
        "data_mean": -0.00043503073,
        "data_std": 1.5410482,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 3088.6873,
        "label_std": 855.37024,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvefault-a": {
        "data_min": -26.48,
        "data_max": 52.32,
        "data_mean": -0.00045603843,
        "data_std": 1.5448948,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 3082.6616,
        "label_std": 852.38995,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatfault-b": {
        "data_min": -24.86,
        "data_max": 50.28,
        "data_mean": -0.0001,
        "data_std": 1.4952,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 500,
        "label_mean": 3055.4231,
        "label_std": 875.8992,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "curvefault-b": {
        "data_min": -24.93,
        "data_max": 50.98,
        "data_mean": -8.882544e-05,
        "data_std": 1.50228,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 3035.5576,
        "label_std": 890.48785,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "style-a": {
        "data_min": -24.96,
        "data_max": 48.93,
        "data_mean": 0.00024991733,
        "data_std": 1.4618,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 2728.5144,
        "label_std": 665.83215,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "style-b": {
        "data_min": -23.76,
        "data_max": 46.01,
        "data_mean": 0.00013498258,
        "data_std": 1.4579,
        "label_min": 1500,
        "label_max": 4500,
        "label_mean": 2837.3164,
        "label_std": 637.6763,
        "file_size": 500,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    },
    "flatvel-tutorial": {
        "data_min": -26.95,
        "data_max": 52.77,
        "label_min": 1500,
        "label_max": 4500,
        "file_size": 120,
        "nbc": 120,
        "dx": 10,
        "nt": 1000,
        "dt": 1e-3,
        "f": 15,
        "n_grid": 70,
        "ns": 5,
        "ng": 70,
        "sz": 10,
        "gz": 10
    }
}

ctx = ctx_dict['flatvel-a']

# WarmupMultiStepLR

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not milestones == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr *
            warmup_factor *
            self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
    
# Data

class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file 
    instead of the batch used during training.

    Args:
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''
    def __init__(self, inputs, outputs, preload=True, sample_ratio=1, file_size=500,
                    transform_data=None, transform_label=None, mask_factor=0.0):
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        if outputs is not None:
            self.batches = [str(inputs[i])+'&'+str(outputs[i]) for i in range(len(inputs))]
        else:
            self.batches = [str(inputs[i]) for i in range(len(inputs))]
            
        if preload:
            self.data_list, self.label_list= (), ()
            for batch in tqdm(self.batches):
                data, label = self.load_every(batch) 

                self.data_list = self.data_list + (data,)
                self.label_list = self.label_list + (label,)

            self.data_list = np.concatenate(self.data_list, 0)
            self.label_list = np.concatenate(self.label_list, 0)

            mask_indices = np.random.choice(len(self.data_list), 
                                              int(mask_factor*len(self.data_list)),
                                              replace=False)
            
            self.mask_list = np.ones(len(self.data_list), dtype=np.int8)
            self.mask_list[mask_indices] = 0

            print("Data concatenation complete.")
            if self.transform_data is not None:
                self.data_list = self.transform_data(self.data_list)
            if self.transform_label is not None:
                self.label_list = self.transform_label(self.label_list)

    # Load from one line
    def load_every(self, batch):
        batch = batch.split('&')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1]
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None
        
        return data, label
        
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            mask = self.mask_list[idx]
            data = self.data_list[idx]
            label = self.label_list[idx] if len(self.label_list) != 0 else None
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
            mask=np.array([1])

            # label = torch.tensor(label, dtype=torch.float32)  
            # data = torch.tensor(data, dtype=torch.float32) 
        
            if self.transform_data is not None:
                data = self.transform_data(data)
            if self.transform_label is not None:
                label = self.transform_label(label)
        return mask, data, label if label is not None else np.array([])
        
    def __len__(self):
        return len(self.batches) * self.file_size
    
# OLD CODE?
    
# class FWIDataset(Dataset):
#     def __init__(self, inputs_files, output_files, n_examples_per_file=500, transform_data=None, transform_label=None):
#         assert len(inputs_files) == len(output_files)
#         self.inputs_files = inputs_files
#         self.output_files = output_files
#         self.n_examples_per_file = n_examples_per_file
#         self.transform_data = transform_data
#         self.transform_label =transform_label

#     def __len__(self):
#         return len(self.inputs_files) * self.n_examples_per_file

#     def __getitem__(self, idx):
#         # Calculate file offset and sample offset within file
#         file_idx = idx // self.n_examples_per_file
#         sample_idx = idx % self.n_examples_per_file

#         X = np.load(self.inputs_files[file_idx], mmap_mode='r')
#         y = np.load(self.output_files[file_idx], mmap_mode='r')

#         try:
#             data = X[sample_idx].copy()
#             label = y[sample_idx].copy()
#             mask = np.array([1])
#             if self.transform_data is not None:
#                 data = self.transform_data(data)
#             if self.transform_label is not None:
#                 label = self.transform_data(label)
#             return mask, data, label
#         finally:
#             del X, y
    
all_inputs = [
    f
    for f in
    Path('/kaggle/input/waveform-inversion/train_samples').rglob('*.npy')
    if ('seis' in f.stem) or ('data' in f.stem)
]

def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]

all_outputs = inputs_files_to_output_files(all_inputs)

train_inputs = [all_inputs[i] for i in range(0, len(all_inputs), 2)] # Sample every two
valid_inputs = [f for f in all_inputs if not f in train_inputs]

train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)

transform_data, transform_label = get_transforms(args, ctx)

print('Loading data')
print('Loading training data')
dataset_train = FWIDataset(
            all_inputs,
            all_outputs,
            preload=True,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_data,
            transform_label=transform_label,
            mask_factor=args.mask_factor
        )

print('Loading validation data')
# dataset_valid = FWIDataset(
#             valid_inputs,
#             valid_outputs,
#             preload=False,
#             sample_ratio=args.sample_temporal,
#             file_size=ctx['file_size'],
#             transform_data=transform_data,
#             transform_label=transform_label,
#             mask_factor=args.mask_factor
#         )


dataset_size = len(dataset_train)
train_size = int(0.9 * dataset_size)  
val_size = dataset_size - train_size 

dataset_train, dataset_valid = random_split(dataset_train, [train_size, val_size])

print('Creating data loaders')
train_sampler = RandomSampler(dataset_train)
valid_sampler = RandomSampler(dataset_valid)

dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

dataloader_valid = DataLoader(
    dataset_valid, batch_size=args.batch_size,
    sampler=valid_sampler, num_workers=args.workers,
    pin_memory=True, collate_fn=default_collate)

# Training

step = 0

def train_one_epoch(model, vgg_model, masked_criterion, optimizer, lr_scheduler, 
                    dataloader, device, epoch, print_freq):
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    vel_vgg_loss = torch.tensor([0.], device=device)
    
    # for VGG amp loss
    upsample = torch.nn.Upsample(size=(70, 70), mode="bicubic")
    
    for mask, amp, vel in metric_logger.log_every(dataloader, print_freq, header):
        start_time = time.time()
        
        optimizer.zero_grad()

        mask, amp, vel = mask.to(device), amp.to(device), vel.to(device)
        identity_mask = torch.ones_like(mask)
        
        vel_pred = model(amp)

        vel_loss, vel_loss_g1v, vel_loss_g2v = masked_criterion(vel_pred, vel, mask)

        # Calculating the perceptual loss using VGG-16 model for velocity and amplitude
        if args.lambda_vgg_vel>0:
            vgg_vel = vel.repeat(1,3,1,1)
            vgg_vel_pred = vel_pred.repeat(1,3,1,1)

            with torch.no_grad():
                vgg_features_vel = vgg_model(vgg_vel, vgg_layer_output=args.vgg_layer_output)   
            vgg_features_vel_pred = vgg_model(vgg_vel_pred, vgg_layer_output=args.vgg_layer_output)

            vel_vgg_loss, vel_vgg_loss_g1v, vel_vgg_loss_g2v = masked_criterion(vgg_features_vel, vgg_features_vel_pred, mask)

            vel_vgg_loss_g1v_val = vel_vgg_loss_g1v.item()
            vel_vgg_loss_g2v_val = vel_vgg_loss_g2v.item()

            metric_logger.update(vel_vgg_loss_g1v = vel_vgg_loss_g1v_val, 
                             vel_vgg_loss_g2v = vel_vgg_loss_g2v_val)
            

        # Calcultaing the reconstruction loss on encoder-decoder for both amp and vel     
        amp_loss_recons = 0  
        vel_loss_recons = 0 
        if args.lambda_recons>0:
            # print("applying reconstruction")
            vel_recons = model.vel_model.forward(vel)
            amp_recons = model.amp_model.forward(amp)
            amp_loss_recons = nn.MSELoss()(amp_recons, amp)  # Compute amplitude loss
            vel_loss_recons = nn.MSELoss()(vel_recons, vel)  # Compute velocity loss
            
            metric_logger.update(amp_loss_recons = amp_loss_recons, 
                             vel_loss_recons = vel_loss_recons)
            


        
        loss = args.lambda_vel * vel_loss + args.lambda_vgg_vel * vel_vgg_loss + args.lambda_recons * amp_loss_recons + args.lambda_recons * vel_loss_recons

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        vel_loss_g1v_val = vel_loss_g1v.item()
        vel_loss_g2v_val = vel_loss_g2v.item()

        batch_size = amp.shape[0]
        metric_logger.update(
                             loss = loss_val, 
                             lr = optimizer.param_groups[0]['lr'],
                             vel_loss_g1v = vel_loss_g1v_val,
                             vel_loss_g2v = vel_loss_g2v_val,
                            )
        
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))

        step += 1


def evaluate(model, criterion, dataloader, device, ctx, transform_data=None, transform_label=None):
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    model.eval()

    all_outputs = []
    all_labels = []
    with torch.no_grad():
        total_samples = 0
        
        eval_metrics = ["vel_sum_abs_error", "vel_sum_squared_error"]
        eval_dict = {}
        for metric in eval_metrics:
            eval_dict[metric] = 0

        val_loss = 0
        for _, amp, vel in metric_logger.log_every(dataloader, 50, header):
            
            amp = amp.to(device, non_blocking=True)
            vel = vel.to(device, non_blocking=True)

            batch_size = amp.shape[0]
            total_samples += batch_size
            
            vel_pred = model(amp)

            vel_loss, vel_loss_g1v, vel_loss_g2v = criterion(vel_pred, vel)

            loss = vel_loss
            val_loss += loss.item()

            eval_dict["vel_sum_abs_error"] += (l1_loss(vel, vel_pred) * batch_size).item()
            eval_dict["vel_sum_squared_error"] += (l2_loss(vel, vel_pred) * batch_size).item()

            all_outputs.append(vel_pred.detach().cpu())
            all_labels.append(vel.detach().cpu())
        
        for metric in eval_metrics:
            eval_dict[metric] /= total_samples 

    val_loss /= len(dataloader)

    all_output = torch.concat(all_outputs, axis=0)
    all_label = torch.concat(all_labels, axis=0)

    all_output = transform_label.inverse_transform(all_output.numpy())
    all_label = transform_label.inverse_transform(all_label.numpy())

    
    l1loss_eval = l1loss(torch.tensor(all_output), torch.tensor(all_label))

    eval_dict["Val_Loss"] = val_loss
    eval_dict["L1_Loss"] = l1loss_eval.item()
    
    return val_loss, l1loss_eval, eval_dict

device = torch.device(args.device)
torch.backends.cudnn.benchmark = True

print('Creating model')
if args.model not in iunet_network.model_dict:
    print('Unsupported model.')
    sys.exit()


def set_inverse_params(args, inverse_model_params):
        inverse_model_params.setdefault('IUnetInverseModel', {})
        inverse_model_params['IUnetInverseModel']['cfg_path'] = args.cfg_path
        inverse_model_params['IUnetInverseModel']['latent_dim'] = args.latent_dim
        
        inverse_model_params.setdefault('UNetInverseModel', {})
        inverse_model_params['UNetInverseModel']['cfg_path'] = args.cfg_path
        inverse_model_params['UNetInverseModel']['latent_dim'] = args.latent_dim
        inverse_model_params['UNetInverseModel']['unet_depth'] = args.unet_depth
        inverse_model_params['UNetInverseModel']['unet_repeat_blocks'] = args.unet_repeat_blocks
        inverse_model_params['UNetInverseModel']['skip'] = args.skip
        return inverse_model_params
    
# creating inverse model
inverse_model_params = iunet_network.inverse_params
inverse_model_params = set_inverse_params(args, inverse_model_params)

model = iunet_network.model_dict[args.model](**inverse_model_params[args.model]).to(device)

print(count_parameters(model))

vgg_model = VGG16FeatureExtractor().to(device)

# Define loss function
l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

def masked_criterion(pred, gt, mask):
    B, C, H, W = pred.shape
    mask = mask.view(B, 1, 1, 1)
    num_elements = mask.sum() + 1

    squared_diff = ((pred - gt)**2) * mask
    abs_diff = (pred-gt).abs() * mask
    norm_l2_loss = torch.sum(squared_diff.mean(dim=[1, 2, 3])/num_elements)
    norm_l1_loss = torch.sum(abs_diff.mean(dim=[1, 2, 3])/num_elements)
    loss = args.lambda_g1v * norm_l1_loss + args.lambda_g2v * norm_l2_loss
    return loss, norm_l1_loss, norm_l2_loss

def criterion(pred, gt):
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
    return loss, loss_g1v, loss_g2v

def relative_l2_error(pred, gt):
    batch_size = gt.shape[0]
    pred = pred.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    numerator = torch.linalg.norm(pred - gt, ord=2, dim=1)
    denominator = torch.linalg.norm(gt, ord=2, dim=1)
    relative_loss = (numerator/denominator).mean()
    return relative_loss

# Scale lr according to effective batch size
lr = args.lr
optimizer = get_optimizer(args, model, lr)
lr_scheduler = get_lr_scheduler(args, optimizer)

# Convert scheduler to be per iteration instead of per epoch
warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]

model_without_ddp = model

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(iunet_network.replace_legacy(checkpoint['model']))
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    args.start_epoch = checkpoint['epoch'] + 1
    step = checkpoint['step']
    lr_scheduler.milestones = lr_milestones

print('Start training')
start_time = time.time()
best_loss = 10000
chp = 1 

train_vis_folder = os.path.join(args.output_path, args.plot_directory, 'train')
valid_vis_folder = os.path.join(args.output_path, args.plot_directory, 'validation')

os.makedirs(train_vis_folder, exist_ok=True)
os.makedirs(valid_vis_folder, exist_ok=True)
for epoch in tqdm(range(args.start_epoch, args.epochs)):
    train_one_epoch(model, vgg_model, masked_criterion, optimizer, lr_scheduler, dataloader_train,
                    device, epoch, args.print_freq)
    
    lr_scheduler.step()    
   
    val_loss, loss, eval_dict = evaluate(model, criterion, dataloader_valid, device, ctx, transform_data, transform_label)
    print("Test Metrics:", eval_dict)

    checkpoint = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'step': step
        }

    if (epoch+1)%args.plot_interval == 0:
        plot_images(args.num_images, dataset_train, model, epoch, train_vis_folder, device, transform_data, transform_label)
        plot_images(args.num_images, dataset_valid, model, epoch, valid_vis_folder, device, transform_data, transform_label)

        torch.save(
            checkpoint,
            os.path.join(args.output_path, 'latest_checkpoint.pth'))

    
    # Save checkpoint per epoch
    if loss < best_loss:
        torch.save(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
        print('saving checkpoint at epoch: ', epoch)
        chp = epoch
        best_loss = loss
        
    # Save checkpoint every epoch block
    print('current best loss: ', best_loss)
    print('current best epoch: ', chp)
    if args.output_path and (epoch + 1) % args.epoch_block == 0:
        torch.save(
            checkpoint,
            os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

# Predict
import csv  # Use "low-level" CSV to save memory on predictions
# %%time
test_files = list(Path('/kaggle/input/waveform-inversion/test').glob('*.npy'))
len(test_files)

x_cols = [f'x_{i}' for i in range(1, 70, 2)]
fieldnames = ['oid_ypos'] + x_cols

class TestDataset(Dataset):
    def __init__(self, test_files, transform_data=None):
        self.test_files = test_files
        self.transform_data = transform_data


    def __len__(self):
        return len(self.test_files)


    def __getitem__(self, i):
        test_file = self.test_files[i]
        data = np.load(test_file)
        if self.transform_data:
            data = self.transform_data(data)

        return data, test_file.stem
    
ds = TestDataset(test_files, transform_data)
dl = DataLoader(ds, batch_size=16, num_workers=4, pin_memory=True)

checkpoint = torch.load('/kaggle/working/Invnet_models/checkpoint.pth')

model_without_ddp.load_state_dict(iunet_network.replace_legacy(checkpoint['model']))

# Train
model.eval()
with open('submission.csv', 'wt', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for inputs, oids_test in tqdm(dl, desc='test'):
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = model(inputs)

        y_preds = outputs[:, 0].cpu().numpy()
        y_preds = transform_label.inverse_transform(y_preds)
        
        for y_pred, oid_test in zip(y_preds, oids_test):
            for y_pos in range(70):
                row = dict(
                    zip(
                        x_cols,
                        [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]
                    )
                )
                row['oid_ypos'] = f"{oid_test}_y_{y_pos}"
            
                writer.writerow(row)

