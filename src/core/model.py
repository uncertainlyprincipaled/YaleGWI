import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from monai.networks.blocks import UpSample, SubpixelUpsample
from copy import deepcopy

class ModelEMA(nn.Module):
    """Exponential Moving Average wrapper for model weights."""
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

class ConvBnAct2d(nn.Module):
    """Convolution block with optional batch norm and activation."""
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
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=norm_layer == nn.Identity,
        )
        self.bn = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SCSEModule2d(nn.Module):
    """Squeeze-and-Excitation module with channel and spatial attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock2d(nn.Module):
    """Decoder block with skip connection and optional attention."""
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
        self.upsample = (
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=scale_factor, stride=scale_factor)
            if upsample_mode == "deconv"
            else UpSample(in_channels, in_channels, scale_factor=scale_factor)
        )
        
        # Skip connection processing
        if intermediate_conv:
            self.skip_conv = ConvBnAct2d(skip_channels, skip_channels, 3, padding=1)
        else:
            self.skip_conv = nn.Identity()
            
        # Attention
        if attention_type == "scse":
            self.attention = SCSEModule2d(skip_channels)
        else:
            self.attention = nn.Identity()
            
        # Final convolution
        self.conv = ConvBnAct2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            skip = self.skip_conv(skip)
            skip = self.attention(skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UnetDecoder2d(nn.Module):
    """UNet decoder with configurable channels and upsampling."""
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
        if skip_channels is None:
            skip_channels = encoder_channels[:-1]
            
        self.blocks = nn.ModuleList([
            DecoderBlock2d(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                norm_layer=norm_layer,
                attention_type=attention_type,
                intermediate_conv=intermediate_conv,
                upsample_mode=upsample_mode,
                scale_factor=scale_factor,
            )
            for in_ch, skip_ch, out_ch, scale_factor in zip(
                encoder_channels[:-1],
                skip_channels,
                decoder_channels,
                scale_factors,
            )
        ])

    def forward(self, feats: list[torch.Tensor]):
        x = feats[-1]
        for block, skip in zip(self.blocks, reversed(feats[:-1])):
            x = block(x, skip)
        return x

class SegmentationHead2d(nn.Module):
    """Final segmentation head with optional upsampling."""
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2),
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        if mode == "nontrainable":
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
        else:
            self.upsample = SubpixelUpsample(in_channels, out_channels, scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample(self.conv(x))

class SpecProjNet(nn.Module):
    """SpecProj model with HGNet backbone."""
    def __init__(
        self,
        backbone: str = "hgnetv2_b2.ssld_stage2_ft_in1k",
        pretrained: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        # Load backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        
        # Update stem stride
        self.backbone.stem[0].stride = (1,1)
        
        # Get encoder channels
        encoder_channels = self.backbone.feature_info.channels()
        
        # Create decoder
        self.decoder = UnetDecoder2d(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32),
            scale_factors=(1,2,2,2),
            attention_type="scse",
            intermediate_conv=True,
        )
        
        # Create head
        self.head = SegmentationHead2d(
            in_channels=32,
            out_channels=1,
            scale_factor=(2,2),
            mode="nontrainable",
        )
        
        # Initialize EMA
        self.ema = ModelEMA(self, decay=ema_decay) if ema_decay > 0 else None
        
    def _update_stem(self):
        """Update stem convolution stride."""
        self.backbone.stem[0].stride = (1,1)
        
    def forward(self, x):
        # Get encoder features
        feats = self.backbone(x)
        
        # Decode
        x = self.decoder(feats)
        
        # Final head
        x = self.head(x)
        
        return x
        
    def update_ema(self):
        """Update EMA weights."""
        if self.ema is not None:
            self.ema.update(self)
            
    def set_ema(self):
        """Set EMA weights."""
        if self.ema is not None:
            self.ema.set(self)
            
    def get_ema_model(self):
        """Get EMA model."""
        return self.ema.module if self.ema is not None else self 