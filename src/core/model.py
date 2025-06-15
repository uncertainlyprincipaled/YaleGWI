import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from copy import deepcopy
# from src.core.config import CFG  # Absolute import
import logging
from pathlib import Path

def get_model(backbone: str = "hgnetv2_b2.ssld_stage2_ft_in1k", pretrained: bool = False, ema_decay: float = 0.99):
    """Create and initialize the model."""
    model = SpecProjNet(backbone=backbone, pretrained=pretrained, ema_decay=ema_decay)
    
    # Move model to device
    model = model.to(CFG.env.device)
    
    # Convert all parameters to float32
    model = model.float()
    
    # Load weights if available
    if CFG.weight_path and Path(CFG.weight_path).exists():
        try:
            state_dict = torch.load(CFG.weight_path, map_location=CFG.env.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"Successfully loaded weights from {CFG.weight_path}")
        except Exception as e:
            logging.error(f"Failed to load weights: {e}")
            logging.info("Continuing with randomly initialized weights")
    
    return model

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
        # Print channel dimensions for debugging
        print(f"DecoderBlock - in_channels: {in_channels}, skip_channels: {skip_channels}, out_channels: {out_channels}")
        
        # Upsampling block
        if upsample_mode == "deconv":
            self.upsample = nn.ConvTranspose2d(
                in_channels, 
                out_channels,  # Directly output desired number of channels
                kernel_size=scale_factor, 
                stride=scale_factor
            )
            self.channel_reduction = nn.Identity()  # No need for additional channel reduction
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.channel_reduction = nn.Identity()
        
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
            out_channels + skip_channels,  # Concatenated channels
            out_channels,  # Output channels
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )

    def forward(self, x, skip=None):
        # Upsample and reduce channels
        x = self.upsample(x)
        
        # Process skip connection if available
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
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
        # Reverse encoder channels to match decoder order
        encoder_channels = list(reversed(encoder_channels))
        if skip_channels is None:
            skip_channels = encoder_channels[1:]  # Skip the first channel as it's the input
            
        # Print channel dimensions for debugging
        print(f"Encoder channels (reversed): {encoder_channels}")
        print(f"Skip channels: {skip_channels}")
        print(f"Decoder channels: {decoder_channels}")
        
        # Initial channel reduction from encoder to decoder
        self.initial_reduction = nn.Conv2d(encoder_channels[0], decoder_channels[0], 1)
        
        # Create channel reduction layers for each encoder feature
        self.channel_reductions = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 1)
            for in_ch, out_ch in zip(encoder_channels[1:], decoder_channels[1:])
        ])
        
        # Create skip connection channel reduction layers
        self.skip_reductions = nn.ModuleList([
            nn.Conv2d(skip_ch, out_ch, 1)
            for skip_ch, out_ch in zip(skip_channels, decoder_channels[1:])  # Skip first decoder channel
        ])
            
        self.blocks = nn.ModuleList([
            DecoderBlock2d(
                in_channels=decoder_channels[i],  # Current block's input channels
                skip_channels=decoder_channels[i+1] if i < len(decoder_channels)-1 else decoder_channels[-1],  # Next block's channels
                out_channels=decoder_channels[i+1] if i < len(decoder_channels)-1 else decoder_channels[-1],  # Next block's channels
                norm_layer=norm_layer,
                attention_type=attention_type,
                intermediate_conv=intermediate_conv,
                upsample_mode=upsample_mode,
                scale_factor=scale_factor,
            )
            for i, scale_factor in enumerate(scale_factors)
        ])

    def forward(self, feats: list[torch.Tensor]):
        # Reverse features to match decoder order
        feats = list(reversed(feats))
        
        # Initial channel reduction for the deepest feature
        x = self.initial_reduction(feats[0])
        
        # Reduce channels of remaining encoder features
        reduced_feats = [reduction(feat) for reduction, feat in zip(self.channel_reductions, feats[1:])]
        
        # Reduce channels of skip connections
        reduced_skips = [reduction(feat) for reduction, feat in zip(self.skip_reductions, feats[1:])]
        
        # Process through decoder blocks
        for block, skip in zip(self.blocks, reduced_skips):
            x = block(x, skip)
            
        return x

class CustomUpSample(nn.Module):
    """Memory-efficient upsampling module that combines bilinear upsampling with convolution."""
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # Use 1x1 conv for channel reduction to save memory
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # Use 3x3 conv for spatial features
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Enable gradient checkpointing if configured
        if CFG.gradient_checkpointing:
            self.conv = torch.utils.checkpoint.checkpoint_sequential(self.conv, 1)
            self.spatial_conv = torch.utils.checkpoint.checkpoint_sequential(self.spatial_conv, 1)
        
    def forward(self, x):
        # Use mixed precision if configured
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            # Upsample first to reduce memory usage during convolution
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
            # Apply convolutions
            x = self.conv(x)
            x = self.spatial_conv(x)
        return x

class CustomSubpixelUpsample(nn.Module):
    """Memory-efficient subpixel upsampling using pixel shuffle."""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        # Use 1x1 conv for initial channel reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2) // 2, 1)
        # Use 3x3 conv for spatial features
        self.conv2 = nn.Conv2d(out_channels * (scale_factor ** 2) // 2, 
                              out_channels * (scale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Enable gradient checkpointing if configured
        if CFG.gradient_checkpointing:
            self.conv1 = torch.utils.checkpoint.checkpoint_sequential(self.conv1, 1)
            self.conv2 = torch.utils.checkpoint.checkpoint_sequential(self.conv2, 1)
        
    def forward(self, x):
        # Use mixed precision if configured
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            # Progressive channel reduction to save memory
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pixel_shuffle(x)
        return x

class SegmentationHead2d(nn.Module):
    """Final segmentation head with optional upsampling."""
    def __init__(
        self,
        in_channels,
        out_channels,
        target_size: tuple[int] = (70, 70),  # Target output size
        kernel_size: int = 3,
        mode: str = "nontrainable",
    ):
        super().__init__()
        # Use 1x1 conv for initial channel reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        # Use 3x3 conv for spatial features
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Enable gradient checkpointing if configured
        self.use_checkpoint = CFG.gradient_checkpointing
        
        if mode == "nontrainable":
            self.upsample = nn.Upsample(size=target_size, mode="bilinear", align_corners=False)
        else:
            # Using custom subpixel upsampling
            self.upsample = CustomSubpixelUpsample(out_channels, out_channels, scale_factor=2)

    def forward(self, x):
        # Use mixed precision if configured
        with torch.cuda.amp.autocast(enabled=CFG.use_amp):
            # Progressive channel reduction to save memory
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(self.conv1, x)
                x = torch.utils.checkpoint.checkpoint(self.conv2, x)
            else:
                x = self.conv1(x)
                x = self.conv2(x)
            x = self.upsample(x)
        return x

class SpecProjNet(nn.Module):
    """SpecProj model with HGNet backbone."""
    def __init__(self, backbone: str = "hgnetv2_b2.ssld_stage2_ft_in1k", pretrained: bool = False, ema_decay: float = 0.99):
        super().__init__()
        
        # Input projection layer to convert to 3 channels
        self.input_proj = nn.Conv2d(5, 3, kernel_size=1)
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )
            
        # Get channel dimensions from backbone
        self.encoder_channels = self.backbone.feature_info.channels()
        print("Backbone output channels:", self.encoder_channels)
        
        # Initialize decoder blocks with correct channel dimensions
        self.decoder_blocks = nn.ModuleList()
        in_channels = self.encoder_channels[-1]  # Start with last feature map channels
        
        for i in range(len(self.encoder_channels)-1):
            skip_channels = self.encoder_channels[-(i+2)]
            out_channels = self.encoder_channels[-(i+2)]
            print(f"DecoderBlock - in_channels: {in_channels}, skip_channels: {skip_channels}, out_channels: {out_channels}")
            self.decoder_blocks.append(
                DecoderBlock2d(in_channels, skip_channels, out_channels)
            )
            in_channels = out_channels
            
        # Final projection layer
        self.final_proj = nn.Conv2d(self.encoder_channels[0], 1, kernel_size=1)
        
        # Final resize layer to match target dimensions
        self.final_resize = nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False)
        
        # Initialize EMA
        self.ema_decay = ema_decay
        self.ema_model = None
        
    def forward(self, x):
        # Handle 5D input [batch, channels, sources, receivers, timesteps]
        if x.dim() == 5:
            B, C, S, R, T = x.shape
            # Reshape to combine sources and channels: [batch, sources*channels, receivers, timesteps]
            x = x.reshape(B, S*C, R, T)
            
        # Project input to 3 channels
        x = self.input_proj(x)
            
        # Get features from backbone
        features = self.backbone(x)
        
        # Decode features
        x = features[-1]  # Start with last feature map
        for i, decoder in enumerate(self.decoder_blocks):
            skip = features[-(i+2)] if i < len(features)-1 else None
            x = decoder(x, skip)
            
        # Final projection
        x = self.final_proj(x)
        
        # Resize to target dimensions
        x = self.final_resize(x)
        
        return x
        
    def update_ema(self):
        """Update EMA weights."""
        if self.ema_model is not None:
            self.ema_model.update(self)
            
    def set_ema(self):
        """Set EMA weights."""
        if self.ema_model is not None:
            self.ema_model.set(self)
            
    def get_ema_model(self):
        """Get EMA model."""
        return self.ema_model if self.ema_model is not None else self 