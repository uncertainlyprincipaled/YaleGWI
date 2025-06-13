import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from copy import deepcopy
from src.core.config import CFG  # Absolute import
import logging

def get_model():
    """Create and return a SpecProjNet model instance."""
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=False,  # Always False to prevent downloading
        ema_decay=CFG.ema_decay,
        weights_path=CFG.weight_path  # Pass the weights path from CFG
    ).to(CFG.env.device)
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
    def __init__(
        self,
        backbone: str = "hgnetv2_b2.ssld_stage2_ft_in1k",
        pretrained: bool = False,  # Changed to False to prevent downloading
        ema_decay: float = 0.99,
        weights_path: str = None,  # Added parameter for local weights path
    ):
        super().__init__()
        # Load backbone with gradient checkpointing enabled
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True, 
            in_chans=5,
            checkpoint_path=''  # Enable gradient checkpointing
        )
        
        # Load local weights if provided
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                # Try loading with strict=False to handle missing keys
                missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys in state dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
            except Exception as e:
                print(f"Warning: Could not load weights from {weights_path}: {str(e)}")
                print("Continuing with randomly initialized weights")
        
        # Update stem stride
        self._update_stem()
        
        # Get encoder channels
        encoder_channels = self.backbone.feature_info.channels()
        print(f"Encoder channels: {encoder_channels}")  # Debug print
        
        # Create decoder
        self.decoder = UnetDecoder2d(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32),
            scale_factors=(1,2,2,2),
            attention_type="scse",
            intermediate_conv=True,
        )
        
        # Create head with target size
        self.head = SegmentationHead2d(
            in_channels=32,
            out_channels=1,
            target_size=(70, 70),  # Set target output size
            mode="nontrainable",
        )
        
        # Initialize EMA
        self.ema = ModelEMA(self, decay=ema_decay) if ema_decay > 0 else None
        
    def _update_stem(self):
        """Update stem convolution stride."""
        if hasattr(self.backbone, 'stem'):
            if hasattr(self.backbone.stem, 'stem1'):
                self.backbone.stem.stem1.conv.stride = (1,1)
            elif hasattr(self.backbone.stem, 'conv'):
                self.backbone.stem.conv.stride = (1,1)
            else:
                raise ValueError("Unknown stem structure in backbone")
        else:
            raise ValueError("Backbone does not have stem attribute")
        
    def forward(self, x, max_sources_per_step=5):
        # Input shape assertions
        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.ndim in [2, 3, 4, 5], f"Unexpected input ndim: {x.ndim}"
        if x.ndim == 5:
            B, S, C, T, R = x.shape
            assert C == 1 or C == 5, f"Expected channel dim 1 or 5, got {C}"
        elif x.ndim == 3:
            B, T, R = x.shape
        elif x.ndim == 2:
            T, R = x.shape
        # Optionally, check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values!")
            
        # Log input tensor stats
        logging.info(f"Input tensor shape: {x.shape}, dtype: {x.dtype}, min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
        
        # Handle different input shapes
        if len(x.shape) == 5:  # (B, S, C, T, R)
            B, S, C, T, R = x.shape
            # Reshape to combine source and channel dimensions
            x = x.reshape(B, S*C, T, R)
        elif len(x.shape) == 3:  # (B, T, R)
            x = x.unsqueeze(1)  # Add source dimension -> (B, 1, T, R)
        elif len(x.shape) == 2:  # (T, R)
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and source dimensions -> (1, 1, T, R)
            
        B, S, T, R = x.shape
        
        # Process each source separately to save memory
        outputs = []
        for s in range(S):
            # Get current source
            x_s = x[:, s:s+1, :, :]  # (B, 1, T, R)
            x_s = x_s.repeat(1, 5, 1, 1)  # (B, 5, T, R)
            
            # Get encoder features
            feats = self.backbone(x_s)
            
            # Decode
            x_s = self.decoder(feats)
            
            # Final head
            x_s = self.head(x_s)
            
            outputs.append(x_s)
            
            # Clear intermediate tensors
            del feats
            torch.cuda.empty_cache()
            
        # Combine outputs
        x = torch.stack(outputs, dim=1)  # (B, S, 1, H, W)
        x = x.mean(dim=1)  # Average over sources
        
        # Clear outputs list
        del outputs
        torch.cuda.empty_cache()
        
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