import torch
from pathlib import Path
from src.core.model import SpecProjNet
from src.core.config import CFG

def analyze_checkpoint(ckpt_path):
    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location='cpu')
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Keys in state_dict: {list(state_dict.keys())[:5]} ...")
    print(f"Total parameters: {len(state_dict)}")
    
    # Optionally, load into model
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=False,
        ema_decay=0.0
    )
    model.load_state_dict(state_dict)
    print(model)
    # Print parameter statistics
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, mean={param.data.mean():.4f}, std={param.data.std():.4f}")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "outputs/best.pth"
    analyze_checkpoint(ckpt)
