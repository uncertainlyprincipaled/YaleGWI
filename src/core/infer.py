# All data IO in this file must go through DataManager (src/core/data_manager.py)
# Do NOT load data directly in this file.
#
# ## Inference

# %%
import torch, pandas as pd
from pathlib import Path
import os
from config import CFG
from data_manager import DataManager
from model import SpecProjNet

def format_submission(vel_map, oid):
    # keep *odd* x-columns only (1,3,…,69)
    cols = vel_map[:,1::2]
    rows = []
    for y,row in enumerate(cols):
        rows.append({'oid_ypos': f'{oid}_y_{y}', **{f'x_{2*i+1}':v
                      for i,v in enumerate(row.tolist())}})
    return rows

def infer(weights=None):
    # Use CFG.weight_path as default if weights not provided
    weights = weights or CFG.weight_path
    # Check for existence and provide a clear error if missing
    if not Path(weights).exists():
        raise FileNotFoundError(f"Model weights not found at {weights}. If running on Kaggle, make sure to add the dataset containing the weights (e.g., 'yalegwi') to your notebook.")
    # Initialize data manager
    data_manager = DataManager()
    
    # Load model
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=False,
        ema_decay=0.0,  # No EMA for inference
    ).to(CFG.env.device)
    
    # Load state dict and remove 'ema.module.' prefix if present
    state_dict = torch.load(weights, map_location=CFG.env.device)
    if any(k.startswith('ema.module.') for k in state_dict.keys()):
        state_dict = {k.replace('ema.module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    test_files = data_manager.get_test_files()
    # For test, family_type is not needed, but we pass 'Fault' for single-sample-per-file
    test_loader = data_manager.create_loader(
        test_files, test_files, 'Fault',
        batch_size=1, shuffle=False
    )

    rows = []
    with torch.no_grad(), torch.amp.autocast('cuda'):  # Updated autocast syntax
        for seis, oid in test_loader:
            seis = seis.to(CFG.env.device)
            
            # Original prediction
            vel = model(seis)
            
            # TTA: Flip prediction
            seis_flip = seis.flip(-1)  # Flip receiver dimension
            vel_flip = model(seis_flip)
            vel_flip = vel_flip.flip(-1)  # Flip back
            
            # Average predictions
            vel = (vel + vel_flip) / 2
            
            vel = vel.cpu().float().numpy()[0,0]
            # Get the stem from the file path
            oid_str = Path(oid[0]).stem if isinstance(oid[0], (str, Path)) else str(oid[0])
            rows += format_submission(vel, oid_str)
            
    # Ensure we're in the project root directory before saving
    project_root = Path(__file__).parent.parent.parent
    if os.getcwd() != str(project_root):
        os.chdir(project_root)
        print(f"Changed working directory to: {project_root}")
    
    pd.DataFrame(rows).to_csv('submission.csv', index=False)

if __name__ == '__main__':
    infer() 