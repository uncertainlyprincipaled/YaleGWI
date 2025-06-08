# All data IO in this file must go through DataManager (src/core/data_manager.py)
# Do NOT load data directly in this file.
#
# ## Inference

# %%
import torch, pandas as pd
from pathlib import Path
import os
import time
from tqdm import tqdm
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

def estimate_inference_time(test_files=None, batch_size=4, disable_tta=False):
    if test_files is None:
        data_manager = DataManager()
        test_files = data_manager.get_test_files()
    sample_size = min(10, len(test_files))
    print(len(test_files))
    sample_files = test_files[:sample_size]  # Only use a small sample!
    try:
        start_time = time.time()
        # Run inference on the sample, not the full set!
        infer(weights=CFG.weight_path, batch_size=batch_size, disable_tta=disable_tta, test_files=sample_files)
        elapsed = time.time() - start_time
        # Estimate total time
        total_samples = len(test_files)
        estimated_time = (elapsed / sample_size) * total_samples
        return estimated_time
    except Exception as e:
        print(f"Error during time estimation: {str(e)}")
        return None

def infer(weights=None, batch_size=8, disable_tta=False, test_files=None):
    """Run inference with batching and checkpointing."""
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

    if test_files is None:
        test_files = data_manager.get_test_files()
    # For test, family_type is not needed, but we pass 'Fault' for single-sample-per-file
    test_loader = data_manager.create_loader(
        test_files, test_files, 'Fault',
        batch_size=batch_size,
        shuffle=False
    )

    rows = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for seis, oid in tqdm(test_loader, desc="Inference"):
            seis = seis.to(CFG.env.device).to(torch.float16)
            
            # Original prediction
            vel = model(seis)
            
            if not disable_tta:
                # TTA: Flip prediction
                seis_flip = seis.flip(-1)
                vel_flip = model(seis_flip)
                vel_flip = vel_flip.flip(-1)
                vel = (vel + vel_flip) / 2
            
            # Keep in float16 until final conversion
            vel = vel.cpu().numpy()
            
            # Process batch results
            for b in range(vel.shape[0]):
                oid_str = Path(oid[b]).stem if isinstance(oid[b], (str, Path)) else str(oid[b])
                rows += format_submission(vel[b,0], oid_str)
                
            # Save intermediate results less frequently
            if len(rows) % 25000 == 0:
                print(f"Saving intermediate results: {len(rows)} predictions processed")
                pd.DataFrame(rows).to_csv('submission_partial.csv', index=False)
                rows = []  # <-- Add this line to free memory
    
    # Save final results
    print(f"Saving final results: {len(rows)} predictions processed")
    pd.DataFrame(rows).to_csv('submission.csv', index=False)
    if Path('submission_partial.csv').exists():
        Path('submission_partial.csv').unlink()  # Remove partial file

if __name__ == '__main__':
    # Estimate time first
    est_time = estimate_inference_time(None, disable_tta=False)
    if est_time is not None:
        print(f"Estimated inference time: {est_time/60:.1f} minutes")
        # If inference time is > 8 hours, fail loudly
        if est_time > 8 * 60 * 60:
            raise ValueError("Estimated inference time is > 8 hours, please run with disable_tta=True")
    else:
        print("Could not estimate inference time, proceeding with full inference")
    
    # Run full inference
    infer(disable_tta=True) 