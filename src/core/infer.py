"""
Inference script that uses DataManager for all data IO.
All data IO in this file must go through DataManager (src/core/data_manager.py)
Do NOT load data directly in this file.
"""
import torch
from tqdm import tqdm
from torch.amp import autocast
from model import SpecProjNet
from data_manager import DataManager
from config import CFG
import pandas as pd
import os
from pathlib import Path
import csv
import numpy as np
from src.core.preprocess import preprocess_for_inference, create_inference_optimized_dataset

def format_submission(vel_map, oid):
    # vel_map: (70, 70) numpy array
    # oid: string (file stem)
    rows = []
    for y in range(vel_map.shape[0]):
        row = {'oid_ypos': f'{oid}_y_{y}'}
        for i, x in enumerate(range(1, 70, 2)):
            row[f'x_{x}'] = float(vel_map[y, x])
        rows.append(row)
    return rows

def infer(use_optimized_dataset: bool = True, 
          preprocess_test_data: bool = True,
          cache_preprocessing: bool = True):
    """
    Enhanced inference function with optimized preprocessing.
    
    Args:
        use_optimized_dataset: Whether to create/use optimized zarr dataset
        preprocess_test_data: Whether to preprocess test data (recommended)
        cache_preprocessing: Whether to cache preprocessing results
    """
    # Use weights path from config
    weights = CFG.weight_path

    # Model setup
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=False,
        ema_decay=0.99,
    ).to(CFG.env.device)
    print("Model created")
    checkpoint = torch.load(weights, map_location=CFG.env.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_ema()
    print("Weights loaded and EMA set")
    model.eval()
    print("Model set to eval")

    # Data setup
    data_manager = DataManager()
    test_files = data_manager.get_test_files()
    print(f"Number of test files: {len(test_files)}")
    
    # Preprocess test data if requested
    if preprocess_test_data:
        print("Preprocessing test data for inference...")
        test_output_dir = Path('/kaggle/working/preprocessed_test')
        
        if use_optimized_dataset:
            # Create optimized zarr dataset
            zarr_path = create_inference_optimized_dataset(
                test_files=test_files,
                output_dir=test_output_dir,
                use_cache=cache_preprocessing
            )
            print(f"Created optimized dataset at: {zarr_path}")
            
            # Load zarr dataset
            import zarr
            test_data = zarr.open(str(zarr_path))
            print(f"Loaded test dataset: {test_data.shape}")
            
            # Create simple dataset class for zarr data
            class ZarrTestDataset:
                def __init__(self, zarr_data, file_names):
                    self.data = zarr_data
                    self.file_names = file_names
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return torch.from_numpy(self.data[idx]).float(), self.file_names[idx]
            
            test_dataset = ZarrTestDataset(test_data, [f.stem for f in test_files])
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=4,  # Adjust based on GPU memory
                shuffle=False,
                num_workers=2
            )
        else:
            # Use individual file processing
            test_dataset = data_manager.create_dataset(test_files, None, 'test')
            test_loader = data_manager.create_loader(
                test_files, None, 'test',
                batch_size=4,
                shuffle=False,
                num_workers=2
            )
    else:
        # Use original data loading (not recommended for float16 models)
        print("⚠️ Warning: Using original data loading without preprocessing")
        print("⚠️ This may cause precision mismatches with float16 models")
        test_dataset = data_manager.create_dataset(test_files, None, 'test')
        test_loader = data_manager.create_loader(
            test_files, None, 'test',
            batch_size=4,
            shuffle=False,
            num_workers=2
        )
    
    print("Test loader created")

    # Prepare CSV header
    x_cols = [f'x_{x}' for x in range(1, 70, 2)]
    fieldnames = ['oid_ypos'] + x_cols

    print("Starting inference loop")
    with open('submission.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        with torch.no_grad():
            for batch_idx, (seis, file_ids) in enumerate(tqdm(test_loader, desc="Processing test files")):
                # Move data to device
                seis = seis.to(CFG.env.device, non_blocking=True)
                
                # Apply preprocessing if not using optimized dataset
                if not preprocess_test_data:
                    # Apply same preprocessing as training
                    seis_np = seis.cpu().numpy()
                    processed_seis = []
                    for i in range(seis_np.shape[0]):
                        processed = preprocess_for_inference(
                            seis_np[i], dt_decimate=4, use_cache=cache_preprocessing
                        )
                        processed_seis.append(processed)
                    seis = torch.from_numpy(np.stack(processed_seis)).to(CFG.env.device)
                
                # Ensure float16 for model compatibility
                if seis.dtype != torch.float16:
                    seis = seis.half()
                
                with autocast('cuda', enabled=CFG.use_amp):
                    preds = model(seis)
                
                # Convert predictions to float32 for submission
                preds = preds.float().cpu().numpy()
                
                for i in range(preds.shape[0]):
                    vel_map = preds[i]
                    if vel_map.shape[0] == 1:
                        vel_map = vel_map[0]
                    
                    # Get file ID
                    if isinstance(file_ids, list):
                        oid = file_ids[i]
                    else:
                        oid = file_ids[i] if hasattr(file_ids, '__getitem__') else str(file_ids)
                    
                    rows = format_submission(vel_map, oid)
                    writer.writerows(rows)  # Write this batch's rows immediately
                    
                    # Flush buffer periodically
                    if batch_idx % 100 == 0:
                        csvfile.flush()
    
    print("Inference complete and submission.csv saved.")

def infer_optimized():
    """
    Optimized inference function with all enhancements enabled.
    """
    return infer(
        use_optimized_dataset=True,
        preprocess_test_data=True,
        cache_preprocessing=True
    )

if __name__ == "__main__":
    # Run optimized inference by default
    infer_optimized() 