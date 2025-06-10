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

def infer():
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
    print("Number of test files:", len(test_files))
    # test_files = test_files[:10]  # Uncomment for debugging only
    test_dataset = data_manager.create_dataset(test_files, None, 'test')
    test_loader = data_manager.create_loader(
        test_files, None, 'test',
        batch_size=2,  # or your preferred batch size
        shuffle=False,
        num_workers=0
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
            for batch_idx, (seis, _) in enumerate(tqdm(test_loader, desc="Processing test files")):
                seis = seis.to(CFG.env.device, non_blocking=True).float()
                with autocast('cuda', enabled=CFG.use_amp):
                    preds = model(seis)
                preds = preds.cpu().numpy()
                for i in range(preds.shape[0]):
                    vel_map = preds[i]
                    if vel_map.shape[0] == 1:
                        vel_map = vel_map[0]
                    oid = Path(test_files[batch_idx * test_loader.batch_size + i]).stem
                    rows = format_submission(vel_map, oid)
                    writer.writerows(rows)  # Write this batch's rows immediately
    print("Inference complete and submission.csv saved.")

if __name__ == "__main__":
    infer() 