"""
train_kaggle.py: Training entry point for Kaggle environment
Usage:
    python src/core/train_kaggle.py --epochs 30 --batch 4 --amp
"""
import torch
from pathlib import Path
from src.core.config import CFG
from src.core.data_manager import DataManager
from src.core.model import get_model
from src.core.losses import get_loss_fn
import logging
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=CFG.epochs)
    parser.add_argument('--batch', type=int, default=CFG.batch)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    data_manager = DataManager(use_mmap=True)
    families = list(CFG.paths.families.keys())
    loaders = []
    for family in families:
        seis_files, vel_files, family_type = data_manager.list_family_files(family)
        loader = data_manager.create_loader(
            seis_files=seis_files,
            vel_files=vel_files,
            family_type=family_type,
            batch_size=args.batch,
            shuffle=True,
            num_workers=CFG.num_workers,
            distributed=False
        )
        loaders.append(loader)

    model = get_model()
    loss_fn = get_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    total_steps = sum(len(loader) for loader in loaders) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=CFG.lr, total_steps=total_steps)
    scaler = GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):
        model.train()
        for loader in loaders:
            for x, y in loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                with autocast(enabled=args.amp):
                    pred = model(x)
                    loss, _ = loss_fn(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        print(f"Epoch {epoch+1} complete.")

if __name__ == "__main__":
    main() 