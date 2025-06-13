"""
train_ec2.py: Multi-GPU (DDP) training entry point for AWS EC2 (g4dn.12xlarge)
Usage:
    python -m torch.distributed.launch --nproc_per_node=4 src/core/train_ec2.py --epochs 120 --batch 4 --amp
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=CFG.epochs)
    parser.add_argument('--batch', type=int, default=CFG.batch)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    local_rank = setup_ddp()
    is_main = (local_rank == 0)
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING)

    # DataManager: use fine-grained dataset if available
    fine_root = Path('/mnt/waveform-inversion/fine/train_samples')
    use_fine = fine_root.exists()
    data_manager = DataManager(use_mmap=True)
    families = list(CFG.paths.families.keys())
    loaders = []
    for family in families:
        if use_fine:
            # Use per-sample files
            family_dir = fine_root / family
            sample_files = sorted(family_dir.glob('sample_*.npy'))
            # Dummy vel_files for now (assume same structure)
            vel_files = sample_files
            loader = data_manager.create_loader(
                seis_files=sample_files,
                vel_files=vel_files,
                family_type=family,
                batch_size=args.batch,
                shuffle=True,
                num_workers=CFG.num_workers,
                distributed=True
            )
        else:
            seis_files, vel_files, family_type = data_manager.list_family_files(family)
            loader = data_manager.create_loader(
                seis_files=seis_files,
                vel_files=vel_files,
                family_type=family_type,
                batch_size=args.batch,
                shuffle=True,
                num_workers=CFG.num_workers,
                distributed=True
            )
        loaders.append(loader)

    model = get_model()
    model = DDP(model, device_ids=[local_rank])
    loss_fn = get_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    total_steps = sum(len(loader) for loader in loaders) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=CFG.lr, total_steps=total_steps)
    scaler = GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):
        model.train()
        for loader in loaders:
            for x, y in loader:
                x = x.cuda(local_rank, non_blocking=True)
                y = y.cuda(local_rank, non_blocking=True)
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
        if is_main:
            print(f"Epoch {epoch+1} complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 