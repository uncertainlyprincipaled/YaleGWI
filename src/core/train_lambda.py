"""
train_lambda.py: Training entry point for Lambda Labs with preprocessed data
Usage:
    python src/core/train_lambda.py --epochs 120 --batch 8 --amp --use_s3
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from src.core.config import CFG
from src.core.data_manager import DataManager
from src.core.geometric_loader import FamilyDataLoader
from src.core.geometric_cv import GeometricCrossValidator
from src.core.model import get_model
from src.core.losses import get_loss_fn
from src.core.registry import ModelRegistry
from src.core.checkpoint import CheckpointManager
import logging
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime
import json

def setup_lambda_environment():
    """Setup Lambda Labs environment variables and paths."""
    # Lambda Labs typically has data in /data or /mnt/data
    data_paths = ['/data', '/mnt/data', '/home/ubuntu/data']
    for path in data_paths:
        if Path(path).exists():
            return path
    return '/tmp/data'  # Fallback

def setup_ddp():
    """Setup distributed training if multiple GPUs available."""
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return local_rank, True
    else:
        return 0, False

def main():
    parser = argparse.ArgumentParser(description='Lambda Labs Training')
    parser.add_argument('--epochs', type=int, default=CFG.epochs)
    parser.add_argument('--batch', type=int, default=CFG.batch)
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use_s3', action='store_true', help='Use S3 for data loading')
    parser.add_argument('--use_drive', action='store_true', help='Use Google Drive for data loading')
    parser.add_argument('--data_path', type=str, default=None, help='Local data path')
    parser.add_argument('--family', type=str, default=None, help='Train on specific family only')
    args = parser.parse_args()

    # Setup environment
    local_rank, is_distributed = setup_ddp()
    is_main = (local_rank == 0) or not is_distributed
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING)

    # Determine data path
    if args.data_path:
        data_root = Path(args.data_path)
    elif args.use_s3:
        data_root = CFG.s3_paths.preprocessed_prefix
    elif args.use_drive:
        data_root = Path('/content/drive/MyDrive/YaleGWI/preprocessed')
    else:
        data_root = Path(setup_lambda_environment()) / 'preprocessed'

    print(f"Using data root: {data_root}")
    print(f"Lambda Labs environment detected: {CFG.env.kind}")

    # Initialize Phase 1 components
    print("Initializing Phase 1 components...")
    
    # Model registry
    registry = ModelRegistry()
    print("✅ Model registry initialized")
    
    # Checkpoint manager
    checkpoint_mgr = CheckpointManager()
    print("✅ Checkpoint manager initialized")
    
    # Data manager
    data_manager = DataManager(use_s3=args.use_s3)
    print("✅ Data manager initialized")
    
    # Cross-validator
    cv = GeometricCrossValidator(n_splits=5)
    print("✅ Cross-validator initialized")

    # Initialize family data loader
    print("Initializing family data loader...")
    family_loader = FamilyDataLoader(
        data_root=str(data_root),
        batch_size=args.batch,
        num_workers=CFG.num_workers,
        extract_features=True
    )
    print("✅ Family data loader initialized")

    # Get available families
    if args.family:
        families = [args.family]
    else:
        families = list(CFG.paths.families.keys())
    
    print(f"Training on families: {families}")

    # Initialize model
    model = get_model()
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    model.train()
    print("✅ Model initialized")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CFG.lr, 
        weight_decay=CFG.weight_decay
    )
    
    # Calculate total steps for OneCycleLR
    total_steps = 0
    for family in families:
        try:
            dataset = family_loader.get_dataset(family)
            total_steps += len(dataset) // args.batch * args.epochs
        except Exception as e:
            print(f"Warning: Could not get dataset for {family}: {e}")
    
    if total_steps == 0:
        total_steps = 1000  # Fallback
    
    scheduler = OneCycleLR(optimizer, max_lr=CFG.lr, total_steps=total_steps)
    scaler = GradScaler(enabled=args.amp)
    loss_fn = get_loss_fn()
    
    print(f"✅ Optimizer and scheduler initialized (total_steps: {total_steps})")

    # Training loop
    print("Starting training loop...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        for family in families:
            try:
                # Get data loader for this family
                loader = family_loader.get_loader(family)
                
                print(f"Training on family {family} (epoch {epoch+1}/{args.epochs})")
                
                for batch_idx, batch in enumerate(loader):
                    try:
                        # Extract data and features
                        data = batch['data'].to(local_rank if is_distributed else CFG.env.device)
                        features = batch.get('features', {})
                        
                        # For now, use data directly (we'll add feature integration later)
                        if isinstance(data, dict):
                            # Handle case where data is already a dict
                            x = data.get('data', data)
                        else:
                            x = data
                        
                        # Create dummy target for now (we need to load velocity data)
                        # TODO: Integrate velocity data loading
                        y = torch.zeros(x.shape[0], 1, 70, 70, device=x.device)
                        
                        # Forward pass with mixed precision
                        with autocast(enabled=args.amp):
                            pred = model(x)
                            loss, loss_components = loss_fn(pred, y)
                        
                        # Backward pass
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        
                        epoch_loss += loss.item()
                        epoch_batches += 1
                        
                        # Log progress
                        if batch_idx % 10 == 0 and is_main:
                            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            print(f"  WARNING: OOM in batch {batch_idx}, skipping")
                            continue
                        else:
                            raise e
                            
            except Exception as e:
                print(f"Error training on family {family}: {e}")
                continue
        
        # Epoch summary
        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
            print(f"Epoch {epoch+1}/{args.epochs} completed. Avg loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                checkpoint_mgr.save_checkpoint(
                    model=model.module if is_distributed else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    metrics={'loss': avg_loss},
                    upload_s3=args.use_s3
                )
        else:
            print(f"Epoch {epoch+1}/{args.epochs} completed. No batches processed.")

    # Final cleanup
    if is_distributed:
        dist.destroy_process_group()
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    main() 