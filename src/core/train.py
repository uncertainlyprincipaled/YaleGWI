"""
Training script that uses DataManager for all data IO.
"""
import torch
import random
import numpy as np
import gc  # Add this import for garbage collection
from pathlib import Path
from tqdm import tqdm
import torch.cuda.amp as amp
import logging
import sys
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import json
import boto3
from botocore.exceptions import ClientError
import signal
import time
from src.core.config import CFG
from src.core.data_manager import DataManager
from src.core.model import get_model
from src.core.losses import get_loss_fn
from src.core.setup import push_to_kaggle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def set_seed(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

class SpotInstanceHandler:
    """Handles spot instance interruptions gracefully and manages S3 uploads/cleanup."""
    def __init__(self, checkpoint_dir: Path, s3_upload_interval: int):
        self.checkpoint_dir = checkpoint_dir
        self.s3_upload_interval = s3_upload_interval
        self.last_checkpoint = 0
        self.interrupted = False
        signal.signal(signal.SIGTERM, self.handle_interruption)
        signal.signal(signal.SIGINT, self.handle_interruption)

    def handle_interruption(self, signum, frame):
        logging.info(f"Received signal {signum}, preparing for interruption...")
        self.interrupted = True

    def should_upload_s3(self, epoch: int, is_last_epoch: bool) -> bool:
        return self.interrupted or (epoch % self.s3_upload_interval == 0) or is_last_epoch

    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, metrics, upload_s3=False):
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'metrics': metrics
            }
            ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, ckpt_path)
            local_checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if len(local_checkpoints) > 3:
                for old_ckpt in local_checkpoints[:-3]:
                    old_ckpt.unlink()
                    logging.info(f"Removed old local checkpoint: {old_ckpt}")
            meta_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_metadata.json'
            with open(meta_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'optimizer_state_dict': convert_to_serializable(optimizer.state_dict()),
                    'scheduler_state_dict': convert_to_serializable(scheduler.state_dict()),
                    'scaler_state_dict': convert_to_serializable(scaler.state_dict()),
                    'metrics': metrics,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                }, f, indent=2)
            if upload_s3 and CFG.env.kind == 'aws':
                s3 = boto3.client('s3', region_name=CFG.env.aws_region)
                s3_key = f"checkpoints/checkpoint_epoch_{epoch}.pt"
                s3.upload_file(str(ckpt_path), CFG.env.s3_bucket, s3_key)
                response = s3.list_objects_v2(Bucket=CFG.env.s3_bucket, Prefix='checkpoints/checkpoint_epoch_')
                if 'Contents' in response:
                    checkpoints = sorted(response['Contents'], key=lambda x: x['LastModified'])
                    if len(checkpoints) > 5:
                        for old_ckpt in checkpoints[:-5]:
                            s3.delete_object(Bucket=CFG.env.s3_bucket, Key=old_ckpt['Key'])
                            logging.info(f"Removed old S3 checkpoint: {old_ckpt['Key']}")
            self.last_checkpoint = time.time()
            logging.info(f"Checkpoint saved for epoch {epoch}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            raise

def train(dryrun: bool = False, fp16: bool = True):
    try:
        logging.info("Starting training...")
        set_seed(CFG.seed)
        save_dir = Path('outputs')
        save_dir.mkdir(exist_ok=True)
        spot_handler = SpotInstanceHandler(save_dir, CFG.s3_upload_interval)
        logging.info("Initializing DataManager...")
        data_manager = DataManager(use_mmap=True)
        train_loaders = []
        families = list(CFG.paths.families.keys())
        for family in families:
            seis_files, vel_files, family_type = data_manager.list_family_files(family)
            print(f"Processing family: {family} ({family_type}), #seis: {len(seis_files)}, #vel: {len(vel_files)}")
            loader = data_manager.create_loader(
                seis_files=seis_files,
                vel_files=vel_files,
                family_type=family_type,
                batch_size=CFG.batch,
                shuffle=True,
                num_workers=CFG.num_workers,
                distributed=CFG.distributed
            )
            train_loaders.append(loader)
        model = get_model()
        loss_fn = get_loss_fn()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        total_steps = sum(len(loader) for loader in train_loaders) * CFG.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=CFG.lr,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )
        scaler = GradScaler(enabled=fp16)
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            if hasattr(torch.cuda, 'memory_summary'):
                torch.cuda.memory_summary(device=0)
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        logging.info("Starting training loop...")
        best_mae = float('inf')
        patience = 5
        patience_counter = 0
        min_delta = 1e-4
        for epoch in range(CFG.epochs):
            is_last_epoch = (epoch == CFG.epochs - 1)
            if spot_handler.interrupted:
                logging.info("Spot instance interruption detected, saving checkpoint and exiting...")
                spot_handler.save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch,
                    {'best_mae': best_mae, 'patience_counter': patience_counter},
                    upload_s3=True
                )
                break
            logging.info(f"Epoch {epoch+1}/{CFG.epochs}")
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            for loader_idx, loader in enumerate(train_loaders):
                logging.info(f"Processing loader {loader_idx+1}/{len(train_loaders)}")
                for batch_idx, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
                    try:
                        x = x.cuda()
                        y = y.cuda()
                        with autocast(enabled=fp16):
                            pred = model(x, max_sources_per_step=5)
                            # Log shapes
                            logging.info(f"pred shape: {pred.shape}, y shape: {y.shape}")
                            # Shape matching logic
                            if pred.shape != y.shape:
                                if pred.shape[1] == 1 and y.shape[1] > 1:
                                    y = y.mean(dim=1, keepdim=True)
                                elif y.shape[1] == 1 and pred.shape[1] > 1:
                                    pred = pred.mean(dim=1, keepdim=True)
                            # Assertion
                            assert pred.shape == y.shape, f"Shape mismatch: pred {pred.shape}, y {y.shape}"
                            if torch.isnan(pred).any() or torch.isinf(pred).any():
                                logging.error(f"NaN or Inf detected in model output at batch {batch_idx}")
                                continue
                            loss, loss_components = loss_fn(pred, y)
                            if torch.isnan(loss) or torch.isinf(loss):
                                logging.error(f"NaN or Inf detected in loss at batch {batch_idx}")
                                continue
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        epoch_loss += loss.item()
                        num_batches += 1
                        # Clear cache periodically
                        if batch_idx % 10 == 0:  # More frequent cache clearing
                            torch.cuda.empty_cache()
                            if torch.cuda.is_available():
                                # Log memory stats
                                allocated = torch.cuda.memory_allocated() / 1e9
                                reserved = torch.cuda.memory_reserved() / 1e9
                                logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                                # Force garbage collection
                                gc.collect()
                                # Reset peak memory stats
                                torch.cuda.reset_peak_memory_stats()
                                # Empty cache again after GC
                                torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logging.error(f"OOM Error at epoch {epoch+1}, loader {loader_idx+1}, batch {batch_idx}")
                            logging.error(f"Last successful operation: Forward pass")
                            if torch.cuda.is_available():
                                logging.error(f"GPU Memory at error: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
                            # Clear cache and continue
                            torch.cuda.empty_cache()
                            gc.collect()  # Add garbage collection
                            continue
                        else:
                            raise e
                    except (AttributeError, NameError) as e:
                        logging.error(f"{type(e).__name__} encountered at batch {batch_idx}: {e}")
                        raise e
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            logging.info(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            logging.info("Starting validation...")
            model.eval()
            val_mae = 0.0
            val_loss = 0.0
            num_val_batches = 0
            try:
                with torch.no_grad():
                    val_loader = train_loaders[0]
                    for batch_idx, (seis, vel) in enumerate(val_loader):
                        seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
                        vel = vel.mean(dim=1, keepdim=True)
                        logging.info(f"vel shape: {vel.shape}, dtype: {vel.dtype}, min: {vel.min().item()}, max: {vel.max().item()}, mean: {vel.mean().item()}")
                        with autocast(enabled=fp16):
                            v_pred = model.get_ema_model()(seis)
                            # Log shapes
                            logging.info(f"v_pred shape: {v_pred.shape}, vel shape: {vel.shape}")
                            # Shape matching logic
                            if v_pred.shape != vel.shape:
                                if v_pred.shape[1] == 1 and vel.shape[1] > 1:
                                    vel = vel.mean(dim=1, keepdim=True)
                                elif vel.shape[1] == 1 and v_pred.shape[1] > 1:
                                    v_pred = v_pred.mean(dim=1, keepdim=True)
                            # Assertion
                            assert v_pred.shape == vel.shape, f"Shape mismatch: v_pred {v_pred.shape}, vel {vel.shape}"
                            if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
                                logging.error(f"NaN or Inf detected in validation output at batch {batch_idx}")
                                continue
                            val_mae += torch.nn.functional.l1_loss(v_pred, vel).item()
                            val_loss += loss_fn(v_pred, vel)[0].item()
                            num_val_batches += 1
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                    val_mae /= num_val_batches
                    val_loss /= num_val_batches
                    # Save checkpoint if needed
                    if spot_handler.should_upload_s3(epoch, is_last_epoch):
                        spot_handler.save_checkpoint(
                            model, optimizer, scheduler, scaler, epoch,
                            {
                                'val_mae': val_mae,
                                'val_loss': val_loss,
                                'avg_epoch_loss': avg_epoch_loss,
                                'best_mae': best_mae,
                                'patience_counter': patience_counter
                            },
                            upload_s3=True
                        )
                    if val_mae < best_mae - min_delta:
                        best_mae = val_mae
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                        }
                        torch.save(checkpoint, save_dir/'best.pth')
                        logging.info(f"New best model saved! MAE: {val_mae:.4f}")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logging.info(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    logging.info(f"Epoch {epoch+1} complete - val_mae = {val_mae:.4f}, val_loss = {val_loss:.4f}")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"OOM Error during validation at epoch {epoch+1}")
                    logging.error(f"Last successful operation: Validation batch {batch_idx}")
                    if torch.cuda.is_available():
                        logging.error(f"GPU Memory at error: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except (AttributeError, NameError) as e:
                logging.error(f"{type(e).__name__} encountered during validation at batch {batch_idx}: {e}")
                raise e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU cache cleared")
    except Exception as e:
        logging.error(f"Uncaught exception in train(): {type(e).__name__}: {e}")
        raise

if __name__ == '__main__':
    train() 