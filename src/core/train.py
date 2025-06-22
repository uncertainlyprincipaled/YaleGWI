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
import torch.nn as nn
import shutil

# Local imports - ensure these are at the top level
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

def save_checkpoint(model, optimizer, scheduler, epoch, loss, data_manager):
    logging.info(f"Starting save_checkpoint for epoch {epoch} with loss {loss}")
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    try:
        torch.save(checkpoint, out_dir / 'checkpoint.pth')
        logging.info(f"Checkpoint saved at outputs/checkpoint.pth for epoch {epoch}")
        best_path = out_dir / 'best.pth'
        is_best = False
        if not best_path.exists():
            is_best = True
        else:
            try:
                prev_checkpoint = torch.load(best_path)
                prev_loss = prev_checkpoint.get('loss', float('inf'))
                is_best = loss < prev_loss
            except Exception as e:
                logging.warning(f"Could not load previous checkpoint: {e}")
                is_best = True
        if is_best:
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved at outputs/best.pth with loss: {loss:.4f}")
            metadata = {
                'epoch': epoch,
                'loss': float(loss),
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'batch_size': CFG.batch,
                    'learning_rate': CFG.lr,
                    'weight_decay': CFG.weight_decay,
                    'backbone': CFG.backbone,
                }
            }
            with open(out_dir / 'best_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Best model metadata saved at outputs/best_metadata.json")
            print("Uploading best model and metadata to S3")
            data_manager.upload_best_model_and_metadata(
                out_dir,
                f"Update best model - loss: {loss:.4f} at epoch {epoch}"
            )
        logging.info(f"save_checkpoint completed for epoch {epoch}")
    except Exception as e:
        logging.error(f"Exception in save_checkpoint: {e}")
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

def train(dryrun: bool = False, fp16: bool = False, use_geometric: bool = True):
    """Main training loop with optional geometric loading support."""
    # Initialize data manager
    data_manager = DataManager()
    
    # Initialize Phase 1 components if using geometric loading
    if use_geometric:
        print("Initializing Phase 1 geometric components...")
        try:
            from src.core.geometric_loader import FamilyDataLoader
            from src.core.geometric_cv import GeometricCrossValidator
            from src.core.registry import ModelRegistry
            from src.core.checkpoint import CheckpointManager
            
            # Initialize components
            registry = ModelRegistry()
            checkpoint_mgr = CheckpointManager()
            cv = GeometricCrossValidator(n_splits=5)
            print("✅ Phase 1 components initialized")
            
            # Try to use geometric loading
            try:
                # Check if preprocessed data exists using config paths
                preprocessed_paths = CFG.paths.preprocessed_paths
                
                data_root = None
                for path in preprocessed_paths:
                    if path.exists():
                        data_root = path
                        break
                
                if data_root:
                    print(f"Using preprocessed data from: {data_root}")
                    
                    # Check data structure compatibility
                    structure_check = CFG.check_preprocessed_data_structure(data_root)
                    print(f"Data structure: {structure_check['current_structure']}")
                    
                    if not structure_check['geometric_compatible']:
                        print("⚠️ Data structure not compatible with geometric loading:")
                        for issue in structure_check['issues']:
                            print(f"  - {issue}")
                        print("Falling back to original loading")
                        use_geometric_loader = False
                    else:
                        family_loader = FamilyDataLoader(
                            data_root=str(data_root),
                            batch_size=CFG.batch,
                            num_workers=CFG.num_workers,
                            extract_features=True
                        )
                        print("✅ Geometric data loader initialized")
                        use_geometric_loader = True
                else:
                    print("⚠️ Preprocessed data not found in any expected location:")
                    for path in preprocessed_paths:
                        print(f"  - {path} (exists: {path.exists()})")
                    print("Falling back to original loading")
                    use_geometric_loader = False
                    
            except Exception as e:
                print(f"⚠️ Geometric loading failed: {e}, falling back to original loading")
                use_geometric_loader = False
                
        except ImportError as e:
            print(f"⚠️ Phase 1 components not available: {e}, using original loading")
            use_geometric_loader = False
    else:
        use_geometric_loader = False
    
    # Create loaders based on available method
    if use_geometric_loader:
        # Use geometric loading
        train_loaders = []
        families = list(CFG.paths.families.keys())
        
        for family in families:
            if CFG.debug_mode and family in getattr(CFG, 'families_to_exclude', []):
                continue
                
            try:
                loader = family_loader.get_loader(family)
                if loader is not None:
                    train_loaders.append(loader)
                    print(f"✅ Loaded geometric data for family: {family}")
            except Exception as e:
                print(f"⚠️ Failed to load geometric data for {family}: {e}")
                continue
                
    else:
        # Use original loading method
        train_loaders = []
        for family in CFG.paths.families:
            if CFG.debug_mode and family in getattr(CFG, 'families_to_exclude', []):
                continue
                
            seis_files, vel_files, family_type = data_manager.list_family_files(family)
            if seis_files is None:  # Skip excluded families
                continue
                
            print(f"Processing family: {family} ({family_type}), #seis: {len(seis_files)}, #vel: {len(vel_files)}")
            
            loader = data_manager.create_loader(
                seis_files, vel_files, family_type,
                batch_size=CFG.batch,
                shuffle=True,
                num_workers=CFG.num_workers
            )
            if loader is not None:
                train_loaders.append(loader)
            
    if not train_loaders:
        raise ValueError("No valid data loaders created!")
        
    print(f"✅ Created {len(train_loaders)} data loaders")
        
    # Initialize model and move to device
    model = get_model()
    model.train()
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.epochs,
        eta_min=CFG.lr/100
    )
    
    # Initialize loss function
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(CFG.epochs):
        total_loss = 0
        total_batches = 0
        
        for loader in train_loaders:
            for batch_idx, batch in enumerate(loader):
                try:
                    # Handle different batch formats
                    if use_geometric_loader:
                        # Geometric loading format
                        if isinstance(batch, dict):
                            x = batch['data']
                            # For now, create dummy target (TODO: integrate velocity data)
                            y = torch.zeros(x.shape[0], 1, 70, 70, device=x.device)
                        else:
                            x, y = batch
                    else:
                        # Original loading format
                        x, y = batch
                    
                    # Move data to device and convert to float32
                    x = x.to(CFG.env.device).float()
                    y = y.to(CFG.env.device).float()
                    
                    # Forward pass
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    total_loss += loss.item()
                    total_batches += 1
                    
                    # Print progress
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{CFG.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        print(f"WARNING: out of memory in batch {batch_idx}. Skipping batch.")
                        continue
                    else:
                        raise e
                        
        # Update learning rate
        scheduler.step()
        
        # Print epoch statistics
        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{CFG.epochs} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint using appropriate method
        if (epoch + 1) % CFG.debug_upload_interval == 0:
            if use_geometric_loader and 'checkpoint_mgr' in locals():
                # Use Phase 1 checkpoint manager
                checkpoint_mgr.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,  # Not using scaler in this version
                    epoch=epoch,
                    metrics={'loss': avg_loss},
                    upload_s3=False  # Can be made configurable
                )
            else:
                # Use original checkpoint method
                save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, data_manager)
            
    # After all epochs:
    # Save best.pth and best_metadata.json to /kaggle/working for persistence
    kaggle_working = Path('/kaggle/working')
    kaggle_working.mkdir(parents=True, exist_ok=True)
    out_dir = Path('outputs')
    for fname in ['best.pth', 'best_metadata.json']:
        src = out_dir / fname
        dst = kaggle_working / fname
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"File {src} does not exist, skipping copy.")
    print("Uploading model to Kaggle dataset (final upload)...")
    push_to_kaggle(out_dir, "Final upload after training")
    print("Upload complete.")

    return model

if __name__ == '__main__':
    # Set debug mode to True and update settings
    CFG.set_debug_mode(True)
    train() 