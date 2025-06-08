"""
Training script that uses DataManager for all data IO.
"""
import torch, random, numpy as np
from config import CFG
from data_manager import DataManager
from model import get_model
from losses import get_loss_fn
from pathlib import Path
from tqdm import tqdm
import torch.cuda.amp as amp
import logging
import sys

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

def train(dryrun: bool = False, fp16: bool = True):
    try:
        logging.info("Starting training...")
        set_seed(CFG.seed)
        save_dir = Path('outputs')
        save_dir.mkdir(exist_ok=True)
        
        # Initialize DataManager with memory tracking
        logging.info("Initializing DataManager...")
        data_manager = DataManager(use_mmap=True)
        
        # Get balanced training files for each family
        logging.info("Setting up balanced data loaders...")
        train_loaders = []
        balanced_families = data_manager.get_balanced_family_files(target_count=1000)
        for family, (seis_files, vel_files, family_type) in balanced_families.items():
            logging.info(f"Processing family: {family}")
            loader = data_manager.create_loader(
                seis_files=seis_files,
                vel_files=vel_files,
                family_type=family_type,
                batch_size=2,  # Reduced batch size
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                distributed=CFG.distributed
            )
            train_loaders.append(loader)
        
        # Initialize model and loss
        logging.info("Initializing model and loss function...")
        model = get_model()
        loss_fn = get_loss_fn()
        
        # Initialize optimizer and scaler for mixed precision
        logging.info("Setting up optimizer and mixed precision training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=fp16)
        
        # Enable memory optimization
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
        
        # Training loop
        logging.info("Starting training loop...")
        best_mae = float('inf')
        for epoch in range(CFG.epochs):
            logging.info(f"Epoch {epoch+1}/{CFG.epochs}")
            model.train()
            for loader_idx, loader in enumerate(train_loaders):
                logging.info(f"Processing loader {loader_idx+1}/{len(train_loaders)}")
                for batch_idx, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
                    try:
                        # Move to GPU if available
                        x = x.cuda()
                        y = y.cuda()
                        
                        # Forward pass with mixed precision
                        with torch.amp.autocast('cuda', enabled=fp16):
                            pred = model(x)
                            # Check for NaNs/Infs in model output
                            if torch.isnan(pred).any() or torch.isinf(pred).any():
                                logging.error(f"NaN or Inf detected in model output at batch {batch_idx}")
                                continue
                            loss, loss_components = loss_fn(pred, y)
                            # Check for NaNs/Infs in loss
                            if torch.isnan(loss) or torch.isinf(loss):
                                logging.error(f"NaN or Inf detected in loss at batch {batch_idx}")
                                continue
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        # Clear cache periodically
                        if batch_idx % 10 == 0:  # More frequent cache clearing
                            torch.cuda.empty_cache()
                        
                        # Log memory stats periodically
                        if batch_idx % 100 == 0:
                            memory_stats = data_manager.memory_tracker.get_stats()
                            logging.info(f"Batch {batch_idx} - Memory stats: {memory_stats}")
                            if torch.cuda.is_available():
                                logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logging.error(f"OOM Error at epoch {epoch+1}, loader {loader_idx+1}, batch {batch_idx}")
                            logging.error(f"Last successful operation: Forward pass")
                            if torch.cuda.is_available():
                                logging.error(f"GPU Memory at error: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
                            # Clear cache and continue
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                    except (AttributeError, NameError) as e:
                        logging.error(f"{type(e).__name__} encountered at batch {batch_idx}: {e}")
                        raise e
            
            # Validation
            logging.info("Starting validation...")
            model.eval()
            val_mae = 0.0
            try:
                with torch.no_grad():
                    # Use the first train_loader for validation (as a placeholder)
                    val_loader = train_loaders[0]
                    for batch_idx, (seis, vel) in enumerate(val_loader):
                        seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
                        with torch.amp.autocast('cuda', enabled=fp16):
                            v_pred = model.get_ema_model()(seis)
                            # Check for NaNs/Infs in validation output
                            if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
                                logging.error(f"NaN or Inf detected in validation output at batch {batch_idx}")
                                continue
                        val_mae += torch.nn.functional.l1_loss(v_pred, vel).item()
                        
                        # Clear cache periodically during validation
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                    
                val_mae /= len(val_loader)
                
                # Save best model
                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(model.state_dict(), save_dir/'best.pth')
                    logging.info(f"New best model saved! MAE: {val_mae:.4f}")
                torch.save(model.state_dict(), save_dir/'last.pth')
                
                logging.info(f"Epoch {epoch+1} complete - val_mae = {val_mae:.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error(f"OOM Error during validation at epoch {epoch+1}")
                    logging.error(f"Last successful operation: Validation batch {batch_idx}")
                    if torch.cuda.is_available():
                        logging.error(f"GPU Memory at error: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
                    # Clear cache and continue
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except (AttributeError, NameError) as e:
                logging.error(f"{type(e).__name__} encountered during validation at batch {batch_idx}: {e}")
                raise e
            
            # Clear cache at end of epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU cache cleared")
    except Exception as e:
        logging.error(f"Uncaught exception in train(): {type(e).__name__}: {e}")
        raise

if __name__ == '__main__':
    train() 