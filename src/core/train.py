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

def set_seed(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(dryrun: bool = False, fp16: bool = True):
    set_seed(CFG.seed)
    save_dir = Path('outputs')
    save_dir.mkdir(exist_ok=True)
    
    # Initialize DataManager with memory tracking
    data_manager = DataManager(use_mmap=True)
    
    # Get training files for each family
    train_loaders = []
    for family in CFG.families:
        seis_files, vel_files, family_type = data_manager.list_family_files(family)
        loader = data_manager.create_loader(
            seis_files=seis_files,
            vel_files=vel_files,
            family_type=family_type,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            distributed=CFG.distributed
        )
        train_loaders.append(loader)
    
    # Initialize model and loss
    model = get_model()
    loss_fn = get_loss_fn()
    
    # Training loop
    for epoch in range(CFG.epochs):
        for loader in train_loaders:
            for batch_idx, (x, y) in enumerate(loader):
                # Move to GPU if available
                x = x.cuda()
                y = y.cuda()
                
                # Forward pass
                pred = model(x)
                loss = loss_fn(pred, y)
                
                # Backward pass
                loss.backward()
                
                # Log memory stats periodically
                if batch_idx % 100 == 0:
                    memory_stats = data_manager.memory_tracker.get_stats()
                    logging.info(f"Memory stats: {memory_stats}")
                
                # Rest of training loop...
                
        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for seis, vel in train_loader:  # Using train set for now
                seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
                v_pred = model.get_ema_model()(seis)
                val_mae += torch.nn.functional.l1_loss(v_pred, vel).item()
        val_mae /= len(train_loader)
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_dir/'best.pth')
        torch.save(model.state_dict(), save_dir/'last.pth')
        
        print(f"Epoch {epoch}: val_mae = {val_mae:.4f}")
        
        # Clear cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    train() 