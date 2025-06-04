# %% [markdown]
# ## Training

# %%
import torch, random, numpy as np
from config import CFG, save_cfg
from data_manager import DataManager
from model import SpecProjNet
from losses import JointLoss
from pathlib import Path
from tqdm import tqdm
import argparse
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp

def set_seed(seed:int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def train(rank, world_size, dryrun: bool = False, fp16: bool = True, empty_cache: bool = True):
    # Initialize distributed training
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    set_seed(CFG.seed)
    save_dir = Path('outputs'); save_dir.mkdir(exist_ok=True)
    
    # Initialize data manager
    data_manager = DataManager(use_mmap=True, cache_size=1000)

    # ----------- data ----------------------------------------------------- #
    train_seis, train_vel = [], []
    for fam in ('FlatVel_A','FlatVel_B'):
        s, v = data_manager.list_family_files(fam)
        # If dryrun and files are missing, mock data to avoid NoneType error
        if dryrun and (not s or not v):
            # Create mock data for dryrun
            import torch
            train_seis = [torch.randn(1, 1, 100, 100)]
            train_vel = [torch.randn(1, 1, 70, 70)]
            break
        train_seis += s if s else []
        train_vel += v if v else []
    if dryrun and (not train_seis or not train_vel):
        # If still empty, create mock data
        import torch
        train_seis = [torch.randn(1, 1, 100, 100)]
        train_vel = [torch.randn(1, 1, 70, 70)]
    train_loader = data_manager.create_loader(
        train_seis, train_vel, 
        batch_size=CFG.batch, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ----------- model/optim --------------------------------------------- #
    model = SpecProjNet(
        backbone=CFG.backbone,
        pretrained=True,
        ema_decay=CFG.ema_decay,
    ).to(CFG.env.device)
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
        
    opt = torch.optim.AdamW(model.parameters(),
                           lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = amp.GradScaler(enabled=fp16)
    loss_fn = JointLoss(λ_inv=CFG.lambda_inv,
                       λ_fwd=CFG.lambda_fwd,
                       λ_pde=CFG.lambda_pde)

    best_mae = 1e9
    for epoch in range(CFG.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for seis, vel in pbar:
            seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
            
            with amp.autocast(enabled=fp16):
                v_pred = model(seis)
                loss, loss_dict = loss_fn(v_pred, vel, seis_batch=seis)
                    
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            
            # Update EMA
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.update_ema()
            else:
                model.update_ema()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': float(loss),
                **loss_dict,
                'mem': f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            })
            
            if dryrun:
                break
                
        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for seis, vel in train_loader:  # Using train set for now
                seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
                # Use EMA model for validation
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    v_pred = model.module.get_ema_model()(seis)
                else:
                    v_pred = model.get_ema_model()(seis)
                val_mae += F.l1_loss(v_pred, vel).item()
        val_mae /= len(train_loader)
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            if rank == 0:  # Only save on main process
                torch.save(model.state_dict(), save_dir/'best.pth')
        if rank == 0:  # Only save on main process
            torch.save(model.state_dict(), save_dir/'last.pth')
        
        print(f"Epoch {epoch}: val_mae = {val_mae:.4f}")
        
        # Clear cache periodically
        data_manager.clear_cache()
        if empty_cache:
            torch.cuda.empty_cache()
            
    if world_size > 1:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dryrun', action='store_true', help='Train one minibatch only')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable mixed precision training')
    parser.add_argument('--empty_cache', action='store_true', default=True, help='Clear CUDA cache after each epoch')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    args = parser.parse_args()
    
    if args.world_size > 1:
        mp.spawn(
            train,
            args=(args.world_size, args.dryrun, args.fp16, args.empty_cache),
            nprocs=args.world_size,
            join=True
        )
    else:
        train(0, 1, args.dryrun, args.fp16, args.empty_cache)

if __name__ == '__main__':
    main() 