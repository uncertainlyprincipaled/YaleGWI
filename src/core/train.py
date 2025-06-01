# %% [markdown]
# ## Training

# %%
import torch, random, numpy as np
from config import CFG, save_cfg
from data_utils import list_family_files, make_loader
from specproj_hybrid import HybridSpecProj
from losses import JointLoss
from pathlib import Path
from tqdm import tqdm
import argparse
import torch.cuda.amp as amp

def set_seed(seed:int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def train(dryrun: bool = False, enable_joint: bool = False):
    set_seed(CFG.seed)
    save_dir = Path('outputs'); save_dir.mkdir(exist_ok=True)
    
    # Update config
    CFG.enable_joint = enable_joint
    save_cfg(save_dir)

    # ----------- data ----------------------------------------------------- #
    train_seis, train_vel = [], []
    for fam in ('FlatVel_A','FlatVel_B'):
        s, v = list_family_files(fam)
        train_seis += s; train_vel += v
    train_loader = make_loader(train_seis, train_vel, CFG.batch, shuffle=True)

    # ----------- model/optim --------------------------------------------- #
    model = HybridSpecProj().to(CFG.env.device)
    if CFG.env.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(model.parameters(),
                           lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = amp.GradScaler()
    loss_fn = JointLoss(λ_inv=CFG.lambda_inv,
                       λ_fwd=CFG.lambda_fwd,
                       λ_pde=CFG.lambda_pde)

    best_mae = 1e9
    for epoch in range(CFG.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for seis, vel in pbar:
            seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
            
            with amp.autocast():
                if CFG.is_joint():
                    v_pred, p_pred = model(seis, mode="joint")
                    loss, loss_dict = loss_fn(v_pred, vel, p_pred, seis, seis)
                else:
                    v_pred, _ = model(seis, mode="inverse")
                    loss, loss_dict = loss_fn(v_pred, vel, seis_batch=seis)
                    
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            
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
                v_pred, _ = model(seis, mode="inverse")
                val_mae += F.l1_loss(v_pred, vel).item()
        val_mae /= len(train_loader)
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_dir/'best.pth')
        torch.save(model.state_dict(), save_dir/'last.pth')
        
        print(f"Epoch {epoch}: val_mae = {val_mae:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dryrun', action='store_true', help='Train one minibatch only')
    parser.add_argument('--enable_joint', action='store_true', help='Enable joint forward-inverse training')
    args = parser.parse_args()
    train(args.dryrun, args.enable_joint) 