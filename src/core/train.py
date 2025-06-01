# %% [markdown]
# ## Training

# %%
import torch, random, numpy as np
from config import CFG, save_cfg
from data_utils import list_family_files, make_loader
from specproj_unet import SpecProjUNet
from losses import HybridLoss
from pathlib import Path
from tqdm import tqdm
import argparse

def set_seed(seed:int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def train(dryrun: bool = False):
    set_seed(CFG.seed)
    save_dir = Path('outputs'); save_dir.mkdir(exist_ok=True)
    save_cfg(save_dir)

    # ----------- data ----------------------------------------------------- #
    train_seis, train_vel = [], []
    for fam in ('FlatVel_A','FlatVel_B'):
        s, v = list_family_files(fam)
        train_seis += s; train_vel += v
    train_loader = make_loader(train_seis, train_vel, CFG.batch, shuffle=True)

    # ----------- model/optim --------------------------------------------- #
    model = SpecProjUNet().to(CFG.env.device)
    if CFG.env.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)
    opt    = torch.optim.AdamW(model.parameters(),
                               lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn= HybridLoss()

    best_mae = 1e9
    for epoch in range(CFG.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for seis, vel in pbar:
            seis, vel = seis.to(CFG.env.device), vel.to(CFG.env.device)
            with torch.amp.autocast('cuda'):
                pred = model(seis)
                loss = loss_fn(pred, vel, seis)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            pbar.set_postfix(loss=float(loss))
            if dryrun:
                break
        # TODO: add validation & MAE tracking
        torch.save(model.state_dict(), save_dir/'last.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--dryrun', action='store_true', help='Train one minibatch only')
    args = parser.parse_args()
    train(args.dryrun) 