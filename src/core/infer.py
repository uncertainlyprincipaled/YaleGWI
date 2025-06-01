# %% [markdown]
# ## Inference

# %%
import torch, pandas as pd
from pathlib import Path
from config import CFG
from data_utils import list_family_files, make_loader
from specproj_hybrid import HybridSpecProj

def format_submission(vel_map, oid):
    # keep *odd* x-columns only (1,3,…,69)
    cols = vel_map[:,1::2]
    rows = []
    for y,row in enumerate(cols):
        rows.append({'oid_ypos': f'{oid}_y_{y}', **{f'x_{2*i+1}':v
                      for i,v in enumerate(row.tolist())}})
    return rows

def infer(weights='outputs/last.pth'):
    model = HybridSpecProj().to(CFG.env.device)
    model.load_state_dict(torch.load(weights, map_location=CFG.env.device))
    model.eval()

    test_files = sorted(CFG.paths.test.glob('*.npy'))
    test_loader = make_loader(test_files, vel=None,
                             batch=1, shuffle=False)

    rows = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for seis, oid in test_loader:
            seis = seis.to(CFG.env.device)
            # Always use inverse mode for inference
            vel, _ = model(seis, mode="inverse")
            vel = vel.cpu().float().numpy()[0,0]
            rows += format_submission(vel, Path(oid[0]).stem)
    pd.DataFrame(rows).to_csv('submission.csv', index=False)

if __name__ == '__main__':
    infer() 