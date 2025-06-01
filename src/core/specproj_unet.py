# %% [markdown]
# ## Model Architecture - UNet

# %%
import torch, torch.nn as nn, torch.nn.functional as F
from proj_mask import PhysMask
from config import CFG

# ---- Minimal residual-UNet building blocks ------------------------------- #

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in , ch_out, 3, padding=1)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.bn1, self.bn2 = nn.BatchNorm2d(ch_out), nn.BatchNorm2d(ch_out)
        self.skip = (nn.Identity() if ch_in==ch_out
                     else nn.Conv2d(ch_in, ch_out, 1))
    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x))

class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.res = ResBlock(ch_in, ch_out)
    def forward(self,x): return self.res(self.mp(x))

class Up(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up  = nn.ConvTranspose2d(ch_in, ch_out, 2, stride=2)
        self.res = ResBlock(ch_in, ch_out)
    def forward(self,x, skip):
        x = torch.cat([self.up(x), skip], 1)
        return self.res(x)

# ---- SpecProj-UNet ------------------------------------------------------- #

class SmallUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=32, depth=4):
        super().__init__()
        self.inc  = ResBlock(in_ch, base)
        self.down = nn.ModuleList([Down(base*2**i, base*2**(i+1))
                                   for i in range(depth)])
        self.up   = nn.ModuleList([Up(base*2**(i+1), base*2**i)
                                   for i in reversed(range(depth))])
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self,x):
        skips=[self.inc(x)]
        for d in self.down: skips.append( d(skips[-1]) )
        x = skips.pop()
        for u in self.up:   x = u(x, skips.pop())
        return self.outc(x)

class SpecProjUNet(nn.Module):
    def __init__(self, chans=32, depth=4):
        super().__init__()
        self.mask = PhysMask()
        self.unet_up   = SmallUNet(in_ch=5, out_ch=1, base=chans, depth=depth)
        self.unet_down = SmallUNet(in_ch=5, out_ch=1, base=chans, depth=depth)
        self.fuse = nn.Conv2d(2, 1, 1)

    def forward(self, x):            # x (B,S,T,R)
        up, down = self.mask(x)      # still (B,S,T,R)
        # merge source dim into batch for UNet (expect 5 channels)
        B,S,T,R = up.shape
        up   = up  .reshape(B*S, 1, T, R).repeat(1,5,1,1)
        down = down.reshape(B*S, 1, T, R).repeat(1,5,1,1)
        vu  = self.unet_up (up )
        vd  = self.unet_down(down)
        fused = self.fuse(torch.cat([vu, vd], 1))
        return fused.view(B, S, 1, *fused.shape[-2:]).mean(1)  # (B,1,H,W) 