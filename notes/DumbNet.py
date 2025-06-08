# DumbNet - Simple UNet

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

# Find files to load and Create Dataset

all_inputs = [
    f
    for f in
    Path('/kaggle/input/waveform-inversion/train_samples').rglob('*.npy')
    if ('seis' in f.stem) or ('data' in f.stem)
]

def inputs_files_to_output_files(input_files):
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]

all_outputs = inputs_files_to_output_files(all_inputs)

assert all(f.exists() for f in all_outputs)

train_inputs = [all_inputs[i] for i in range(0, len(all_inputs), 2)] # Sample every two
valid_inputs = [f for f in all_inputs if not f in train_inputs]

train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)

class SeismicDataset(Dataset):
    def __init__(self, inputs_files, output_files, n_examples_per_file=500):
        assert len(inputs_files) == len(output_files)
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file

    def __len__(self):
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx):
        # Calculate file offset and sample offset within file
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        X = np.load(self.inputs_files[file_idx], mmap_mode='r')
        y = np.load(self.output_files[file_idx], mmap_mode='r')

        try:
            return X[sample_idx].copy(), y[sample_idx].copy()
        finally:
            del X, y

dstrain = SeismicDataset(train_inputs, train_outputs)
dltrain = DataLoader(dstrain, batch_size=64, shuffle=True, pin_memory=True, drop_last=True, num_workers=4, persistent_workers=True)

dsvalid = SeismicDataset(valid_inputs, valid_outputs)
dlvalid = DataLoader(dsvalid, batch_size=64, shuffle=False, pin_memory=True, drop_last=False, num_workers=4, persistent_workers=True)

# DumbNet

class DumbNet(nn.Module):
    '''DumbNet is just a MLP Model, with a avg-pool first to reduze input size'''
    def __init__(self, pool_size=(8, 2), input_size=5 * 1000 * 70, hidden_size=70 * 70, output_size=70 * 70):
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=pool_size)

        self.model = nn.Sequential(
            nn.Linear(input_size // (pool_size[0] * pool_size[1]), hidden_size),
            
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        bs = x.shape[0]

        # We apply a pool to reduze input size
        x_pool = self.pool(x)

        #Model is just a
        out = self.model(x_pool.view(bs, -1))

        return out.view(bs, 1, 70, 70) * 1000 + 1500
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
model = DumbNet().to(device)

# Train Loop
criterion = nn.L1Loss()
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

n_epochs = 50

history = []

for epoch in range(1, n_epochs+1):
    print(f'[{epoch:02d}] Begin train')

    # Train
    model.train()
    train_losses = []
    for inputs, targets in tqdm(dltrain, desc='train', leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optim.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        optim.step()

        train_losses.append(loss.item())

    print('Train loss: {:.5f}'.format( np.mean(train_losses) ))

    # Valid
    model.eval()
    valid_losses = []
    for inputs, targets in tqdm(dlvalid, desc='valid', leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.inference_mode():
            outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        valid_losses.append(loss.item())
    
    print('Valid loss: {:.5f}'.format( np.mean(valid_losses)) )
    history.append({
        'train': np.mean(train_losses),
        'valid': np.mean(valid_losses)
    })

    # Plot last result
    if epoch % 4 == 0:
        y = targets[0, 0].detach().cpu()
        y_pred = outputs[0, 0].detach().cpu()
        
        fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
        fig.suptitle(f'Epoch {epoch} | Valid: {np.mean(valid_losses):.5f}')
        ax[0].imshow(y)
        ax[1].imshow(y_pred)
        plt.show()

pd.DataFrame(history).plot();

# Predict Test
import csv  # Use "low-level" CSV to save memory on predictions

# %%time
test_files = list(Path('/kaggle/input/waveform-inversion/test').glob('*.npy'))
len(test_files)

x_cols = [f'x_{i}' for i in range(1, 70, 2)]
fieldnames = ['oid_ypos'] + x_cols

class TestDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files


    def __len__(self):
        return len(self.test_files)


    def __getitem__(self, i):
        test_file = self.test_files[i]

        return np.load(test_file), test_file.stem
    
ds = TestDataset(test_files)
dl = DataLoader(ds, batch_size=8, num_workers=4, pin_memory=True)

# Train
model.eval()
with open('submission.csv', 'wt', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for inputs, oids_test in tqdm(dl, desc='test'):
        inputs = inputs.to(device)
        with torch.inference_mode():
            outputs = model(inputs)

        y_preds = outputs[:, 0].cpu().numpy()
        
        for y_pred, oid_test in zip(y_preds, oids_test):
            for y_pos in range(70):
                row = dict(
                    zip(
                        x_cols,
                        [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]
                    )
                )
                row['oid_ypos'] = f"{oid_test}_y_{y_pos}"
            
                writer.writerow(row)