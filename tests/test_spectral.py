import torch
import pytest
from src.core.proj_mask import PhysMask, SpectralAssembler, split_and_reassemble

def test_spectral_assembler():
    # Create test data
    batch_size = 2
    sources = 1
    time_steps = 100
    receivers = 100
    x = torch.randn(batch_size, sources, time_steps, receivers)
    
    # Create components
    mask = PhysMask()
    assembler = SpectralAssembler()
    
    # Test reconstruction
    x_recon = split_and_reassemble(x, mask, assembler)
    
    # Check shapes
    assert x_recon.shape == x.shape
    
    # Check reconstruction error
    rel_error = torch.norm(x_recon - x) / torch.norm(x)
    assert rel_error < 1e-3, f"Reconstruction error {rel_error} too high"
    
    # Test with different batch sizes
    x = torch.randn(4, 2, time_steps, receivers)
    x_recon = split_and_reassemble(x, mask, assembler)
    assert x_recon.shape == x.shape
    rel_error = torch.norm(x_recon - x) / torch.norm(x)
    assert rel_error < 1e-3 