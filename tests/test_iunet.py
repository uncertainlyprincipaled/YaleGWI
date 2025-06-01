import torch
import pytest
from src.core.iunet import IUNet, create_iunet

def test_iunet_creation():
    model = create_iunet()
    assert isinstance(model, IUNet)
    assert model.in_channels == 1
    assert model.hidden_channels == 64
    assert len(model.couplings) == 4

def test_iunet_forward():
    model = create_iunet()
    batch_size = 2
    height = 16
    width = 16
    
    # Test forward pass
    x = torch.randn(batch_size, 1, height, width)
    y_p2v = model(x, "p→v")
    y_v2p = model(x, "v→p")
    
    assert y_p2v.shape == (batch_size, 1, height, width)
    assert y_v2p.shape == (batch_size, 1, height, width)
    
    # Test invalid direction
    with pytest.raises(ValueError):
        model(x, "invalid")

def test_iunet_invertibility():
    model = create_iunet()
    x = torch.randn(2, 1, 16, 16)
    
    # Forward and inverse should approximately recover input
    y = model(x, "p→v")
    x_recon = model(y, "v→p")
    
    # Check reconstruction error
    error = torch.abs(x - x_recon).mean()
    assert error < 1e-3, f"Reconstruction error {error} too high" 