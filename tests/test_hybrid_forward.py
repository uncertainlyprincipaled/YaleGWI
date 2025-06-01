import torch
import pytest
from src.core.specproj_hybrid import HybridSpecProj
from src.core.config import CFG

def test_hybrid_forward():
    # Create test data
    batch_size = 2
    sources = 1
    time_steps = 100
    receivers = 100
    seis = torch.randn(batch_size, sources, time_steps, receivers)
    
    # Test inverse-only mode
    CFG.enable_joint = False
    model = HybridSpecProj()
    v_pred, p_pred = model(seis, mode="inverse")
    
    # Check shapes
    assert v_pred.shape == (batch_size, 1, 70, 70)  # Velocity map size
    assert p_pred is None
    
    # Test joint mode
    CFG.enable_joint = True
    model = HybridSpecProj()
    v_pred, p_pred = model(seis, mode="joint")
    
    # Check shapes
    assert v_pred.shape == (batch_size, 1, 70, 70)  # Velocity map size
    assert p_pred.shape == (batch_size, 1, time_steps, receivers)  # Wavefield size
    
    # Test invalid mode
    with pytest.raises(ValueError):
        model(seis, mode="invalid") 