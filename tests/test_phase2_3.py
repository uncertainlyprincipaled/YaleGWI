import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import Phase 2 and 3 components
from core.egnn import EGNN, ReceiverGeometryEncoder
from core.heat_kernel import HeatKernelLayer, WaveEquationApproximator, PhysicsGuidedDiffusion
from core.se3_transformer import SE3Transformer, SE3WavefieldProcessor
from core.ensemble import StaticEnsemble, DynamicEnsemble, BayesianEnsemble, GeometricEnsemble, EnsembleManager
from core.crf import ContinuousCRF, CRFPostProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase2_3Test:
    """
    Tests for Phase 2 and Phase 3 components.
    """
    
    def __init__(self):
        """Initialize test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()
        
        # Test parameters
        self.batch_size = 2
        self.num_nodes = 10
        self.feature_dim = 32
        self.hidden_dim = 64
        self.num_classes = 1
        
        logger.info(f"Initialized Phase 2/3 test on {self.device}")
    
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_egnn(self) -> bool:
        """Test EGNN implementation."""
        logger.info("Testing EGNN...")
        
        try:
            # Create EGNN
            egnn = EGNN(
                node_dim=self.feature_dim,
                edge_dim=0,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            # Create dummy data
            h = torch.randn(self.batch_size, self.num_nodes, self.feature_dim).to(self.device)
            x = torch.randn(self.batch_size, self.num_nodes, 2).to(self.device)  # 2D coordinates
            edge_index = torch.randint(0, self.num_nodes, (2, self.num_nodes * 3)).to(self.device)
            
            # Forward pass
            output = egnn(h, x, edge_index)
            
            logger.info(f"EGNN output shape: {output.shape}")
            assert output.shape == (self.batch_size, self.num_nodes, self.feature_dim)
            
            # Test receiver geometry encoder
            receiver_encoder = ReceiverGeometryEncoder(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            features = torch.randn(self.batch_size, self.num_nodes, self.feature_dim).to(self.device)
            positions = torch.randn(self.batch_size, self.num_nodes, 2).to(self.device)
            
            encoded = receiver_encoder(features, positions)
            logger.info(f"Receiver encoder output shape: {encoded.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"EGNN test failed: {e}")
            return False
    
    def test_heat_kernel(self) -> bool:
        """Test heat kernel diffusion implementation."""
        logger.info("Testing Heat Kernel Diffusion...")
        
        try:
            # Create heat kernel layer
            heat_layer = HeatKernelLayer(
                in_channels=self.feature_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                diffusion_steps=3
            ).to(self.device)
            
            # Create dummy data
            x = torch.randn(self.batch_size, self.feature_dim, 32, 32).to(self.device)
            
            # Forward pass
            output = heat_layer(x)
            logger.info(f"Heat kernel layer output shape: {output.shape}")
            
            # Test wave equation approximator
            wave_approx = WaveEquationApproximator(
                in_channels=self.feature_dim,
                hidden_channels=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            output = wave_approx(x)
            logger.info(f"Wave equation approximator output shape: {output.shape}")
            
            # Test physics-guided diffusion
            physics_diffusion = PhysicsGuidedDiffusion(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            output, residual = physics_diffusion(x)
            logger.info(f"Physics-guided diffusion output shape: {output.shape}")
            logger.info(f"Physics residual shape: {residual.shape}")
            
            # Test physics loss
            physics_loss = physics_diffusion.compute_physics_loss(x)
            logger.info(f"Physics loss: {physics_loss.item()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Heat kernel test failed: {e}")
            return False
    
    def test_se3_transformer(self) -> bool:
        """Test SE(3)-Transformer implementation."""
        logger.info("Testing SE(3)-Transformer...")
        
        try:
            # Create SE(3)-Transformer
            se3_transformer = SE3Transformer(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            # Create dummy data
            x = torch.randn(self.batch_size, self.num_nodes, self.feature_dim).to(self.device)
            pos = torch.randn(self.batch_size, self.num_nodes, 3).to(self.device)  # 3D coordinates
            
            # Forward pass
            output = se3_transformer(x, pos)
            logger.info(f"SE(3)-Transformer output shape: {output.shape}")
            
            # Test SE(3) wavefield processor
            se3_processor = SE3WavefieldProcessor(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            # Create 3D wavefield data
            x_3d = torch.randn(self.batch_size, self.feature_dim, 8, 16, 16).to(self.device)
            
            output = se3_processor(x_3d)
            logger.info(f"SE(3) wavefield processor output shape: {output.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"SE(3)-Transformer test failed: {e}")
            return False
    
    def test_ensemble(self) -> bool:
        """Test ensemble implementations."""
        logger.info("Testing Ensemble Framework...")
        
        try:
            # Create dummy models
            models = []
            for i in range(3):
                model = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 1, 3, padding=1)
                ).to(self.device)
                models.append(model)
            
            # Test static ensemble
            static_ensemble = StaticEnsemble(models).to(self.device)
            x = torch.randn(self.batch_size, 1, 32, 32).to(self.device)
            output = static_ensemble(x)
            logger.info(f"Static ensemble output shape: {output.shape}")
            
            # Test dynamic ensemble
            dynamic_ensemble = DynamicEnsemble(
                models=models,
                feature_dim=32*32,  # Flattened spatial dimensions
                hidden_dim=64
            ).to(self.device)
            
            output = dynamic_ensemble(x)
            logger.info(f"Dynamic ensemble output shape: {output.shape}")
            
            # Test Bayesian ensemble
            bayesian_ensemble = BayesianEnsemble(
                models=models,
                mc_samples=5
            ).to(self.device)
            
            mean_pred, uncertainty = bayesian_ensemble(x)
            logger.info(f"Bayesian ensemble mean shape: {mean_pred.shape}")
            logger.info(f"Bayesian ensemble uncertainty shape: {uncertainty.shape}")
            
            # Test geometric ensemble
            geometric_ensemble = GeometricEnsemble(
                models=models,
                feature_dim=32*32,
                num_geometric_features=8
            ).to(self.device)
            
            output = geometric_ensemble(x)
            logger.info(f"Geometric ensemble output shape: {output.shape}")
            
            # Test ensemble manager
            manager = EnsembleManager()
            manager.add_ensemble("static", static_ensemble)
            manager.add_ensemble("dynamic", dynamic_ensemble)
            manager.add_ensemble("bayesian", bayesian_ensemble)
            
            manager.set_current_ensemble("static")
            output = manager.predict(x)
            logger.info(f"Ensemble manager output shape: {output.shape}")
            
            # Test weight computation
            weights = manager.get_ensemble_weights(x)
            logger.info(f"Ensemble weights: {list(weights.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ensemble test failed: {e}")
            return False
    
    def test_crf(self) -> bool:
        """Test CRF implementation."""
        logger.info("Testing CRF...")
        
        try:
            # Create continuous CRF
            crf = ContinuousCRF(
                num_classes=self.num_classes,
                window_size=5,
                num_iterations=3,
                trainable=False
            ).to(self.device)
            
            # Create dummy data
            unary = torch.randn(self.batch_size, self.num_classes, 32, 32).to(self.device)
            features = torch.randn(self.batch_size, 16, 32, 32).to(self.device)
            
            # Forward pass
            output = crf(unary, features)
            logger.info(f"CRF output shape: {output.shape}")
            
            # Test CRF post-processor
            post_processor = CRFPostProcessor(
                num_classes=self.num_classes,
                window_size=5,
                num_iterations=3,
                trainable=False
            ).to(self.device)
            
            output = post_processor(unary)
            logger.info(f"CRF post-processor output shape: {output.shape}")
            
            # Test CRF energy computation
            energy = post_processor.compute_crf_energy(unary, features)
            logger.info(f"CRF energy: {energy.item()}")
            
            return True
            
        except Exception as e:
            logger.error(f"CRF test failed: {e}")
            return False
    
    def test_integration(self) -> bool:
        """Test integration of Phase 2 and 3 components."""
        logger.info("Testing Phase 2/3 Integration...")
        
        try:
            # Create components
            egnn = EGNN(
                node_dim=self.feature_dim,
                edge_dim=0,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            heat_kernel = PhysicsGuidedDiffusion(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            se3_transformer = SE3Transformer(
                input_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            ).to(self.device)
            
            # Create ensemble of different model types
            models = [
                nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 1, 3, padding=1)
                ).to(self.device),
                nn.Sequential(
                    nn.Linear(32*32, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32*32)
                ).to(self.device)
            ]
            
            ensemble = StaticEnsemble(models).to(self.device)
            
            # Create CRF post-processor
            crf = CRFPostProcessor(
                num_classes=1,
                window_size=5,
                num_iterations=3
            ).to(self.device)
            
            # Test end-to-end pipeline
            x_2d = torch.randn(self.batch_size, 1, 32, 32).to(self.device)
            x_3d = torch.randn(self.batch_size, self.feature_dim, 8, 16, 16).to(self.device)
            
            # Process with different components
            heat_output, heat_residual = heat_kernel(x_2d)
            se3_output = se3_transformer(
                x_3d.view(self.batch_size, -1, self.feature_dim),
                torch.randn(self.batch_size, x_3d.shape[2]*x_3d.shape[3]*x_3d.shape[4], 3).to(self.device)
            )
            
            # Ensemble prediction
            ensemble_output = ensemble(x_2d)
            
            # CRF post-processing
            final_output = crf(ensemble_output)
            
            logger.info(f"Integration test completed successfully")
            logger.info(f"Heat kernel output: {heat_output.shape}")
            logger.info(f"SE3 output: {se3_output.shape}")
            logger.info(f"Ensemble output: {ensemble_output.shape}")
            logger.info(f"Final CRF output: {final_output.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """Run all Phase 2 and 3 tests."""
        logger.info("Starting Phase 2/3 tests...")
        
        results = {}
        
        # Run component tests
        results['egnn'] = self.test_egnn()
        results['heat_kernel'] = self.test_heat_kernel()
        results['se3_transformer'] = self.test_se3_transformer()
        results['ensemble'] = self.test_ensemble()
        results['crf'] = self.test_crf()
        
        # Run integration test
        results['integration'] = self.test_integration()
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        logger.info(f"\nPhase 2/3 Test Results:")
        logger.info(f"Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        return results

def main():
    """Run Phase 2 and 3 tests."""
    test_suite = Phase2_3Test()
    
    try:
        results = test_suite.run_all_tests()
        
        # Check if all tests passed
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("🎉 All Phase 2/3 tests passed!")
        else:
            logger.error("❌ Some Phase 2/3 tests failed!")
        
        return all_passed
        
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1) 