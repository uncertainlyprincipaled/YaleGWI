import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.registry import ModelRegistry
from core.checkpoint import CheckpointManager
from core.geometric_loader import FamilyDataLoader, GeometricDataset
from core.geometric_cv import GeometricCrossValidator
from core.model import SpecProjNet
from core.config import CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase1IntegrationTest:
    """
    Comprehensive integration tests for Phase 1 components.
    """
    
    def __init__(self):
        """Initialize test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test data parameters
        self.batch_size = 4
        self.num_samples = 20
        self.input_shape = (32, 256, 64)  # [channels, height, width]
        
        logger.info(f"Initialized Phase 1 integration test in {self.temp_dir}")
        logger.info(f"Using device: {self.device}")
    
    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
        logger.info("Cleaned up temporary files")
    
    def create_dummy_data(self) -> Dict[str, Any]:
        """
        Create dummy data for testing.
        
        Returns:
            Dictionary containing dummy data and metadata
        """
        # Create dummy seismic data
        seismic_data = torch.randn(self.num_samples, *self.input_shape)
        
        # Create dummy velocity data
        velocity_data = torch.randn(self.num_samples, 1, 70, 70)
        
        # Create dummy family labels
        families = ['FlatVel_A', 'CurveVel_A', 'Style_A', 'Fault_A']
        family_labels = [families[i % len(families)] for i in range(self.num_samples)]
        
        # Create dummy metadata
        metadata = {
            'families': family_labels,
            'data_shape': self.input_shape,
            'num_samples': self.num_samples
        }
        
        return {
            'seismic': seismic_data,
            'velocity': velocity_data,
            'families': family_labels,
            'metadata': metadata
        }
    
    def test_model_registry(self) -> bool:
        """
        Test ModelRegistry functionality.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing ModelRegistry...")
        
        try:
            # Initialize registry
            registry_dir = Path(self.temp_dir) / "models"
            registry = ModelRegistry(registry_dir=str(registry_dir))
            
            # Create dummy model
            model = SpecProjNet()
            model_id = "test_model"
            family = "FlatVel_A"
            equivariance = ["translation", "rotation"]
            
            # Register model
            version_id = registry.register_model(
                model=model,
                model_id=model_id,
                family=family,
                equivariance=equivariance,
                metadata={'test': True}
            )
            
            logger.info(f"Registered model with version ID: {version_id}")
            
            # List models
            models = registry.list_models()
            logger.info(f"Found {len(models)} registered models")
            
            # Load model
            loaded_model = registry.load_model(version_id, SpecProjNet)
            logger.info("Successfully loaded model from registry")
            
            # Get model info
            model_info = registry.get_model_info(version_id)
            logger.info(f"Model info: {model_info['family']}, {model_info['equivariance']}")
            
            # Test family filtering
            family_models = registry.list_models(family=family)
            logger.info(f"Found {len(family_models)} models for family {family}")
            
            return True
            
        except Exception as e:
            logger.error(f"ModelRegistry test failed: {e}")
            return False
    
    def test_checkpoint_manager(self) -> bool:
        """
        Test CheckpointManager functionality.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing CheckpointManager...")
        
        try:
            # Initialize checkpoint manager
            checkpoint_dir = Path(self.temp_dir) / "checkpoints"
            checkpoint_manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
            
            # Create dummy model and optimizer
            model = SpecProjNet()
            optimizer = torch.optim.Adam(model.parameters())
            
            # Test parameters
            model_id = "test_model"
            family = "CurveVel_A"
            equivariance = ["translation"]
            metrics = {'train_loss': 0.5, 'val_loss': 0.3}
            
            # Save checkpoint
            checkpoint_id = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                model_id=model_id,
                family=family,
                equivariance=equivariance,
                metrics=metrics,
                metadata={'test': True}
            )
            
            logger.info(f"Saved checkpoint with ID: {checkpoint_id}")
            
            # List checkpoints
            checkpoints = checkpoint_manager.list_checkpoints()
            logger.info(f"Found {len(checkpoints)} checkpoints")
            
            # Load checkpoint
            new_model = SpecProjNet()
            new_optimizer = torch.optim.Adam(new_model.parameters())
            
            epoch, loaded_metrics = checkpoint_manager.load_checkpoint(
                checkpoint_id, new_model, new_optimizer
            )
            
            logger.info(f"Loaded checkpoint from epoch {epoch}")
            logger.info(f"Loaded metrics: {loaded_metrics}")
            
            # Test filtering
            family_checkpoints = checkpoint_manager.list_checkpoints(family=family)
            logger.info(f"Found {len(family_checkpoints)} checkpoints for family {family}")
            
            # Get checkpoint info
            checkpoint_info = checkpoint_manager.get_checkpoint_info(checkpoint_id)
            logger.info(f"Checkpoint info: {checkpoint_info['family']}, {checkpoint_info['equivariance']}")
            
            return True
            
        except Exception as e:
            logger.error(f"CheckpointManager test failed: {e}")
            return False
    
    def test_geometric_loader(self) -> bool:
        """
        Test FamilyDataLoader functionality.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing FamilyDataLoader...")
        
        try:
            # Create dummy data structure
            data_root = Path(self.temp_dir) / "data"
            data_root.mkdir(parents=True, exist_ok=True)
            
            # Create family directories and dummy data
            families = ['FlatVel_A', 'CurveVel_A', 'Style_A']
            dummy_data = self.create_dummy_data()
            
            for i, family in enumerate(families):
                family_dir = data_root / family
                family_dir.mkdir(exist_ok=True)
                
                # Save dummy data for this family
                family_indices = [j for j, f in enumerate(dummy_data['families']) if f == family]
                if family_indices:
                    family_seismic = dummy_data['seismic'][family_indices]
                    torch.save(family_seismic, family_dir / "seismic.pt")
                    
                    family_velocity = dummy_data['velocity'][family_indices]
                    torch.save(family_velocity, family_dir / "velocity.pt")
            
            # Initialize family data loader
            family_loader = FamilyDataLoader(
                data_root=str(data_root),
                batch_size=self.batch_size,
                num_workers=0,  # Use 0 for testing
                extract_features=True
            )
            
            # Test loading for each family
            for family in families:
                try:
                    loader = family_loader.get_loader(family)
                    batch = next(iter(loader))
                    
                    logger.info(f"Successfully loaded batch from {family}")
                    logger.info(f"Batch keys: {batch.keys()}")
                    logger.info(f"Data shape: {batch['data'].shape}")
                    
                    if 'features' in batch:
                        logger.info(f"Features keys: {batch['features'].keys()}")
                    
                except Exception as e:
                    logger.warning(f"Could not load data for family {family}: {e}")
            
            # Test getting all loaders
            all_loaders = family_loader.get_all_loaders()
            logger.info(f"Created {len(all_loaders)} data loaders")
            
            # Test getting family stats
            for family in families:
                try:
                    stats = family_loader.get_family_stats(family)
                    logger.info(f"Stats for {family}: {stats}")
                except Exception as e:
                    logger.warning(f"Could not get stats for family {family}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"FamilyDataLoader test failed: {e}")
            return False
    
    def test_geometric_cv(self) -> bool:
        """
        Test GeometricCrossValidator functionality.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing GeometricCrossValidator...")
        
        try:
            # Initialize cross validator
            cross_validator = GeometricCrossValidator(n_splits=3)
            
            # Create dummy dataset
            dummy_data = self.create_dummy_data()
            
            # Create tensor dataset
            dataset = torch.utils.data.TensorDataset(
                dummy_data['seismic'],
                dummy_data['velocity']
            )
            
            # Test family-based splitting
            family_splits = cross_validator.split_by_family(
                dataset=dataset,
                family_labels=dummy_data['families']
            )
            
            logger.info(f"Created {len(family_splits)} family-based splits")
            
            # Test geometric feature splitting
            geometric_features = {
                'gradient': torch.randn(len(dataset), 10),
                'spectral': torch.randn(len(dataset), 8)
            }
            
            geometry_splits = cross_validator.split_by_geometry(
                dataset=dataset,
                geometric_features=geometric_features
            )
            
            logger.info(f"Created {len(geometry_splits)} geometry-based splits")
            
            # Test geometric metrics computation
            y_true = torch.randn(10, 70, 70)
            y_pred = torch.randn(10, 70, 70)
            
            metrics = cross_validator.compute_geometric_metrics(
                y_true.numpy(),
                y_pred.numpy()
            )
            
            logger.info(f"Computed geometric metrics: {metrics}")
            
            # Test cross-validation
            results = cross_validator.cross_validate(
                model=SpecProjNet(),
                dataset=dataset,
                family_labels=dummy_data['families'],
                geometric_features=geometric_features,
                batch_size=2,
                device=self.device
            )
            
            logger.info(f"Cross-validation results: {list(results.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"GeometricCrossValidator test failed: {e}")
            return False
    
    def test_end_to_end_integration(self) -> bool:
        """
        Test end-to-end integration of all Phase 1 components.
        
        Returns:
            True if test passes, False otherwise
        """
        logger.info("Testing end-to-end integration...")
        
        try:
            # Create dummy data
            dummy_data = self.create_dummy_data()
            
            # Initialize all components
            registry = ModelRegistry(registry_dir=str(Path(self.temp_dir) / "models"))
            checkpoint_manager = CheckpointManager(checkpoint_dir=str(Path(self.temp_dir) / "checkpoints"))
            cross_validator = GeometricCrossValidator(n_splits=2)
            
            # Create and register model
            model = SpecProjNet()
            model_id = "integration_test_model"
            family = "FlatVel_A"
            equivariance = ["translation", "rotation"]
            
            version_id = registry.register_model(
                model=model,
                model_id=model_id,
                family=family,
                equivariance=equivariance
            )
            
            # Create optimizer and save checkpoint
            optimizer = torch.optim.Adam(model.parameters())
            metrics = {'train_loss': 0.4, 'val_loss': 0.2, 'ssim': 0.8}
            
            checkpoint_id = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=3,
                model_id=model_id,
                family=family,
                equivariance=equivariance,
                metrics=metrics
            )
            
            # Test cross-validation with registered model
            dataset = torch.utils.data.TensorDataset(
                dummy_data['seismic'][:10],  # Use subset for testing
                dummy_data['velocity'][:10]
            )
            
            family_labels = dummy_data['families'][:10]
            geometric_features = {
                'gradient': torch.randn(10, 5),
                'spectral': torch.randn(10, 3)
            }
            
            cv_results = cross_validator.cross_validate(
                model=model,
                dataset=dataset,
                family_labels=family_labels,
                geometric_features=geometric_features,
                batch_size=2,
                device=self.device
            )
            
            logger.info("End-to-end integration test completed successfully")
            logger.info(f"CV results keys: {list(cv_results.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"End-to-end integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all Phase 1 integration tests.
        
        Returns:
            Dictionary of test results
        """
        logger.info("Starting Phase 1 integration tests...")
        
        results = {}
        
        # Run individual component tests
        results['model_registry'] = self.test_model_registry()
        results['checkpoint_manager'] = self.test_checkpoint_manager()
        results['geometric_loader'] = self.test_geometric_loader()
        results['geometric_cv'] = self.test_geometric_cv()
        
        # Run end-to-end test
        results['end_to_end'] = self.test_end_to_end_integration()
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        logger.info(f"\nPhase 1 Integration Test Results:")
        logger.info(f"Passed: {passed}/{total}")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        return results

def main():
    """Run Phase 1 integration tests."""
    test_suite = Phase1IntegrationTest()
    
    try:
        results = test_suite.run_all_tests()
        
        # Check if all tests passed
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("🎉 All Phase 1 integration tests passed!")
        else:
            logger.error("❌ Some Phase 1 integration tests failed!")
        
        return all_passed
        
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 