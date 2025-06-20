import os
from pathlib import Path
import logging
import torch
from src.core.preprocess import main as preprocess_main
from src.core.data_manager import DataManager
from src.core.geometric_loader import FamilyDataLoader
from src.core.geometric_cv import GeometricCrossValidator
from src.core.registry import ModelRegistry
from src.core.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_preprocessing_pipeline():
    """Test the preprocessing pipeline and verify data loading."""
    try:
        # 1. Run preprocessing
        logger.info("Starting preprocessing...")
        preprocess_main()
        logger.info("Preprocessing completed successfully")
        
        # 2. Test DataManager
        logger.info("\nTesting DataManager...")
        data_manager = DataManager(data_root="/kaggle/working/preprocessed")
        families = data_manager.list_family_files("gpu0")
        logger.info(f"Found {len(families)} files in gpu0")
        
        # 3. Test FamilyDataLoader
        logger.info("\nTesting FamilyDataLoader...")
        family_loader = FamilyDataLoader(
            data_root="/kaggle/working/preprocessed/gpu0",
            batch_size=4,
            num_workers=2
        )
        
        # Test loading a batch
        for family, loader in family_loader.get_all_loaders().items():
            batch = next(iter(loader))
            logger.info(f"Successfully loaded batch from {family}")
            logger.info(f"Batch shape: {batch['data'].shape}")
            break
        
        # 4. Test GeometricCrossValidator
        logger.info("\nTesting GeometricCrossValidator...")
        cross_validator = GeometricCrossValidator(n_splits=3)
        
        # Create dummy data for testing
        dummy_data = torch.randn(100, 32, 256, 64)
        dummy_labels = ["family1"] * 50 + ["family2"] * 50
        
        # Test cross-validation splits
        splits = cross_validator.split_by_family(
            dataset=torch.utils.data.TensorDataset(dummy_data),
            family_labels=dummy_labels
        )
        logger.info(f"Created {len(splits)} cross-validation splits")
        
        # 5. Test ModelRegistry
        logger.info("\nTesting ModelRegistry...")
        registry = ModelRegistry(registry_dir="/kaggle/working/models")
        logger.info("ModelRegistry initialized successfully")
        
        # 6. Test CheckpointManager
        logger.info("\nTesting CheckpointManager...")
        checkpoint_manager = CheckpointManager(checkpoint_dir="/kaggle/working/checkpoints")
        logger.info("CheckpointManager initialized successfully")
        
        logger.info("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_preprocessing_pipeline() 