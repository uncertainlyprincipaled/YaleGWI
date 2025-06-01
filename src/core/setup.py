import os
import subprocess
from pathlib import Path
try:
    import kagglehub
except ImportError:
    kagglehub = None

def setup_environment():
    """Setup environment-specific configurations and download datasets if needed."""
    from src.core.config import CFG  # Import here to avoid circular dependency
    
    # Allow explicit environment override
    env_override = os.environ.get('GWI_ENV', '').lower()
    if env_override:
        CFG.env.kind = env_override
    
    if CFG.env.kind == 'colab':
        # Create data directory
        data_dir = Path('/content/data')
        data_dir.mkdir(exist_ok=True)
        
        # Update paths for Colab
        CFG.paths.root = data_dir / 'waveform-inversion'
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        
        # Update family paths
        CFG.paths.families = {
            'FlatVel_A'   : CFG.paths.train/'FlatVel_A',
            'FlatVel_B'   : CFG.paths.train/'FlatVel_B',
            'CurveVel_A'  : CFG.paths.train/'CurveVel_A',
            'CurveVel_B'  : CFG.paths.train/'CurveVel_B',
            'Style_A'     : CFG.paths.train/'Style_A',
            'Style_B'     : CFG.paths.train/'Style_B',
            'FlatFault_A' : CFG.paths.train/'FlatFault_A',
            'FlatFault_B' : CFG.paths.train/'FlatFault_B',
            'CurveFault_A': CFG.paths.train/'CurveFault_A',
            'CurveFault_B': CFG.paths.train/'CurveFault_B',
        }
        
        print("Environment setup complete for Colab")
    
    elif CFG.env.kind == 'sagemaker':
        # AWS SageMaker specific setup
        data_dir = Path('/opt/ml/input/data')  
        data_dir.mkdir(exist_ok=True)

        # Create a symbolic link to the dataset
        dataset_path = Path('/opt/ml/input/data/waveform-inversion')
        dataset_path.symlink_to(data_dir / 'waveform-inversion')

        # Download dataset
        print("Downloading dataset from Kaggle...")
        kagglehub.model_download('jamie-morgan/waveform-inversion', path=str(data_dir))

        # Update paths for SageMaker
        CFG.paths.root = data_dir / 'waveform-inversion'
        CFG.paths.train = CFG.paths.root / 'train_samples'
        CFG.paths.test = CFG.paths.root / 'test'
        
        # Update family paths
        CFG.paths.families = {
            'FlatVel_A'   : CFG.paths.train/'FlatVel_A',
            'FlatVel_B'   : CFG.paths.train/'FlatVel_B',
            'CurveVel_A'  : CFG.paths.train/'CurveVel_A',
            'CurveVel_B'  : CFG.paths.train/'CurveVel_B',
            'Style_A'     : CFG.paths.train/'Style_A',
            'Style_B'     : CFG.paths.train/'Style_B',
            'FlatFault_A' : CFG.paths.train/'FlatFault_A',
            'FlatFault_B' : CFG.paths.train/'FlatFault_B',
            'CurveFault_A': CFG.paths.train/'CurveFault_A',
            'CurveFault_B': CFG.paths.train/'CurveFault_B',
        }
        
        print("Paths configured for SageMaker environment") 
    
    elif CFG.env.kind == 'kaggle':
        # In Kaggle, do not use kagglehub or Kaggle API. Data is already available.
        print("Environment setup complete for Kaggle (no kagglehub needed)")
    
    # Add any additional logic for 'local' or other environments as needed 