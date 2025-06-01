import os
import subprocess
from pathlib import Path
import kagglehub

def setup_environment():
    """Setup environment-specific configurations and download datasets if needed."""
    from config import CFG  # Import here to avoid circular dependency
    
    if CFG.env.kind == 'colab':
        # Install required packages
        try:
            import kagglehub
        except ImportError:
            subprocess.run(['pip', 'install', 'kagglehub'], check=True)
            import kagglehub

        # Clone repository if not already present
        repo_dir = Path('/content/YaleGWI')
        if not repo_dir.exists():
            print("Cloning repository from GitHub...")
            subprocess.run(['git', 'clone', 'https://github.com/your-username/YaleGWI.git', str(repo_dir)], check=True)
            os.chdir(repo_dir)
        
        # Create data directory
        data_dir = Path('/content/data')
        data_dir.mkdir(exist_ok=True)
        
        # Download dataset
        print("Downloading dataset from Kaggle...")
        kagglehub.model_download('jamie-morgan/waveform-inversion', path=str(data_dir))
        
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
        # Clone repository if not already present
        repo_dir = Path('/opt/ml/code/YaleGWI')
        if not repo_dir.exists():
            print("Cloning repository from GitHub...")
            subprocess.run(['git', 'clone', 'https://github.com/your-username/YaleGWI.git', str(repo_dir)], check=True)
            os.chdir(repo_dir)
            
        # AWS SageMaker specific setup
        data_dir = Path('/opt/ml/input/data')
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
        
        print("Environment setup complete for SageMaker") 