"""
Colab Setup Utility

This module provides automated setup functions for Google Colab environment,
including environment configuration, data verification, and preprocessing.
"""

import os
import sys
import subprocess
import gc
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_colab_environment(
    repo_url: str = "https://github.com/uncertainlyprincipaled/YaleGWI.git",
    install_extra_packages: bool = True
) -> Dict[str, Any]:
    """
    Set up the Colab environment with all necessary dependencies.
    
    Args:
        repo_url: URL of the repository to clone
        install_extra_packages: Whether to install additional packages for preprocessing
        
    Returns:
        Dict containing setup information
    """
    print("🚀 Setting up Colab environment...")
    
    # Clone repository if not already present
    if not Path('/content/YaleGWI').exists():
        print("📥 Cloning repository...")
        subprocess.run(['git', 'clone', repo_url], check=True)
    
    # Change to project directory
    os.chdir('/content/YaleGWI')
    
    # Install base requirements
    print("📦 Installing base requirements...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    
    # Install additional packages for preprocessing
    if install_extra_packages:
        print("📦 Installing additional packages for preprocessing...")
        extra_packages = ['zarr', 'dask', 'scipy']
        for package in extra_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
    
    # Set up environment variables
    os.environ['GWI_ENV'] = 'colab'
    os.environ['DEBUG_MODE'] = '0'
    
    # Add src to Python path
    sys.path.append('/content/YaleGWI/src')
    
    # Verify PyTorch and CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"✅ CUDA available: {cuda_available}")
        if cuda_available:
            print(f"✅ GPU count: {gpu_count}")
            print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch installation failed: {e}")
        cuda_available = False
        gpu_count = 0
    
    # Verify environment configuration
    try:
        from src.core.config import CFG
        print(f"✅ Environment detected: {CFG.env.kind}")
        print(f"✅ Device: {CFG.env.device}")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
    
    return {
        'repo_cloned': Path('/content/YaleGWI').exists(),
        'requirements_installed': True,
        'cuda_available': cuda_available,
        'gpu_count': gpu_count,
        'environment': 'colab'
    }

def mount_google_drive() -> bool:
    """
    Mount Google Drive for persistent storage.
    
    Returns:
        bool: True if mounting was successful
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create persistent directories
        persistent_dirs = [
            '/content/drive/MyDrive/YaleGWI/data',
            '/content/drive/MyDrive/YaleGWI/outputs',
            '/content/drive/MyDrive/YaleGWI/preprocessed'
        ]
        
        for dir_path in persistent_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✅ Google Drive mounted successfully")
        print("✅ Persistent directories created")
        return True
        
    except ImportError:
        print("❌ Google Drive mounting failed - not in Colab environment")
        return False
    except Exception as e:
        print(f"❌ Google Drive mounting failed: {e}")
        return False

def verify_data_availability(data_path: str = '/content/YaleGWI/train_samples', use_s3: bool = False) -> Dict[str, Any]:
    """
    Verify that training data is available and properly structured, either locally or in S3.
    If use_s3 is True, check S3 for at least one file per family (cost-efficient, one request per family).
    Args:
        data_path: Path to the training data directory (local, ignored if use_s3)
        use_s3: Whether to check S3 instead of local
    Returns:
        Dict containing verification results
    """
    if use_s3:
        print(f"🔍 Verifying data availability in S3 bucket...")
        
        import boto3
        import os
        from src.core.config import CFG, FAMILY_FILE_MAP
        bucket = os.environ.get('AWS_S3_BUCKET') or CFG.s3_paths.bucket
        region = os.environ.get('AWS_REGION', 'us-east-1')
        s3 = boto3.client('s3', region_name=region)
        found = []
        missing = []
        for family, info in FAMILY_FILE_MAP.items():
            s3_family_prefix = CFG.s3_paths.families[family]
            # Check seis_dir
            seis_prefix = f"{s3_family_prefix}/{info['seis_dir']}/" if info['seis_dir'] else f"{s3_family_prefix}/"
            vel_prefix = f"{s3_family_prefix}/{info['vel_dir']}/" if info['vel_dir'] else f"{s3_family_prefix}/"
            found_seis = False
            found_vel = False
            try:
                resp = s3.list_objects_v2(Bucket=bucket, Prefix=seis_prefix, MaxKeys=5)
                seis_keys = [obj['Key'] for obj in resp.get('Contents', [])]
                print(f"DEBUG: {family} seis_prefix={seis_prefix} keys={seis_keys}")
                if len(seis_keys) > 0:
                    found_seis = True
            except Exception as e:
                print(f"❌ {family}: error checking seis S3 ({e})")
            try:
                resp = s3.list_objects_v2(Bucket=bucket, Prefix=vel_prefix, MaxKeys=5)
                vel_keys = [obj['Key'] for obj in resp.get('Contents', [])]
                print(f"DEBUG: {family} vel_prefix={vel_prefix} keys={vel_keys}")
                if len(vel_keys) > 0:
                    found_vel = True
            except Exception as e:
                print(f"❌ {family}: error checking vel S3 ({e})")
            if found_seis and found_vel:
                found.append(family)
                print(f"✅ {family}: found in S3 (seis+vel)")
            else:
                missing.append(family)
                print(f"❌ {family}: not found in S3 (seis: {found_seis}, vel: {found_vel})")
        print(f"📊 S3 Summary: {len(found)}/{len(FAMILY_FILE_MAP)} families found")
        structure_valid = len(found) > 0
        if not structure_valid:
            print("⚠️ WARNING: No data found in S3 for any family!")
        return {
            'data_exists': structure_valid,
            'families_found': found,
            'families_missing': missing,
            'total_files': None,  # Not counting all files to save cost
            'structure_valid': structure_valid,
            'source': 's3',
        }
    else:
        print(f"🔍 Verifying data availability at {data_path}...")
        from pathlib import Path
        data_root = Path(data_path)
        result = {
            'data_exists': False,
            'families_found': [],
            'families_missing': [],
            'total_files': 0,
            'structure_valid': False,
            'source': 'local',
        }
        if not data_root.exists():
            print(f"❌ Data directory not found: {data_path}")
            print("⚠️ WARNING: No local data found!")
            return result
        result['data_exists'] = True
        expected_families = [
            'FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B',
            'Style_A', 'Style_B', 'FlatFault_A', 'FlatFault_B',
            'CurveFault_A', 'CurveFault_B'
        ]
        for family in expected_families:
            family_dir = data_root / family
            if family_dir.exists():
                npy_files = list(family_dir.glob('*.npy'))
                if npy_files:
                    result['families_found'].append(family)
                    result['total_files'] += len(npy_files)
                    print(f"✅ {family}: {len(npy_files)} files")
                else:
                    result['families_missing'].append(family)
                    print(f"⚠️ {family}: Directory exists but no .npy files")
            else:
                result['families_missing'].append(family)
                print(f"❌ {family}: Not found")
        # Verify data structure if families are found
        if result['families_found']:
            try:
                from src.core.preprocess import verify_data_structure
                result['structure_valid'] = verify_data_structure(data_root)
            except ImportError:
                print("⚠️ Could not import verification function")
                result['structure_valid'] = True  # Assume valid if we can't verify
        print(f"📊 Local Summary: {len(result['families_found'])}/{len(expected_families)} families found")
        if not result['families_found']:
            print("⚠️ WARNING: No local data found for any family!")
        return result

def run_preprocessing(
    input_root: str = '/content/YaleGWI/train_samples',
    output_root: str = '/content/YaleGWI/preprocessed',
    use_s3: bool = False,
    save_to_drive: bool = True
) -> Dict[str, Any]:
    """
    Run the preprocessing pipeline with monitoring and error handling.
    
    Args:
        input_root: Input data directory
        output_root: Output directory for processed data
        use_s3: Whether to use S3 for data operations
        save_to_drive: Whether to save results to Google Drive
        
    Returns:
        Dict containing preprocessing results
    """
    print("🔄 Starting preprocessing pipeline...")
    
    result = {
        'success': False,
        'processed_files': 0,
        'error': None,
        'output_path': output_root
    }
    
    # Monitor memory before preprocessing
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"💾 Initial memory usage: {initial_memory:.1f} MB")
    except ImportError:
        print("⚠️ psutil not available for memory monitoring")
    
    try:
        # Import preprocessing function
        from src.core.preprocess import load_data
        
        # Run preprocessing
        processed_paths = load_data(
            input_root=input_root,
            output_root=output_root,
            use_s3=use_s3
        )
        
        result['success'] = True
        result['processed_files'] = len(processed_paths)
        
        print(f"✅ Preprocessing completed successfully!")
        print(f"📊 Processed {len(processed_paths)} files")
        
        # Verify output
        output_dir = Path(output_root)
        if output_dir.exists():
            gpu0_dir = output_dir / 'gpu0'
            gpu1_dir = output_dir / 'gpu1'
            
            if gpu0_dir.exists() and gpu1_dir.exists():
                print("✅ GPU-specific datasets created")
                
                # Check zarr datasets
                try:
                    import zarr
                    for gpu_dir in [gpu0_dir, gpu1_dir]:
                        zarr_path = gpu_dir / 'seismic.zarr'
                        if zarr_path.exists():
                            data = zarr.open(str(zarr_path))
                            print(f"✅ {gpu_dir.name}: {data.shape} samples")
                except ImportError:
                    print("⚠️ zarr not available for dataset verification")
            else:
                print("⚠️ GPU-specific datasets not found")
        
        # Save to Google Drive if requested
        if save_to_drive and Path('/content/drive').exists():
            try:
                drive_output = '/content/drive/MyDrive/YaleGWI/preprocessed'
                subprocess.run(['cp', '-r', output_root, drive_output], check=True)
                print(f"✅ Preprocessed data saved to Google Drive: {drive_output}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to save to Google Drive: {e}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ Preprocessing failed: {e}")
        raise
    
    # Clean up memory
    gc.collect()
    
    # Monitor memory after preprocessing
    try:
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"💾 Final memory usage: {final_memory:.1f} MB")
        print(f"💾 Memory change: {final_memory - initial_memory:.1f} MB")
    except:
        pass
    
    return result

def setup_training_config(
    batch_size: int = 16,
    epochs: int = 10,
    use_amp: bool = True,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Configure training parameters for Colab environment.
    
    Args:
        batch_size: Training batch size
        epochs: Number of training epochs
        use_amp: Whether to use automatic mixed precision
        debug_mode: Whether to enable debug mode
        
    Returns:
        Dict containing training configuration
    """
    print("⚙️ Configuring training parameters...")
    
    try:
        from src.core.config import CFG
        
        # Set training parameters
        CFG.batch = batch_size
        CFG.epochs = epochs
        CFG.use_amp = use_amp
        CFG.debug_mode = debug_mode
        
        # Update debug settings
        CFG.set_debug_mode(debug_mode)
        
        config = {
            'batch_size': CFG.batch,
            'epochs': CFG.epochs,
            'use_amp': CFG.use_amp,
            'debug_mode': CFG.debug_mode,
            'device': CFG.env.device,
            'dtype': CFG.dtype
        }
        
        print("✅ Training configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        return config
        
    except ImportError as e:
        print(f"❌ Failed to configure training: {e}")
        return {}

def complete_colab_setup(
    data_path: str = '/content/YaleGWI/train_samples',
    use_s3: bool = False,
    mount_drive: bool = True,
    download_dataset: bool = False,
    dataset_source: str = 'manual',
    setup_aws: bool = True
) -> Dict[str, Any]:
    """
    Complete Colab setup including environment, data verification, and preprocessing.
    
    Args:
        data_path: Path to training data
        use_s3: Whether to use S3 for data operations
        mount_drive: Whether to mount Google Drive
        download_dataset: Whether to attempt dataset download
        dataset_source: Source for dataset download ('kaggle', 'url', 'gdrive', 'manual')
        setup_aws: Whether to set up AWS credentials
        
    Returns:
        Dict containing complete setup results
    """
    print("🎯 Starting complete Colab setup...")
    
    results = {}
    
    # Step 1: Environment setup
    print("\n" + "="*50)
    print("STEP 1: Environment Setup")
    print("="*50)
    results['environment'] = setup_colab_environment()
    
    # Step 2: AWS Setup (if requested)
    if setup_aws:
        print("\n" + "="*50)
        print("STEP 2: AWS Setup")
        print("="*50)
        
        # Try to load from Colab secrets first
        aws_from_secrets = setup_aws_credentials_from_secrets()
        if aws_from_secrets:
            results['aws_credentials'] = 'secrets'
            results['aws_verification'] = verify_aws_setup()
        else:
            # Fall back to manual setup
            aws_manual = setup_aws_credentials_manual()
            results['aws_credentials'] = 'manual' if aws_manual else 'none'
            if aws_manual:
                results['aws_verification'] = verify_aws_setup()
    
    # Step 3: Mount Google Drive (optional)
    if mount_drive:
        print("\n" + "="*50)
        print("STEP 3: Google Drive Setup")
        print("="*50)
        results['drive_mounted'] = mount_google_drive()
    
    # Step 4: Dataset download (optional)
    if download_dataset:
        print("\n" + "="*50)
        print("STEP 4: Dataset Download")
        print("="*50)
        try:
            from src.utils.download_dataset import (
                download_from_kaggle, 
                download_from_url, 
                download_from_google_drive
            )
            
            if dataset_source == 'kaggle':
                results['dataset_download'] = download_from_kaggle()
            elif dataset_source == 'url':
                url = input("Enter download URL: ").strip()
                results['dataset_download'] = download_from_url(url) if url else False
            elif dataset_source == 'gdrive':
                file_id = input("Enter Google Drive file ID: ").strip()
                results['dataset_download'] = download_from_google_drive(file_id) if file_id else False
            else:
                print("📋 Manual dataset setup:")
                print("1. Upload your train_samples.zip to Colab")
                print("2. Extract to /content/YaleGWI/train_samples/")
                results['dataset_download'] = True
                
        except ImportError:
            print("⚠️ Dataset download utilities not available")
            results['dataset_download'] = False
    
    # Step 5: Data verification
    print("\n" + "="*50)
    print("STEP 5: Data Verification")
    print("="*50)
    results['data_verification'] = verify_data_availability(data_path, use_s3=use_s3)
    
    # Step 6: Preprocessing (if data is available)
    if results['data_verification']['data_exists'] and results['data_verification']['families_found']:
        print("\n" + "="*50)
        print("STEP 6: Preprocessing")
        print("="*50)
        results['preprocessing'] = run_preprocessing(
            input_root=data_path,
            use_s3=use_s3,
            save_to_drive=mount_drive
        )
    
    # Step 7: Training configuration
    print("\n" + "="*50)
    print("STEP 7: Training Configuration")
    print("="*50)
    results['training_config'] = setup_training_config()
    
    print("\n" + "="*50)
    print("🎉 Setup Complete!")
    print("="*50)
    
    # Summary
    print("\n📋 Setup Summary:")
    print(f"  Environment: {'✅' if results['environment']['repo_cloned'] else '❌'}")
    print(f"  CUDA: {'✅' if results['environment']['cuda_available'] else '❌'}")
    if setup_aws:
        aws_status = results.get('aws_credentials', 'none')
        print(f"  AWS Credentials: {'✅' if aws_status != 'none' else '❌'} ({aws_status})")
        if 'aws_verification' in results:
            aws_ver = results['aws_verification']
            print(f"  S3 Access: {'✅' if aws_ver.get('s3_accessible', False) else '❌'}")
            print(f"  S3 Bucket: {'✅' if aws_ver.get('bucket_exists', False) else '❌'}")
    if mount_drive:
        print(f"  Google Drive: {'✅' if results.get('drive_mounted', False) else '❌'}")
    if download_dataset:
        print(f"  Dataset Download: {'✅' if results.get('dataset_download', False) else '❌'}")
    print(f"  Data: {'✅' if results['data_verification']['data_exists'] else '❌'}")
    print(f"  Families: {len(results['data_verification']['families_found'])}/10")
    if 'preprocessing' in results:
        print(f"  Preprocessing: {'✅' if results['preprocessing']['success'] else '❌'}")
    
    return results

def setup_aws_credentials_from_secrets() -> bool:
    """
    Set up AWS credentials from Google Colab secrets.
    
    Returns:
        bool: True if credentials were loaded successfully
    """
    try:
        from google.colab import userdata
        import os
        # Load AWS credentials from Colab secrets
        os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('aws_access_key_id') or ''
        os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('aws_secret_access_key') or ''
        os.environ['AWS_REGION'] = userdata.get('aws_region') or 'us-east-1'
        os.environ['AWS_S3_BUCKET'] = userdata.get('aws_s3_bucket') or ''
        print("✅ AWS credentials loaded from Colab secrets")
        return True
    except ImportError:
        print("❌ Google Colab userdata not available - not in Colab environment")
        return False
    except Exception as e:
        print(f"❌ Failed to load AWS credentials from secrets: {e}")
        return False

def setup_aws_credentials_manual() -> bool:
    """
    Set up AWS credentials manually (for development/testing).
    
    Returns:
        bool: True if credentials were set up successfully
    """
    try:
        import os
        import json
        from pathlib import Path
        
        # Create .env/aws directory
        os.makedirs('.env/aws', exist_ok=True)
        
        # Check if credentials file already exists
        creds_path = Path('.env/aws/credentials.json')
        if creds_path.exists():
            print("✅ AWS credentials file already exists")
            return True
        
        print("📋 Manual AWS credentials setup:")
        print("Please create .env/aws/credentials.json with the following structure:")
        print("""
{
    "aws_access_key_id": "YOUR_ACCESS_KEY_ID",
    "aws_secret_access_key": "YOUR_SECRET_ACCESS_KEY", 
    "region_name": "us-east-1",
    "s3_bucket": "YOUR_BUCKET_NAME"
}
        """)
        
        return False
        
    except Exception as e:
        print(f"❌ Failed to set up AWS credentials: {e}")
        return False

def verify_aws_setup() -> Dict[str, Any]:
    """
    Verify AWS setup and credentials.
    
    Returns:
        Dict containing AWS setup verification results
    """
    print("🔍 Verifying AWS setup...")
    
    results = {
        'credentials_loaded': False,
        's3_accessible': False,
        'bucket_exists': False,
        'region': None,
        'bucket_name': None
    }
    
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # Check if credentials are available
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        results['credentials_loaded'] = True
        results['account_id'] = identity['Account']
        results['user_arn'] = identity['Arn']
        
        print(f"✅ AWS credentials valid for account: {identity['Account']}")
        
        # Check S3 access
        s3_client = boto3.client('s3')
        s3_client.list_buckets()
        results['s3_accessible'] = True
        print("✅ S3 access verified")
        
        # Check specific bucket
        bucket_name = os.environ.get('AWS_S3_BUCKET')
        if bucket_name:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                results['bucket_exists'] = True
                results['bucket_name'] = bucket_name
                print(f"✅ S3 bucket '{bucket_name}' exists and accessible")
            except ClientError as e:
                print(f"⚠️ S3 bucket '{bucket_name}' not accessible: {e}")
        
        # Get region
        region = os.environ.get('AWS_REGION', 'us-east-1')
        results['region'] = region
        print(f"✅ Using AWS region: {region}")
        
    except NoCredentialsError:
        print("❌ AWS credentials not found")
    except Exception as e:
        print(f"❌ AWS verification failed: {e}")
    
    return results

if __name__ == "__main__":
    # Example usage
    results = complete_colab_setup()
    print("\nSetup completed with results:", results) 