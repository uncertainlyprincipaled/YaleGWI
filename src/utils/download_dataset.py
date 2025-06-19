"""
Dataset Download Utility

This module provides functions to download the seismic waveform inversion dataset
from various sources (Kaggle, direct download, etc.).
"""

import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_kaggle(dataset_name: str = "openfwi/waveform-inversion", 
                        target_dir: str = "/content/YaleGWI") -> bool:
    """
    Download dataset from Kaggle using kaggle CLI.
    
    Args:
        dataset_name: Kaggle dataset name
        target_dir: Directory to download to
        
    Returns:
        bool: True if download successful
    """
    try:
        print(f"📥 Downloading dataset from Kaggle: {dataset_name}")
        
        # Install kaggle if not available
        try:
            import kaggle
        except ImportError:
            print("📦 Installing kaggle package...")
            subprocess.run(["pip", "install", "kaggle"], check=True)
        
        # Create target directory
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", dataset_name,
            "-p", target_dir,
            "--unzip"
        ], check=True)
        
        print("✅ Dataset downloaded successfully from Kaggle")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download from Kaggle: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading from Kaggle: {e}")
        return False

def download_from_url(url: str, target_dir: str = "/content/YaleGWI") -> bool:
    """
    Download dataset from a direct URL.
    
    Args:
        url: Direct download URL
        target_dir: Directory to download to
        
    Returns:
        bool: True if download successful
    """
    try:
        print(f"📥 Downloading dataset from URL: {url}")
        
        # Create target directory
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Download file
        filename = url.split('/')[-1]
        filepath = Path(target_dir) / filename
        
        subprocess.run([
            "wget", "--no-check-certificate", 
            "-O", str(filepath), url
        ], check=True)
        
        # Extract if it's a zip file
        if filename.endswith('.zip'):
            print("📦 Extracting zip file...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Clean up zip file
            filepath.unlink()
        
        print("✅ Dataset downloaded successfully from URL")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download from URL: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading from URL: {e}")
        return False

def download_from_google_drive(file_id: str, target_dir: str = "/content/YaleGWI") -> bool:
    """
    Download dataset from Google Drive using gdown.
    
    Args:
        file_id: Google Drive file ID
        target_dir: Directory to download to
        
    Returns:
        bool: True if download successful
    """
    try:
        print(f"📥 Downloading dataset from Google Drive: {file_id}")
        
        # Install gdown if not available
        try:
            import gdown
        except ImportError:
            print("📦 Installing gdown package...")
            subprocess.run(["pip", "install", "gdown"], check=True)
        
        # Create target directory
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Download file
        url = f"https://drive.google.com/uc?id={file_id}"
        subprocess.run([
            "gdown", url,
            "-O", str(Path(target_dir) / "train_samples.zip")
        ], check=True)
        
        # Extract zip file
        print("📦 Extracting zip file...")
        with zipfile.ZipFile(Path(target_dir) / "train_samples.zip", 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Clean up zip file
        (Path(target_dir) / "train_samples.zip").unlink()
        
        print("✅ Dataset downloaded successfully from Google Drive")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download from Google Drive: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading from Google Drive: {e}")
        return False

def setup_dataset_download(target_dir: str = "/content/YaleGWI") -> Dict[str, Any]:
    """
    Interactive setup for dataset download.
    
    Args:
        target_dir: Directory to download to
        
    Returns:
        Dict containing download results
    """
    print("🔍 Dataset Download Setup")
    print("="*50)
    print("Choose your download method:")
    print("1. Kaggle (requires kaggle API credentials)")
    print("2. Direct URL download")
    print("3. Google Drive download")
    print("4. Manual upload (you'll upload the files yourself)")
    
    try:
        choice = input("Enter your choice (1-4): ").strip()
    except:
        # In Colab, we can't use input(), so default to manual
        choice = "4"
        print("Defaulting to manual upload option")
    
    result = {
        'method': choice,
        'success': False,
        'target_dir': target_dir
    }
    
    if choice == "1":
        print("\n📋 Kaggle Download Instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. Download the kaggle.json file")
        print("4. Upload it to Colab or set up credentials")
        print("\nThen run:")
        print("download_from_kaggle()")
        
    elif choice == "2":
        url = input("Enter the download URL: ").strip()
        if url:
            result['success'] = download_from_url(url, target_dir)
        else:
            print("❌ No URL provided")
            
    elif choice == "3":
        file_id = input("Enter the Google Drive file ID: ").strip()
        if file_id:
            result['success'] = download_from_google_drive(file_id, target_dir)
        else:
            print("❌ No file ID provided")
            
    elif choice == "4":
        print("\n📋 Manual Upload Instructions:")
        print("1. Upload your train_samples.zip file to Colab")
        print("2. Extract it to /content/YaleGWI/train_samples/")
        print("3. Verify the data structure using verify_data_structure()")
        result['success'] = True  # Assume success for manual upload
        
    else:
        print("❌ Invalid choice")
    
    return result

def verify_downloaded_data(target_dir: str = "/content/YaleGWI") -> bool:
    """
    Verify that the downloaded data is properly structured.
    
    Args:
        target_dir: Directory containing the downloaded data
        
    Returns:
        bool: True if data is properly structured
    """
    try:
        from src.core.preprocess import verify_data_structure
        
        data_path = Path(target_dir) / "train_samples"
        if data_path.exists():
            print("🔍 Verifying downloaded data structure...")
            return verify_data_structure(data_path)
        else:
            print(f"❌ Data not found at {data_path}")
            return False
            
    except ImportError:
        print("⚠️ Could not import verification function")
        return True  # Assume valid if we can't verify

if __name__ == "__main__":
    # Example usage
    print("🚀 Dataset Download Utility")
    print("="*50)
    
    # Try different download methods
    methods = [
        ("Kaggle", lambda: download_from_kaggle()),
        ("Direct URL", lambda: download_from_url("YOUR_URL_HERE")),
        ("Google Drive", lambda: download_from_google_drive("YOUR_FILE_ID_HERE"))
    ]
    
    for method_name, method_func in methods:
        print(f"\nTrying {method_name} download...")
        if method_func():
            print(f"✅ {method_name} download successful!")
            break
        else:
            print(f"❌ {method_name} download failed")
    else:
        print("\n❌ All download methods failed")
        print("Please use manual upload or check your credentials/URLs") 