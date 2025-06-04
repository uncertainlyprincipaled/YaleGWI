# ─ src/utils/push_to_drive.py ─
import subprocess
import os
import sys
from pathlib import Path
import json
from typing import Optional, Iterator
from google.colab import drive
import shutil
from src.core.data_manager import DataManager
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
import zipfile
import psutil
from tqdm import tqdm
import gc
import weakref

KAGGLE_DATA_DIR = Path("/kaggle/input/waveform-inversion")
WORK_DIR = Path("/kaggle/working")
ARCHIVE_PATH = WORK_DIR / "waveform-inversion.zip"
CONFIG_FILE = Path.home() / ".config" / "gwi_drive_config.json"
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

def load_config() -> dict:
    """Load Google Drive configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"folder_id": None}

def save_config(folder_id: str):
    """Save Google Drive configuration."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"folder_id": folder_id}, f)

def get_drive_folder_id() -> Optional[str]:
    """Get the Google Drive folder ID from config or environment."""
    config = load_config()
    folder_id = config.get("folder_id") or os.environ.get("GWI_DRIVE_FOLDER_ID")
    if not folder_id:
        print("❌ No Google Drive folder ID found!")
        print("Please set it using one of these methods:")
        print("1. Set GWI_DRIVE_FOLDER_ID environment variable")
        print("2. Run this script with --folder-id <ID>")
        print("3. Edit the config file at:", CONFIG_FILE)
        sys.exit(1)
    return folder_id

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Force garbage collection and cleanup memory."""
    # Clear any cached data
    if 'data_manager' in globals():
        data_manager = globals()['data_manager']
        if hasattr(data_manager, 'clear_cache'):
            data_manager.clear_cache()
    
    # Clear any cached credentials
    if 'service' in globals():
        service = globals()['service']
        if hasattr(service, '_http'):
            service._http.close()
    
    # Force garbage collection
    gc.collect()
    
    # Clear any remaining references except for essential modules
    essential_modules = {'__builtins__', '__name__', '__file__', 'gc', 'sys', 'os'}
    for name in list(globals().keys()):
        if name not in essential_modules:
            globals()[name] = None
    
    gc.collect()

def get_drive_service():
    """Get authenticated Google Drive service."""
    creds = None
    token_path = Path.home() / '.config' / 'token.pickle'
    
    # Load existing credentials if available
    if token_path.exists():
        with open(token_path, 'rb') as f:
            creds = pickle.load(f)
        f = None  # Clear file handle
    
    # If credentials are invalid or don't exist, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            flow = None  # Clear flow object
        
        # Save credentials
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, 'wb') as f:
            pickle.dump(creds, f)
        f = None  # Clear file handle
    
    service = build('drive', 'v3', credentials=creds)
    creds = None  # Clear credentials
    gc.collect()
    return service

def upload_to_drive(folder_id: str, file_path: Path):
    """Upload the given zip archive to Google Drive using the Drive API."""
    print(f"📤 Uploading {file_path.name} to Google Drive...")
    try:
        service = get_drive_service()
        
        file_metadata = {
            'name': file_path.name,
            'parents': [folder_id]
        }
        
        # Get file size for progress bar
        file_size = file_path.stat().st_size
        
        media = MediaFileUpload(
            str(file_path),
            mimetype='application/zip',
            resumable=True,
            chunksize=CHUNK_SIZE
        )
        
        # Create progress bar
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading")
        
        # Track upload progress
        def callback(request_id, response, exception):
            if exception:
                raise exception
            pbar.update(media._chunk_size)
            # Periodically cleanup
            if pbar.n % (CHUNK_SIZE * 10) == 0:  # Every 10 chunks
                cleanup_memory()
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            callback=callback
        ).execute()
        
        pbar.close()
        print(f"✅ Upload complete. File ID: {file.get('id')}")
        
        # Cleanup after upload
        media = None
        file = None
        cleanup_memory()
        
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        cleanup_memory()
        sys.exit(1)

def zip_families():
    """Zip each family/subfolder separately using DataManager, avoiding duplicates."""
    print("\U0001F4E6 Zipping families in batches...")
    data_manager = DataManager()
    families = [
        'FlatVel_A', 'FlatVel_B', 'CurveVel_A', 'CurveVel_B',
        'Style_A', 'Style_B', 'FlatFault_A', 'FlatFault_B',
        'CurveFault_A', 'CurveFault_B'
    ]
    output_dir = WORK_DIR / 'zipped_families'
    output_dir.mkdir(exist_ok=True)
    for family in families:
        seis_files, vel_files = data_manager.list_family_files(family)
        zip_path = output_dir / f"{family}.zip"
        added_files = set()
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f in seis_files:
                arcname = f.name
                if arcname not in added_files:
                    zipf.write(f, arcname=arcname)
                    added_files.add(arcname)
            if vel_files:
                for f in vel_files:
                    arcname = f.name
                    if arcname not in added_files:
                        zipf.write(f, arcname=arcname)
                        added_files.add(arcname)
        print(f"\u2705 Zipped {family} to {zip_path}")
    print("\U0001F389 All families zipped!")
    cleanup_memory()

def upload_all_zipped_families(folder_id: str):
    """Upload all zipped family archives to Google Drive."""
    zipped_dir = WORK_DIR / 'zipped_families'
    for zip_file in zipped_dir.glob('*.zip'):
        print(f"Uploading {zip_file.name}...")
        upload_to_drive(folder_id, zip_file)

def main():
    """Main entry point."""
    import argparse
    try:
        # Check if we're running in a Jupyter notebook
        in_notebook = False
        try:
            import IPython
            in_notebook = IPython.get_ipython() is not None
        except (ImportError, NameError):
            pass
        if in_notebook:
            folder_id = get_drive_folder_id()
        else:
            parser = argparse.ArgumentParser(description="Push dataset to Google Drive")
            parser.add_argument("--folder-id", help="Google Drive folder ID")
            args = parser.parse_args()
            if args.folder_id:
                save_config(args.folder_id)
            folder_id = get_drive_folder_id()
        zip_families()
        upload_all_zipped_families(folder_id)
    finally:
        cleanup_memory()

if __name__ == "__main__":
    main()

# Mount Google Drive
drive.mount('/content/drive')

# Create a directory in Google Drive
drive_path = Path('/content/drive/MyDrive/waveform_inversion_data')
drive_path.mkdir(exist_ok=True)

# Copy data using the DataManager
data_manager = DataManager(use_mmap=True, cache_size=1000)

# For each family of data
for family in ['FlatVel_A', 'FlatVel_B']:
    seis_files, vel_files = data_manager.list_family_files(family)
    
    # Create family directory
    family_dir = drive_path / family
    family_dir.mkdir(exist_ok=True)
    
    # Copy files in batches
    for i in range(0, len(seis_files), 100):  # Process 100 files at a time
        batch_seis = seis_files[i:i+100]
        batch_vel = vel_files[i:i+100] if vel_files else None
        
        for seis_file in batch_seis:
            shutil.copy2(seis_file, family_dir / 'data')
        if batch_vel:
            for vel_file in batch_vel:
                shutil.copy2(vel_file, family_dir / 'model')
