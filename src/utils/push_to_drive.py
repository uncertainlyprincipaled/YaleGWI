# ─ src/utils/push_to_drive.py ─
import subprocess
import os
import sys
from pathlib import Path
import json
from typing import Optional

KAGGLE_DATA_DIR = Path("/kaggle/input/waveform-inversion")
WORK_DIR = Path("/kaggle/working")
ARCHIVE_PATH = WORK_DIR / "waveform-inversion.zip"
DRIVE_REMOTE = "gdrive:"
CONFIG_FILE = Path.home() / ".config" / "gwi_drive_config.json"

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

def zip_dataset():
    """Create a zip archive of the dataset."""
    print("📦 Zipping dataset...")
    if ARCHIVE_PATH.exists():
        print(f"⚠️  Archive already exists at {ARCHIVE_PATH}")
        if input("Overwrite? (y/N): ").lower() != 'y':
            print("Aborting...")
            sys.exit(0)
        ARCHIVE_PATH.unlink()
    
    cmd = [
        "zip", "-r", "-q", str(ARCHIVE_PATH),
        str(KAGGLE_DATA_DIR)
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Created {ARCHIVE_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create zip: {e}")
        sys.exit(1)

def upload_to_drive(folder_id: str):
    """Upload the zip archive to Google Drive."""
    print("📤 Uploading to Google Drive via rclone...")
    cmd = [
        "rclone", "copy", str(ARCHIVE_PATH),
        f"{DRIVE_REMOTE}{folder_id}/",
        "--drive-chunk-size", "64M",
        "--progress",
        "--stats", "30s"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✅ Upload complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to upload: {e}")
        sys.exit(1)

def ensure_rclone():
    """Ensure rclone is installed and configured."""
    if not Path("/usr/bin/rclone").exists():
        print("📦 Installing rclone...")
        try:
            subprocess.run(["pip", "install", "-q", "rclone"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install rclone: {e}")
            sys.exit(1)
    
    # Check if rclone is configured
    try:
        result = subprocess.run(["rclone", "listremotes"], 
                              capture_output=True, text=True, check=True)
        if "gdrive:" not in result.stdout:
            print("⚠️  Google Drive remote not configured!")
            print("Please run: rclone config")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check rclone configuration: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Push dataset to Google Drive")
    parser.add_argument("--folder-id", help="Google Drive folder ID")
    args = parser.parse_args()
    
    if args.folder_id:
        save_config(args.folder_id)
    
    folder_id = get_drive_folder_id()
    ensure_rclone()
    zip_dataset()
    upload_to_drive(folder_id)

if __name__ == "__main__":
    main()
