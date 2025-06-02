from pathlib import Path
import json
import os

def setup_drive_folder():
    """Set up Google Drive folder ID for the project."""
    # Create config directory if it doesn't exist
    config_dir = Path.home() / ".config"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "gwi_drive_config.json"
    
    # Get folder ID from user
    folder_id = input("Please enter your Google Drive folder ID: ").strip()
    
    # Save to config file
    with open(config_file, 'w') as f:
        json.dump({"folder_id": folder_id}, f)
    
    print(f"✅ Google Drive folder ID saved to {config_file}")
    print("You can now run push_to_drive.py")

if __name__ == "__main__":
    setup_drive_folder() 