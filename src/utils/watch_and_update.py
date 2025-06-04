import time
from pathlib import Path
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class SourceFileHandler(FileSystemEventHandler):
    def __init__(self, src_dir: Path, update_script: Path):
        self.src_dir = src_dir
        self.update_script = update_script
        self.last_update = 0
        self.cooldown = 1  # seconds between updates

    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only process Python files
        if not event.src_path.endswith('.py'):
            return
            
        # Check if file is in our source directory
        try:
            rel_path = Path(event.src_path).relative_to(self.src_dir)
        except ValueError:
            return
            
        # Prevent multiple rapid updates
        current_time = time.time()
        if current_time - self.last_update < self.cooldown:
            return
            
        print(f"\nDetected change in {rel_path}")
        print("Updating Kaggle notebook...")
        
        # Run the update script
        try:
            result = subprocess.run([sys.executable, str(self.update_script)], 
                                 check=True, 
                                 capture_output=True, 
                                 text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error updating notebook: {e}")
            if e.stdout:
                print("Output:", e.stdout)
            if e.stderr:
                print("Errors:", e.stderr)
            
        self.last_update = current_time

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Setup paths
    src_dir = project_root / 'src'
    update_script = src_dir / 'utils' / 'update_kaggle_notebook.py'
    
    if not update_script.exists():
        print(f"Error: Update script not found at {update_script}")
        return
        
    # Create and start the observer
    event_handler = SourceFileHandler(src_dir, update_script)
    observer = Observer()
    observer.schedule(event_handler, str(src_dir), recursive=True)
    observer.start()
    
    print(f"Watching for changes in {src_dir}")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped watching for changes")
    
    observer.join()

if __name__ == '__main__':
    if os.environ.get("AUTOUPDATE") == "1":
        main()
    else:
        print("File watcher disabled. Set AUTOUPDATE=1 to enable.") 