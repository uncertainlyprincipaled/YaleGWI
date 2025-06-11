import os
import boto3
from pathlib import Path
from tqdm import tqdm
import hashlib
import json
from datetime import datetime
import time
import logging
from typing import Dict, List, Tuple
from botocore.exceptions import ClientError

# --- KAGGLE ENVIRONMENT SETUP ---
KAGGLE_WORKING_DIR = Path('/kaggle/working')
KAGGLE_INPUT_DIR = Path('/kaggle/input')

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(KAGGLE_WORKING_DIR / 's3_upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- INSTRUCTIONS ---
# 1. Set your AWS credentials securely (see below).
# 2. Set the S3 bucket name and prefix.
# 3. Run this notebook in Kaggle to stream-upload files to S3.

# --- SET AWS CREDENTIALS ---
# Option 1: Paste credentials here (remove after use!)
AWS_ACCESS_KEY_ID = 'YOUR_ACCESS_KEY'
AWS_SECRET_ACCESS_KEY = 'YOUR_SECRET_KEY'
AWS_REGION = 'us-east-1'

os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['AWS_DEFAULT_REGION'] = AWS_REGION

# --- S3 CONFIG ---
S3_BUCKET = 'yale-gwi'
S3_PREFIX = 'raw/'  # Folder in S3 bucket
BATCH_SIZE = 100  # Number of files to process in one batch
CHECKPOINT_INTERVAL = 300  # Save checkpoint every 5 minutes
MAX_RETRIES = 3  # Maximum number of retries for failed uploads
RETRY_DELAY = 5  # Delay between retries in seconds

# --- DATASET PATH ---
DATASET_PATH = KAGGLE_INPUT_DIR / 'waveform-inversion'

# --- S3 CLIENT ---
s3 = boto3.client('s3')

class S3BucketManager:
    def __init__(self, bucket_name: str, prefix: str):
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/'  # Ensure prefix ends with /
        self.s3 = boto3.client('s3')
        
    def validate_bucket(self) -> bool:
        """Validate bucket exists and is accessible."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to bucket: {self.bucket_name}")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"Bucket {self.bucket_name} does not exist")
            elif error_code == '403':
                logger.error(f"Access denied to bucket {self.bucket_name}")
            else:
                logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
            return False
            
    def validate_prefix(self) -> bool:
        """Validate prefix exists in bucket."""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix,
                MaxKeys=1
            )
            return 'Contents' in response
        except ClientError as e:
            logger.error(f"Error validating prefix {self.prefix}: {e}")
            return False
            
    def create_prefix(self) -> bool:
        """Create prefix in bucket if it doesn't exist."""
        try:
            # Create an empty object with the prefix as the key
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=self.prefix
            )
            logger.info(f"Created prefix {self.prefix} in bucket {self.bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Error creating prefix {self.prefix}: {e}")
            return False
            
    def get_existing_files(self) -> List[str]:
        """Get list of existing files in prefix."""
        existing_files = []
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        existing_files.append(obj['Key'])
            return existing_files
        except ClientError as e:
            logger.error(f"Error listing files in prefix {self.prefix}: {e}")
            return []

class UploadState:
    def __init__(self, checkpoint_file: str, s3_manager: S3BucketManager):
        self.checkpoint_file = KAGGLE_WORKING_DIR / checkpoint_file
        self.s3_manager = s3_manager
        self.state = self._load_state()
        self.last_checkpoint = time.time()
        
    def _load_state(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted checkpoint file, starting fresh")
                return self._create_initial_state()
        return self._create_initial_state()
    
    def _create_initial_state(self) -> Dict:
        return {
            'uploaded_files': self.s3_manager.get_existing_files(),
            'failed_files': [],
            'current_batch': [],
            'last_processed_path': None,
            'start_time': datetime.now().isoformat(),
            'total_files_processed': 0,
            'total_files_uploaded': 0,
            'total_files_failed': 0
        }
    
    def save(self):
        """Save current state to checkpoint file"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.state, f)
            self.last_checkpoint = time.time()
            logger.info(f"Checkpoint saved: {len(self.state['uploaded_files'])} files uploaded")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def should_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint"""
        return time.time() - self.last_checkpoint >= CHECKPOINT_INTERVAL

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def file_exists_in_s3(s3_bucket, s3_key):
    """Check if file exists in S3 and verify its hash if it does."""
    try:
        response = s3.head_object(Bucket=s3_bucket, Key=s3_key)
        return True
    except:
        return False

def upload_file(local_path, s3_bucket, s3_key, state: UploadState, retry_count=0):
    """Upload a file to S3 with error handling and progress tracking."""
    try:
        # Check if file already exists
        if file_exists_in_s3(s3_bucket, s3_key):
            logger.info(f"Skipping {local_path} - already exists in S3")
            return True

        # Upload with progress tracking
        file_size = os.path.getsize(local_path)
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {local_path.name}") as pbar:
            s3.upload_file(
                str(local_path),
                s3_bucket,
                s3_key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
        return True
    except Exception as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Upload failed for {local_path}, retrying ({retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            return upload_file(local_path, s3_bucket, s3_key, state, retry_count + 1)
        else:
            logger.error(f"Failed to upload {local_path} after {MAX_RETRIES} retries: {e}")
            state.state['failed_files'].append(str(local_path))
            state.state['total_files_failed'] += 1
            return False

def process_batch(state: UploadState):
    """Process a batch of files"""
    for local_path, s3_key in tqdm(state.state['current_batch'], desc="Processing batch"):
        success = upload_file(Path(local_path), S3_BUCKET, s3_key, state)
        if success:
            state.state['uploaded_files'].append(s3_key)
            state.state['total_files_uploaded'] += 1
    state.state['current_batch'] = []
    state.save()

def main():
    # Verify Kaggle environment
    if not KAGGLE_WORKING_DIR.exists():
        raise RuntimeError("Not running in Kaggle environment")
    
    if not DATASET_PATH.exists():
        raise RuntimeError(f"Dataset path not found: {DATASET_PATH}")
    
    # Initialize S3 bucket manager
    s3_manager = S3BucketManager(S3_BUCKET, S3_PREFIX)
    
    # Validate S3 bucket and prefix
    if not s3_manager.validate_bucket():
        raise RuntimeError(f"Failed to validate S3 bucket: {S3_BUCKET}")
        
    if not s3_manager.validate_prefix():
        logger.info(f"Prefix {S3_PREFIX} does not exist, creating it...")
        if not s3_manager.create_prefix():
            raise RuntimeError(f"Failed to create prefix: {S3_PREFIX}")
    
    # Initialize state
    state = UploadState('upload_progress.json', s3_manager)
    logger.info(f"Resuming upload from {len(state.state['uploaded_files'])} previously uploaded files")
    
    # Process files
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            local_path = Path(root) / file
            s3_key = S3_PREFIX + str(local_path.relative_to(DATASET_PATH))
            
            # Skip if already uploaded
            if s3_key in state.state['uploaded_files']:
                continue
                
            state.state['current_batch'].append((str(local_path), s3_key))
            state.state['last_processed_path'] = str(local_path)
            state.state['total_files_processed'] += 1
            
            # Process batch when it reaches the batch size
            if len(state.state['current_batch']) >= BATCH_SIZE:
                process_batch(state)
            
            # Save checkpoint periodically
            if state.should_checkpoint():
                state.save()
    
    # Process remaining files
    if state.state['current_batch']:
        process_batch(state)
    
    # Final checkpoint
    state.save()
    
    logger.info(f"""
    Upload Summary:
    - Total files processed: {state.state['total_files_processed']}
    - Total files uploaded: {state.state['total_files_uploaded']}
    - Total files failed: {state.state['total_files_failed']}
    - Failed files: {state.state['failed_files']}
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Upload interrupted by user")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

# --- SECURITY NOTE ---
# Remove your credentials from this notebook after use!
# Or, upload a private Kaggle dataset with your credentials and load them securely. 