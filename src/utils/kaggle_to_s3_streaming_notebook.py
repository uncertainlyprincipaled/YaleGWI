import os
import boto3
from pathlib import Path
from tqdm import tqdm

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

# --- DATASET PATH ---
DATASET_PATH = Path('/kaggle/input/waveform-inversion')

# --- S3 CLIENT ---
s3 = boto3.client('s3')

def upload_file(local_path, s3_bucket, s3_key):
    try:
        s3.upload_file(str(local_path), s3_bucket, s3_key)
        return True
    except Exception as e:
        print(f"Failed to upload {local_path}: {e}")
        return False

# --- STREAMING UPLOAD ---
file_count = 0
for root, dirs, files in os.walk(DATASET_PATH):
    for file in tqdm(files, desc=f"Uploading files from {root}"):
        local_path = Path(root) / file
        s3_key = S3_PREFIX + str(local_path.relative_to(DATASET_PATH))
        success = upload_file(local_path, S3_BUCKET, s3_key)
        if success:
            file_count += 1
print(f"Uploaded {file_count} files to s3://{S3_BUCKET}/{S3_PREFIX}")

# --- SECURITY NOTE ---
# Remove your credentials from this notebook after use!
# Or, upload a private Kaggle dataset with your credentials and load them securely. 