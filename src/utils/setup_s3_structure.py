import boto3
import logging

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_s3_structure():
    """Create necessary folders in S3 bucket while preserving raw/"""
    s3 = boto3.client('s3')
    bucket = 'yale-gwi'
    
    # Create empty folders (S3 doesn't have real folders, so we create empty objects)
    folders = ['checkpoints/', 'logs/']
    
    for folder in folders:
        try:
            # Create empty object to represent folder
            s3.put_object(
                Bucket=bucket,
                Key=folder,
                Body=''
            )
            logger.info(f"Created folder: {folder}")
        except Exception as e:
            logger.error(f"Error creating folder {folder}: {e}")

if __name__ == "__main__":
    setup_s3_structure()
