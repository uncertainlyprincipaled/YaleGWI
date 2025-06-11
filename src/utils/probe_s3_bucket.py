import boto3
import logging
from typing import List, Dict
from botocore.exceptions import ClientError

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- S3 CONFIG ---
S3_BUCKET = 'yale-gwi'
S3_PREFIX = 'raw/'  # Folder in S3 bucket

def list_bucket_contents(prefix: str = '', max_keys: int = 1000) -> List[Dict]:
    """List contents of the S3 bucket with optional prefix filtering."""
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                logger.info(f"Key: {obj['Key']}")
                logger.info(f"Size: {obj['Size']} bytes")
                logger.info(f"Last Modified: {obj['LastModified']}")
                logger.info("-" * 50)
            return response['Contents']
        else:
            logger.info(f"No objects found with prefix: {prefix}")
            return []
            
    except ClientError as e:
        logger.error(f"Error listing bucket contents: {e}")
        return []

def get_object_metadata(key: str) -> Dict:
    """Get metadata for a specific object in the bucket."""
    s3 = boto3.client('s3')
    try:
        response = s3.head_object(Bucket=S3_BUCKET, Key=key)
        logger.info(f"Metadata for {key}:")
        for key, value in response.items():
            logger.info(f"{key}: {value}")
        return response
    except ClientError as e:
        logger.error(f"Error getting object metadata: {e}")
        return {}

def count_objects_by_prefix(prefix: str = '') -> int:
    """Count number of objects with a specific prefix."""
    s3 = boto3.client('s3')
    try:
        paginator = s3.get_paginator('list_objects_v2')
        count = 0
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            if 'Contents' in page:
                count += len(page['Contents'])
        logger.info(f"Found {count} objects with prefix: {prefix}")
        return count
    except ClientError as e:
        logger.error(f"Error counting objects: {e}")
        return 0

def main():
    # List all objects in the bucket
    logger.info("Listing all objects in bucket:")
    list_bucket_contents()
    
    # Count objects in the raw prefix
    logger.info("\nCounting objects in raw prefix:")
    count_objects_by_prefix(S3_PREFIX)
    
    # Get metadata for a specific object (replace with actual key)
    # logger.info("\nGetting metadata for specific object:")
    # get_object_metadata("raw/example.txt")

if __name__ == "__main__":
    main() 