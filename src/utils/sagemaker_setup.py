"""
SageMaker Setup for Seismic Data Preprocessing

This module provides setup instructions and configuration for running the seismic data
preprocessing pipeline on AWS SageMaker. It includes:
1. SageMaker instance configuration
2. Data access setup for S3
3. Environment setup
4. Dependencies installation
"""

import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.session import Session
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_sagemaker_environment():
    """
    Set up the SageMaker environment with necessary configurations.
    
    Returns:
        tuple: (sagemaker_session, role, bucket)
    """
    try:
        # Initialize SageMaker session
        sagemaker_session = Session()
        
        # Get the SageMaker execution role
        role = get_execution_role()
        
        # Get the default bucket
        bucket = sagemaker_session.default_bucket()
        
        logger.info(f"Using SageMaker bucket: {bucket}")
        logger.info(f"Using SageMaker role: {role}")
        
        return sagemaker_session, role, bucket
        
    except Exception as e:
        logger.error(f"Error setting up SageMaker environment: {str(e)}")
        raise

def create_sagemaker_notebook():
    """
    Create a SageMaker notebook instance with the specified configuration.
    """
    try:
        # Initialize SageMaker client
        sagemaker_client = boto3.client('sagemaker')
        
        # Notebook instance configuration
        notebook_config = {
            'NotebookInstanceName': 'seismic-preprocessing',
            'InstanceType': 'ml.c5.2xlarge',  # CPU-only instance
            'RoleArn': get_execution_role(),
            'SubnetId': os.environ.get('SUBNET_ID'),  # Optional: specify subnet
            'SecurityGroups': [os.environ.get('SECURITY_GROUP_ID')],  # Optional: specify security group
            'VolumeSizeInGB': 100,  # Adjust based on your data size
            'DirectInternetAccess': True,
            'RootAccess': True,
            'Tags': [
                {
                    'Key': 'Project',
                    'Value': 'SeismicPreprocessing'
                }
            ]
        }
        
        # Create the notebook instance
        response = sagemaker_client.create_notebook_instance(**notebook_config)
        
        logger.info(f"Notebook instance creation initiated: {response['NotebookInstanceArn']}")
        return response
        
    except Exception as e:
        logger.error(f"Error creating SageMaker notebook: {str(e)}")
        raise

def setup_data_access(bucket: str):
    """
    Set up data access configuration for S3 bucket.
    
    Args:
        bucket: S3 bucket name
    """
    try:
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # Create bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=bucket)
        except:
            s3_client.create_bucket(Bucket=bucket)
        
        # Set up bucket policy for SageMaker access
        bucket_policy = {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Sid': 'SageMakerAccess',
                    'Effect': 'Allow',
                    'Principal': {
                        'Service': 'sagemaker.amazonaws.com'
                    },
                    'Action': [
                        's3:GetObject',
                        's3:PutObject',
                        's3:ListBucket'
                    ],
                    'Resource': [
                        f'arn:aws:s3:::{bucket}',
                        f'arn:aws:s3:::{bucket}/*'
                    ]
                }
            ]
        }
        
        s3_client.put_bucket_policy(
            Bucket=bucket,
            Policy=json.dumps(bucket_policy)
        )
        
        logger.info(f"Data access configured for bucket: {bucket}")
        
    except Exception as e:
        logger.error(f"Error setting up data access: {str(e)}")
        raise

def main():
    """
    Main function to set up SageMaker environment and create notebook instance.
    """
    try:
        # Set up SageMaker environment
        sagemaker_session, role, bucket = setup_sagemaker_environment()
        
        # Set up data access
        setup_data_access(bucket)
        
        # Create notebook instance
        create_sagemaker_notebook()
        
        logger.info("SageMaker setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main setup: {str(e)}")
        raise

if __name__ == "__main__":
    main() 