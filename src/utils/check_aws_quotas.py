"""
AWS Service Quota Checker

This module provides utilities to check AWS service quotas locally using the AWS CLI.
It focuses on quotas relevant to the seismic waveform inversion project.
"""

import subprocess
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AWSQuotaChecker:
    """Utility class to check AWS service quotas."""
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize the quota checker.
        
        Args:
            region: AWS region to check quotas for
        """
        self.region = region
        self.service_quotas_client = None
        self.ec2_client = None
        
        try:
            self.service_quotas_client = boto3.client('service-quotas', region_name=region)
            self.ec2_client = boto3.client('ec2', region_name=region)
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI or set environment variables.")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
    
    def check_credentials(self) -> bool:
        """
        Check if AWS credentials are properly configured.
        
        Returns:
            bool: True if credentials are valid
        """
        try:
            # Try to get caller identity
            sts_client = boto3.client('sts')
            identity = sts_client.get_caller_identity()
            logger.info(f"✅ AWS credentials valid for account: {identity['Account']}")
            logger.info(f"✅ User/role: {identity['Arn']}")
            return True
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found")
            return False
        except Exception as e:
            logger.error(f"❌ AWS credentials error: {e}")
            return False
    
    def get_service_quota(self, service_code: str, quota_code: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific service quota.
        
        Args:
            service_code: AWS service code (e.g., 'ec2', 's3')
            quota_code: Quota code
            
        Returns:
            Dict containing quota information or None if not found
        """
        if not self.service_quotas_client:
            return None
            
        try:
            response = self.service_quotas_client.get_service_quota(
                ServiceCode=service_code,
                QuotaCode=quota_code
            )
            return response['Quota']
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchResourceException':
                logger.warning(f"Quota {quota_code} not found for service {service_code}")
            else:
                logger.error(f"Error getting quota {quota_code}: {e}")
            return None
    
    def list_service_quotas(self, service_code: str) -> List[Dict[str, Any]]:
        """
        List all quotas for a service.
        
        Args:
            service_code: AWS service code
            
        Returns:
            List of quota dictionaries
        """
        if not self.service_quotas_client:
            return []
            
        quotas = []
        try:
            paginator = self.service_quotas_client.get_paginator('list_service_quotas')
            for page in paginator.paginate(ServiceCode=service_code):
                quotas.extend(page['Quotas'])
        except ClientError as e:
            logger.error(f"Error listing quotas for {service_code}: {e}")
        
        return quotas
    
    def check_ec2_quotas(self) -> Dict[str, Any]:
        """
        Check EC2-related quotas important for ML training.
        
        Returns:
            Dict containing EC2 quota information
        """
        logger.info("🔍 Checking EC2 quotas...")
        
        quotas = {}
        
        # Important EC2 quotas for ML training
        ec2_quotas = {
            'Running On-Demand G and VT instances': 'L-85EED4F2',
            'Running On-Demand P instances': 'L-417A185B',
            'Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances': 'L-1216C47A',
            'VPCs per Region': 'L-F678F1CE',
            'VPC Elastic IPs': 'L-0263D0A3',
            'EBS General Purpose (SSD) volume storage (GiB)': 'L-D18FCD1D',
            'EBS Throughput Optimized HDD (st1) volume storage (GiB)': 'L-7425B8A9'
        }
        
        for quota_name, quota_code in ec2_quotas.items():
            quota = self.get_service_quota('ec2', quota_code)
            if quota:
                quotas[quota_name] = {
                    'value': quota['Value'],
                    'adjustable': quota['Adjustable'],
                    'unit': quota.get('Unit', 'N/A')
                }
        
        return quotas
    
    def check_s3_quotas(self) -> Dict[str, Any]:
        """
        Check S3-related quotas.
        
        Returns:
            Dict containing S3 quota information
        """
        logger.info("🔍 Checking S3 quotas...")
        
        quotas = {}
        
        # S3 quotas
        s3_quotas = {
            'Buckets per account': 'L-DC2B2D3D',
            'Objects per bucket': 'L-432A5B9D'
        }
        
        for quota_name, quota_code in s3_quotas.items():
            quota = self.get_service_quota('s3', quota_code)
            if quota:
                quotas[quota_name] = {
                    'value': quota['Value'],
                    'adjustable': quota['Adjustable'],
                    'unit': quota.get('Unit', 'N/A')
                }
        
        return quotas
    
    def check_sagemaker_quotas(self) -> Dict[str, Any]:
        """
        Check SageMaker-related quotas.
        
        Returns:
            Dict containing SageMaker quota information
        """
        logger.info("🔍 Checking SageMaker quotas...")
        
        quotas = {}
        
        # SageMaker quotas
        sagemaker_quotas = {
            'Training jobs per account': 'L-85EED4F2',
            'Notebook instances per account': 'L-85EED4F2',
            'Processing jobs per account': 'L-85EED4F2'
        }
        
        for quota_name, quota_code in sagemaker_quotas.items():
            quota = self.get_service_quota('sagemaker', quota_code)
            if quota:
                quotas[quota_name] = {
                    'value': quota['Value'],
                    'adjustable': quota['Adjustable'],
                    'unit': quota.get('Unit', 'N/A')
                }
        
        return quotas
    
    def check_current_usage(self) -> Dict[str, Any]:
        """
        Check current resource usage.
        
        Returns:
            Dict containing current usage information
        """
        logger.info("🔍 Checking current resource usage...")
        
        usage = {}
        
        try:
            # Check EC2 instances
            response = self.ec2_client.describe_instances(
                Filters=[{'Name': 'instance-state-name', 'Values': ['running', 'pending']}]
            )
            
            instance_count = sum(len(reservation['Instances']) for reservation in response['Reservations'])
            usage['Running EC2 instances'] = instance_count
            
            # Check EBS volumes
            response = self.ec2_client.describe_volumes()
            usage['EBS volumes'] = len(response['Volumes'])
            
            # Calculate total EBS storage
            total_storage = sum(volume['Size'] for volume in response['Volumes'])
            usage['Total EBS storage (GB)'] = total_storage
            
        except Exception as e:
            logger.error(f"Error checking current usage: {e}")
        
        return usage
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """
        Run a comprehensive check of all relevant quotas and usage.
        
        Returns:
            Dict containing comprehensive quota and usage information
        """
        logger.info("🚀 Starting comprehensive AWS quota check...")
        
        results = {
            'credentials_valid': self.check_credentials(),
            'region': self.region,
            'ec2_quotas': {},
            's3_quotas': {},
            'sagemaker_quotas': {},
            'current_usage': {},
            'recommendations': []
        }
        
        if not results['credentials_valid']:
            results['recommendations'].append("Configure AWS credentials using 'aws configure' or environment variables")
            return results
        
        # Check quotas
        results['ec2_quotas'] = self.check_ec2_quotas()
        results['s3_quotas'] = self.check_s3_quotas()
        results['sagemaker_quotas'] = self.check_sagemaker_quotas()
        results['current_usage'] = self.check_current_usage()
        
        # Generate recommendations
        self._generate_recommendations(results)
        
        return results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> None:
        """
        Generate recommendations based on quota and usage information.
        
        Args:
            results: Results dictionary to update with recommendations
        """
        recommendations = []
        
        # Check EC2 instance quotas
        if 'Running On-Demand G and VT instances' in results['ec2_quotas']:
            quota = results['ec2_quotas']['Running On-Demand G and VT instances']['value']
            usage = results['current_usage'].get('Running EC2 instances', 0)
            
            if usage >= quota * 0.8:  # 80% threshold
                recommendations.append(f"⚠️ EC2 instance quota nearly reached: {usage}/{quota}")
        
        # Check EBS storage
        if 'EBS General Purpose (SSD) volume storage (GiB)' in results['ec2_quotas']:
            quota = results['ec2_quotas']['EBS General Purpose (SSD) volume storage (GiB)']['value']
            usage = results['current_usage'].get('Total EBS storage (GB)', 0)
            
            if usage >= quota * 0.8:  # 80% threshold
                recommendations.append(f"⚠️ EBS storage quota nearly reached: {usage}/{quota} GB")
        
        # Check S3 buckets
        if 'Buckets per account' in results['s3_quotas']:
            quota = results['s3_quotas']['Buckets per account']['value']
            if quota < 100:
                recommendations.append(f"⚠️ S3 bucket quota is low: {quota} buckets")
        
        results['recommendations'] = recommendations

def print_quota_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted quota report.
    
    Args:
        results: Results from comprehensive quota check
    """
    print("="*60)
    print("AWS SERVICE QUOTA REPORT")
    print("="*60)
    
    print(f"Region: {results['region']}")
    print(f"Credentials: {'✅ Valid' if results['credentials_valid'] else '❌ Invalid'}")
    print()
    
    if not results['credentials_valid']:
        print("❌ Cannot check quotas without valid AWS credentials")
        return
    
    # EC2 Quotas
    print("🔧 EC2 QUOTAS:")
    print("-" * 30)
    for quota_name, quota_info in results['ec2_quotas'].items():
        print(f"  {quota_name}: {quota_info['value']} {quota_info['unit']}")
        if quota_info['adjustable']:
            print(f"    (Adjustable)")
    print()
    
    # S3 Quotas
    print("📦 S3 QUOTAS:")
    print("-" * 30)
    for quota_name, quota_info in results['s3_quotas'].items():
        print(f"  {quota_name}: {quota_info['value']} {quota_info['unit']}")
        if quota_info['adjustable']:
            print(f"    (Adjustable)")
    print()
    
    # Current Usage
    print("📊 CURRENT USAGE:")
    print("-" * 30)
    for resource, usage in results['current_usage'].items():
        print(f"  {resource}: {usage}")
    print()
    
    # Recommendations
    if results['recommendations']:
        print("💡 RECOMMENDATIONS:")
        print("-" * 30)
        for rec in results['recommendations']:
            print(f"  {rec}")
    else:
        print("✅ No immediate action required")
    
    print("="*60)

def main():
    """Main function to run quota check."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check AWS service quotas")
    parser.add_argument('--region', default='us-east-1', help='AWS region to check')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    # Run quota check
    checker = AWSQuotaChecker(region=args.region)
    results = checker.run_comprehensive_check()
    
    # Output results
    if args.output == 'json':
        print(json.dumps(results, indent=2))
    else:
        print_quota_report(results)

if __name__ == "__main__":
    main() 