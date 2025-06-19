"""
EBS Volume Cleanup Utility

This module provides utilities to clean up EBS volumes with advanced features
like cost estimation, detailed reporting, and batch operations.
"""

import boto3
import argparse
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError, NoCredentialsError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EBSVolumeCleaner:
    """Utility class to clean up EBS volumes."""
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize the EBS volume cleaner.
        
        Args:
            region: AWS region to operate in
        """
        self.region = region
        self.ec2_client = None
        
        try:
            self.ec2_client = boto3.client('ec2', region_name=region)
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI or set environment variables.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize EC2 client: {e}")
            raise
    
    def check_credentials(self) -> bool:
        """
        Check if AWS credentials are properly configured.
        
        Returns:
            bool: True if credentials are valid
        """
        try:
            sts_client = boto3.client('sts')
            identity = sts_client.get_caller_identity()
            logger.info(f"✅ AWS credentials valid for account: {identity['Account']}")
            return True
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found")
            return False
        except Exception as e:
            logger.error(f"❌ AWS credentials error: {e}")
            return False
    
    def get_volume_details(self, volume_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a volume.
        
        Args:
            volume_id: EBS volume ID
            
        Returns:
            Dict containing volume details or None if not found
        """
        try:
            response = self.ec2_client.describe_volumes(VolumeIds=[volume_id])
            if response['Volumes']:
                return response['Volumes'][0]
            return None
        except ClientError as e:
            logger.error(f"Error getting volume details for {volume_id}: {e}")
            return None
    
    def list_volumes(self, 
                    unattached_only: bool = False,
                    older_than_days: Optional[int] = None,
                    volume_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List EBS volumes with optional filters.
        
        Args:
            unattached_only: Only list unattached volumes
            older_than_days: Only list volumes older than N days
            volume_type: Filter by volume type (gp2, gp3, io1, etc.)
            
        Returns:
            List of volume dictionaries
        """
        filters = []
        
        if unattached_only:
            filters.append({'Name': 'status', 'Values': ['available']})
        
        if volume_type:
            filters.append({'Name': 'volume-type', 'Values': [volume_type]})
        
        try:
            response = self.ec2_client.describe_volumes(Filters=filters)
            volumes = response['Volumes']
            
            # Filter by age if specified
            if older_than_days:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                volumes = [
                    vol for vol in volumes 
                    if datetime.fromisoformat(vol['CreateTime'].replace('Z', '+00:00')) < cutoff_date
                ]
            
            return volumes
            
        except ClientError as e:
            logger.error(f"Error listing volumes: {e}")
            return []
    
    def is_volume_safe_to_delete(self, volume_id: str) -> tuple[bool, str]:
        """
        Check if a volume is safe to delete.
        
        Args:
            volume_id: EBS volume ID
            
        Returns:
            Tuple of (is_safe, reason)
        """
        volume = self.get_volume_details(volume_id)
        if not volume:
            return False, "Volume not found"
        
        # Check if attached
        if volume['Attachments']:
            instance_id = volume['Attachments'][0]['InstanceId']
            return False, f"Volume is attached to instance {instance_id}"
        
        # Check if has snapshots
        if volume['SnapshotId']:
            return False, f"Volume has snapshot {volume['SnapshotId']}"
        
        # Check state
        if volume['State'] != 'available':
            return False, f"Volume is in state '{volume['State']}'"
        
        return True, "Safe to delete"
    
    def estimate_cost_savings(self, volumes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate cost savings from deleting volumes.
        
        Args:
            volumes: List of volume dictionaries
            
        Returns:
            Dict containing cost estimation
        """
        # Rough cost estimates per GB per month (varies by region and type)
        cost_per_gb_month = {
            'gp2': 0.10,  # General Purpose SSD
            'gp3': 0.08,  # General Purpose SSD v3
            'io1': 0.125, # Provisioned IOPS SSD
            'io2': 0.125, # Provisioned IOPS SSD v2
            'st1': 0.045, # Throughput Optimized HDD
            'sc1': 0.015, # Cold HDD
            'standard': 0.05  # Magnetic
        }
        
        total_size_gb = 0
        total_cost_per_month = 0
        volume_types = {}
        
        for volume in volumes:
            size_gb = volume['Size']
            vol_type = volume['VolumeType']
            
            total_size_gb += size_gb
            
            # Get cost for this volume type
            cost_per_gb = cost_per_gb_month.get(vol_type, 0.10)  # Default to gp2 cost
            volume_cost = size_gb * cost_per_gb
            total_cost_per_month += volume_cost
            
            volume_types[vol_type] = volume_types.get(vol_type, 0) + size_gb
        
        return {
            'total_size_gb': total_size_gb,
            'total_cost_per_month': total_cost_per_month,
            'total_cost_per_year': total_cost_per_month * 12,
            'volume_types': volume_types,
            'volume_count': len(volumes)
        }
    
    def delete_volume(self, volume_id: str, dry_run: bool = True) -> tuple[bool, str]:
        """
        Delete a volume.
        
        Args:
            volume_id: EBS volume ID
            dry_run: If True, don't actually delete
            
        Returns:
            Tuple of (success, message)
        """
        if dry_run:
            return True, f"[DRY RUN] Would delete volume {volume_id}"
        
        is_safe, reason = self.is_volume_safe_to_delete(volume_id)
        if not is_safe:
            return False, f"Cannot delete {volume_id}: {reason}"
        
        try:
            self.ec2_client.delete_volume(VolumeId=volume_id)
            return True, f"Successfully deleted volume {volume_id}"
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            return False, f"Failed to delete {volume_id}: {error_code} - {error_message}"
    
    def delete_volumes_batch(self, 
                           volume_ids: List[str], 
                           dry_run: bool = True,
                           force: bool = False) -> Dict[str, Any]:
        """
        Delete multiple volumes in batch.
        
        Args:
            volume_ids: List of volume IDs to delete
            dry_run: If True, don't actually delete
            force: Skip safety checks
            
        Returns:
            Dict containing results
        """
        results = {
            'successful': [],
            'failed': [],
            'skipped': [],
            'total': len(volume_ids)
        }
        
        for volume_id in volume_ids:
            if not force:
                is_safe, reason = self.is_volume_safe_to_delete(volume_id)
                if not is_safe:
                    results['skipped'].append({'volume_id': volume_id, 'reason': reason})
                    continue
            
            success, message = self.delete_volume(volume_id, dry_run)
            if success:
                results['successful'].append({'volume_id': volume_id, 'message': message})
            else:
                results['failed'].append({'volume_id': volume_id, 'message': message})
        
        return results
    
    def cleanup_unattached_volumes(self, 
                                 older_than_days: Optional[int] = None,
                                 dry_run: bool = True,
                                 force: bool = False) -> Dict[str, Any]:
        """
        Clean up all unattached volumes.
        
        Args:
            older_than_days: Only delete volumes older than N days
            dry_run: If True, don't actually delete
            force: Skip safety checks
            
        Returns:
            Dict containing cleanup results
        """
        volumes = self.list_volumes(
            unattached_only=True,
            older_than_days=older_than_days
        )
        
        if not volumes:
            return {
                'message': 'No unattached volumes found',
                'volumes': [],
                'cost_savings': None,
                'results': {'successful': [], 'failed': [], 'skipped': [], 'total': 0}
            }
        
        volume_ids = [vol['VolumeId'] for vol in volumes]
        cost_savings = self.estimate_cost_savings(volumes)
        results = self.delete_volumes_batch(volume_ids, dry_run, force)
        
        return {
            'volumes': volumes,
            'cost_savings': cost_savings,
            'results': results
        }
    
    def print_volume_table(self, volumes: List[Dict[str, Any]]) -> None:
        """
        Print volumes in a formatted table.
        
        Args:
            volumes: List of volume dictionaries
        """
        if not volumes:
            print("No volumes found.")
            return
        
        print(f"\n{'Volume ID':<15} {'Size(GB)':<10} {'Type':<8} {'State':<12} {'Created':<20} {'Instance':<15}")
        print("-" * 85)
        
        for volume in volumes:
            volume_id = volume['VolumeId']
            size = volume['Size']
            vol_type = volume['VolumeType']
            state = volume['State']
            created = volume['CreateTime'].strftime('%Y-%m-%d %H:%M:%S')
            instance = volume['Attachments'][0]['InstanceId'] if volume['Attachments'] else 'N/A'
            
            print(f"{volume_id:<15} {size:<10} {vol_type:<8} {state:<12} {created:<20} {instance:<15}")
    
    def print_cost_analysis(self, cost_savings: Dict[str, Any]) -> None:
        """
        Print cost analysis.
        
        Args:
            cost_savings: Cost savings dictionary
        """
        if not cost_savings:
            return
        
        print(f"\n💰 Cost Analysis:")
        print(f"  Total volumes: {cost_savings['volume_count']}")
        print(f"  Total size: {cost_savings['total_size_gb']:,} GB")
        print(f"  Monthly cost: ${cost_savings['total_cost_per_month']:.2f}")
        print(f"  Annual cost: ${cost_savings['total_cost_per_year']:.2f}")
        
        if cost_savings['volume_types']:
            print(f"  Volume types:")
            for vol_type, size in cost_savings['volume_types'].items():
                print(f"    {vol_type}: {size:,} GB")

def main():
    """Main function to run EBS cleanup."""
    parser = argparse.ArgumentParser(description="Clean up EBS volumes")
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--volume-ids', nargs='+', help='Specific volume IDs to delete')
    parser.add_argument('--all-unattached', action='store_true', help='Delete all unattached volumes')
    parser.add_argument('--older-than', type=int, help='Only delete volumes older than N days')
    parser.add_argument('--volume-type', help='Filter by volume type')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run mode (default)')
    parser.add_argument('--execute', action='store_true', help='Actually perform deletions')
    parser.add_argument('--force', action='store_true', help='Skip safety checks')
    parser.add_argument('--list-only', action='store_true', help='List volumes without deleting')
    parser.add_argument('--output', choices=['text', 'json'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    # Set dry run mode
    if args.execute:
        args.dry_run = False
    
    try:
        # Initialize cleaner
        cleaner = EBSVolumeCleaner(region=args.region)
        
        # Check credentials
        if not cleaner.check_credentials():
            return
        
        # List only mode
        if args.list_only:
            volumes = cleaner.list_volumes(
                unattached_only=args.all_unattached,
                older_than_days=args.older_than,
                volume_type=args.volume_type
            )
            
            if args.output == 'json':
                print(json.dumps(volumes, default=str, indent=2))
            else:
                cleaner.print_volume_table(volumes)
            return
        
        # Delete specific volumes
        if args.volume_ids:
            results = cleaner.delete_volumes_batch(
                args.volume_ids, 
                dry_run=args.dry_run,
                force=args.force
            )
            
            if args.output == 'json':
                print(json.dumps(results, indent=2))
            else:
                print(f"Results: {len(results['successful'])} successful, {len(results['failed'])} failed, {len(results['skipped'])} skipped")
        else:
            # Clean up unattached volumes
            results = cleaner.cleanup_unattached_volumes(
                older_than_days=args.older_than,
                dry_run=args.dry_run,
                force=args.force
            )
            
            if args.output == 'json':
                print(json.dumps(results, default=str, indent=2))
            else:
                if results['volumes']:
                    cleaner.print_volume_table(results['volumes'])
                    cleaner.print_cost_analysis(results['cost_savings'])
                
                print(f"\nResults: {len(results['results']['successful'])} successful, {len(results['results']['failed'])} failed, {len(results['results']['skipped'])} skipped")
        
        if args.dry_run:
            print("\n⚠️  This was a dry run. Use --execute to actually delete volumes.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main() 