"""
S3 Debugging Utility

This script helps debug S3 setup and data structure issues.
Run this to understand what files are available in your S3 bucket.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

def debug_s3_setup():
    """Debug S3 setup and list available files."""
    print("🔍 S3 Setup Debugging")
    print("=" * 50)
    
    try:
        # Check AWS credentials
        print("1. Checking AWS credentials...")
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_region = os.environ.get('AWS_REGION')
        aws_bucket = os.environ.get('AWS_S3_BUCKET')
        
        if aws_access_key:
            print(f"   ✅ AWS Access Key ID: {aws_access_key[:8]}...")
        else:
            print("   ❌ AWS Access Key ID not found")
            
        if aws_secret_key:
            print(f"   ✅ AWS Secret Access Key: {aws_secret_key[:8]}...")
        else:
            print("   ❌ AWS Secret Access Key not found")
            
        if aws_region:
            print(f"   ✅ AWS Region: {aws_region}")
        else:
            print("   ❌ AWS Region not found")
            
        if aws_bucket:
            print(f"   ✅ S3 Bucket: {aws_bucket}")
        else:
            print("   ❌ S3 Bucket not found")
        
        # Test S3 connection
        print("\n2. Testing S3 connection...")
        from src.core.data_manager import DataManager
        data_manager = DataManager(use_s3=True)
        
        try:
            # Test basic S3 operations
            response = data_manager.s3.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            print(f"   ✅ Connected to S3. Available buckets: {buckets}")
            
            if aws_bucket in buckets:
                print(f"   ✅ Target bucket '{aws_bucket}' found")
            else:
                print(f"   ❌ Target bucket '{aws_bucket}' not found in available buckets")
                
        except Exception as e:
            print(f"   ❌ S3 connection failed: {e}")
            return False
        
        # List files in different prefixes
        print("\n3. Listing files in S3...")
        from src.core.config import CFG, FAMILY_FILE_MAP
        
        # Check raw data prefix
        raw_prefix = CFG.s3_paths.raw_prefix
        print(f"   Raw data prefix: {raw_prefix}")
        raw_files = data_manager.list_s3_files(raw_prefix)
        print(f"   Raw files found: {len(raw_files)}")
        if raw_files:
            print(f"   Sample raw files: {raw_files[:5]}")
        
        # Check family prefixes
        print(f"\n   Family prefixes:")
        for family in FAMILY_FILE_MAP.keys():
            try:
                family_prefix = CFG.s3_paths.families[family]
                family_files = data_manager.list_s3_files(family_prefix)
                print(f"   {family}: {len(family_files)} files")
                if family_files:
                    print(f"     Sample: {family_files[:2]}")
            except Exception as e:
                print(f"   {family}: Error - {e}")
        
        # Check preprocessed prefix
        print(f"\n   Preprocessed prefix: preprocessed")
        preprocessed_files = data_manager.list_s3_files("preprocessed")
        print(f"   Preprocessed files found: {len(preprocessed_files)}")
        if preprocessed_files:
            print(f"   Sample preprocessed files: {preprocessed_files[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 debugging failed: {e}")
        return False

def debug_family_structure(family: str = 'FlatVel_A'):
    """Debug the structure of a specific family."""
    print(f"\n🔍 Debugging family structure: {family}")
    print("=" * 50)
    
    try:
        from src.core.data_manager import DataManager
        from src.core.config import CFG, FAMILY_FILE_MAP
        
        data_manager = DataManager(use_s3=True)
        
        # Get family configuration
        family_config = FAMILY_FILE_MAP.get(family, {})
        print(f"Family config: {family_config}")
        
        # Get S3 family prefix
        s3_family_prefix = CFG.s3_paths.families[family]
        print(f"S3 family prefix: {s3_family_prefix}")
        
        # List all files in family prefix
        all_files = data_manager.list_s3_files(s3_family_prefix)
        print(f"Total files in family: {len(all_files)}")
        
        if all_files:
            print("All files:")
            for file in all_files:
                print(f"  {file}")
        
        # Try different subdirectory patterns
        patterns = ['', 'data/', 'model/', 'seis/', 'vel/']
        
        for pattern in patterns:
            prefix = f"{s3_family_prefix}/{pattern}"
            files = data_manager.list_s3_files(prefix)
            print(f"\nPattern '{pattern}': {len(files)} files")
            if files:
                print(f"  Sample: {files[:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Family debugging failed: {e}")
        return False

def main():
    """Main debugging function."""
    print("🚀 S3 Debugging Utility")
    print("=" * 50)
    
    # Debug S3 setup
    s3_ok = debug_s3_setup()
    
    if s3_ok:
        # Debug specific family
        debug_family_structure('FlatVel_A')
        
        print("\n💡 Recommendations:")
        print("1. If no files are found, check your S3 bucket structure")
        print("2. Ensure your AWS credentials have read access to the bucket")
        print("3. Verify the S3 prefixes in your config match your bucket structure")
        print("4. Try running the preprocessing with debug_mode=True to see detailed logs")
    else:
        print("\n❌ S3 setup issues detected. Please check:")
        print("1. AWS credentials are properly set")
        print("2. S3 bucket exists and is accessible")
        print("3. AWS region is correct")

if __name__ == "__main__":
    main() 