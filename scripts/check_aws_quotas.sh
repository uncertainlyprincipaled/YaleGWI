#!/bin/bash

# AWS Service Quota Checker Script
# This script checks AWS service quotas relevant to the seismic waveform inversion project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
REGION="us-east-1"
OUTPUT_FORMAT="text"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "error")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "info")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

# Function to show usage
show_usage() {
    echo "AWS Service Quota Checker"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --region REGION    AWS region to check (default: us-east-1)"
    echo "  -o, --output FORMAT    Output format: text or json (default: text)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Check quotas for us-east-1"
    echo "  $0 -r us-west-2                       # Check quotas for us-west-2"
    echo "  $0 -r us-east-1 -o json               # Get JSON output"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_status "error" "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate output format
if [[ "$OUTPUT_FORMAT" != "text" && "$OUTPUT_FORMAT" != "json" ]]; then
    print_status "error" "Invalid output format: $OUTPUT_FORMAT. Use 'text' or 'json'"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_status "error" "AWS CLI is not installed. Please install it first."
    echo "Installation instructions: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    print_status "error" "AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

print_status "info" "Checking AWS service quotas for region: $REGION"

# Get account information
ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
USER_ARN=$(aws sts get-caller-identity --query 'Arn' --output text)

print_status "success" "AWS Account: $ACCOUNT_ID"
print_status "success" "User/Role: $USER_ARN"

# Check if Python script exists
SCRIPT_PATH="src/utils/check_aws_quotas.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    print_status "error" "Quota checker script not found: $SCRIPT_PATH"
    print_status "info" "Make sure you're running this from the project root directory"
    exit 1
fi

# Run the Python quota checker
print_status "info" "Running comprehensive quota check..."

if [[ "$OUTPUT_FORMAT" == "json" ]]; then
    python -m src.utils.check_aws_quotas --region "$REGION" --output json
else
    python -m src.utils.check_aws_quotas --region "$REGION" --output text
fi

print_status "success" "Quota check completed!"

# Additional useful commands
echo ""
print_status "info" "Additional useful AWS CLI commands:"
echo "  # Check current EC2 instances"
echo "  aws ec2 describe-instances --region $REGION --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name]' --output table"
echo ""
echo "  # Check EBS volumes"
echo "  aws ec2 describe-volumes --region $REGION --query 'Volumes[*].[VolumeId,Size,VolumeType,State]' --output table"
echo ""
echo "  # Check S3 buckets"
echo "  aws s3 ls"
echo ""
echo "  # Check SageMaker resources"
echo "  aws sagemaker list-notebook-instances --region $REGION"
echo "  aws sagemaker list-training-jobs --region $REGION --max-items 10" 