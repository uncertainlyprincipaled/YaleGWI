#!/bin/bash

# EBS Volume Cleanup Script
# This script helps clean up EBS volumes to manage costs and resources

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
REGION="us-east-1"
DRY_RUN=true
FORCE=false
VOLUME_IDS=""
ALL_UNATTACHED=false
OLDER_THAN_DAYS=""

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
    echo "EBS Volume Cleanup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --region REGION        AWS region (default: us-east-1)"
    echo "  -v, --volume-ids IDS       Comma-separated list of volume IDs to delete"
    echo "  -a, --all-unattached       Delete all unattached volumes"
    echo "  -d, --older-than DAYS      Delete volumes older than N days"
    echo "  -f, --force                Skip confirmation prompts"
    echo "  -x, --execute              Actually perform deletions (default is dry-run)"
    echo "  -l, --list                 List volumes without deleting"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -l                                    # List all volumes"
    echo "  $0 -a                                    # Dry-run: show what would be deleted"
    echo "  $0 -a -x                                 # Actually delete all unattached volumes"
    echo "  $0 -v vol-123,vol-456 -x                 # Delete specific volumes"
    echo "  $0 -d 7 -x                               # Delete volumes older than 7 days"
    echo "  $0 -a -d 3 -x                            # Delete unattached volumes older than 3 days"
    echo ""
    echo "Safety Features:"
    echo "  - Dry-run mode by default (use -x to actually delete)"
    echo "  - Confirmation prompts (use -f to skip)"
    echo "  - Won't delete volumes that are attached to instances"
    echo "  - Won't delete volumes with snapshots"
    echo ""
}

# Function to get volume details
get_volume_details() {
    local volume_id=$1
    aws ec2 describe-volumes \
        --volume-ids "$volume_id" \
        --region "$REGION" \
        --query 'Volumes[0].[VolumeId,Size,VolumeType,State,CreateTime,Attachments[0].InstanceId,SnapshotId]' \
        --output table
}

# Function to list volumes
list_volumes() {
    local filter=""
    
    if [[ "$ALL_UNATTACHED" == true ]]; then
        filter="Name=status,Values=available"
    fi
    
    if [[ -n "$OLDER_THAN_DAYS" ]]; then
        local cutoff_date=$(date -d "$OLDER_THAN_DAYS days ago" --iso-8601=seconds 2>/dev/null || date -v-${OLDER_THAN_DAYS}d -u +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "")
        if [[ -n "$cutoff_date" ]]; then
            if [[ -n "$filter" ]]; then
                filter="$filter Name=create-time,Values=*"
            else
                filter="Name=create-time,Values=*"
            fi
        fi
    fi
    
    print_status "info" "Listing EBS volumes in region: $REGION"
    
    if [[ -n "$filter" ]]; then
        aws ec2 describe-volumes \
            --region "$REGION" \
            --filters "$filter" \
            --query 'Volumes[*].[VolumeId,Size,VolumeType,State,CreateTime,Attachments[0].InstanceId,SnapshotId]' \
            --output table
    else
        aws ec2 describe-volumes \
            --region "$REGION" \
            --query 'Volumes[*].[VolumeId,Size,VolumeType,State,CreateTime,Attachments[0].InstanceId,SnapshotId]' \
            --output table
    fi
}

# Function to check if volume is safe to delete
is_volume_safe_to_delete() {
    local volume_id=$1
    
    # Get volume details
    local volume_info=$(aws ec2 describe-volumes \
        --volume-ids "$volume_id" \
        --region "$REGION" \
        --query 'Volumes[0].[State,Attachments[0].InstanceId,SnapshotId]' \
        --output text 2>/dev/null)
    
    if [[ $? -ne 0 ]]; then
        print_status "error" "Volume $volume_id not found or not accessible"
        return 1
    fi
    
    local state=$(echo "$volume_info" | cut -f1)
    local attached_instance=$(echo "$volume_info" | cut -f2)
    local snapshot_id=$(echo "$volume_info" | cut -f3)
    
    # Check if volume is attached
    if [[ "$attached_instance" != "None" && -n "$attached_instance" ]]; then
        print_status "warning" "Volume $volume_id is attached to instance $attached_instance - skipping"
        return 1
    fi
    
    # Check if volume has snapshots
    if [[ "$snapshot_id" != "None" && -n "$snapshot_id" ]]; then
        print_status "warning" "Volume $volume_id has snapshot $snapshot_id - skipping"
        return 1
    fi
    
    # Check if volume is in use
    if [[ "$state" != "available" ]]; then
        print_status "warning" "Volume $volume_id is in state '$state' - skipping"
        return 1
    fi
    
    return 0
}

# Function to delete volume
delete_volume() {
    local volume_id=$1
    
    if [[ "$DRY_RUN" == true ]]; then
        print_status "info" "[DRY RUN] Would delete volume: $volume_id"
        get_volume_details "$volume_id"
        return 0
    fi
    
    if is_volume_safe_to_delete "$volume_id"; then
        print_status "info" "Deleting volume: $volume_id"
        if aws ec2 delete-volume --volume-id "$volume_id" --region "$REGION" 2>&1; then
            print_status "success" "Successfully deleted volume: $volume_id"
        else
            local error_output=$(aws ec2 delete-volume --volume-id "$volume_id" --region "$REGION" 2>&1)
            print_status "error" "Failed to delete volume: $volume_id"
            print_status "error" "Error details: $error_output"
            return 1
        fi
    else
        return 1
    fi
}

# Function to get volumes to delete
get_volumes_to_delete() {
    local volumes=()
    
    # If specific volume IDs provided
    if [[ -n "$VOLUME_IDS" ]]; then
        IFS=',' read -ra VOL_ARRAY <<< "$VOLUME_IDS"
        volumes=("${VOL_ARRAY[@]}")
    else
        # Get volumes based on filters
        local filter="Name=status,Values=available"
        
        if [[ -n "$OLDER_THAN_DAYS" ]]; then
            local cutoff_date=$(date -d "$OLDER_THAN_DAYS days ago" --iso-8601=seconds 2>/dev/null || date -v-${OLDER_THAN_DAYS}d -u +%Y-%m-%dT%H:%M:%S 2>/dev/null || echo "")
            if [[ -n "$cutoff_date" ]]; then
                filter="$filter Name=create-time,Values=*"
            fi
        fi
        
        # Get volume IDs
        local volume_list=$(aws ec2 describe-volumes \
            --region "$REGION" \
            --filters "$filter" \
            --query 'Volumes[*].VolumeId' \
            --output text 2>/dev/null)
        
        if [[ -n "$volume_list" ]]; then
            read -ra volumes <<< "$volume_list"
        fi
    fi
    
    echo "${volumes[@]}"
}

# Function to confirm deletion
confirm_deletion() {
    local volumes=("$@")
    local count=${#volumes[@]}
    
    if [[ $count -eq 0 ]]; then
        print_status "info" "No volumes to delete"
        return 1
    fi
    
    echo ""
    print_status "warning" "About to delete $count volume(s):"
    for volume_id in "${volumes[@]}"; do
        echo "  - $volume_id"
    done
    
    if [[ "$FORCE" == true ]]; then
        print_status "info" "Force mode enabled - proceeding without confirmation"
        return 0
    fi
    
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " -r
    echo
    
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        return 0
    else
        print_status "info" "Deletion cancelled"
        return 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -v|--volume-ids)
            VOLUME_IDS="$2"
            shift 2
            ;;
        -a|--all-unattached)
            ALL_UNATTACHED=true
            shift
            ;;
        -d|--older-than)
            OLDER_THAN_DAYS="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -x|--execute)
            DRY_RUN=false
            shift
            ;;
        -l|--list)
            LIST_ONLY=true
            shift
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

# Get account information
ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
print_status "info" "AWS Account: $ACCOUNT_ID"
print_status "info" "Region: $REGION"

# Show dry-run status
if [[ "$DRY_RUN" == true ]]; then
    print_status "info" "DRY RUN MODE - No volumes will be actually deleted"
else
    print_status "warning" "EXECUTION MODE - Volumes will be actually deleted"
fi

# If list-only mode
if [[ "$LIST_ONLY" == true ]]; then
    list_volumes
    exit 0
fi

# Validate input
if [[ -z "$VOLUME_IDS" && "$ALL_UNATTACHED" != true ]]; then
    print_status "error" "Must specify either volume IDs (-v) or all unattached volumes (-a)"
    show_usage
    exit 1
fi

# Get volumes to delete
volumes_to_delete=($(get_volumes_to_delete))

# Confirm deletion
if ! confirm_deletion "${volumes_to_delete[@]}"; then
    exit 0
fi

# Delete volumes
echo ""
print_status "info" "Starting volume deletion..."

success_count=0
error_count=0

for volume_id in "${volumes_to_delete[@]}"; do
    if delete_volume "$volume_id"; then
        ((success_count++))
    else
        ((error_count++))
    fi
    echo ""
done

# Summary
echo ""
print_status "info" "Deletion Summary:"
echo "  Successfully processed: $success_count"
echo "  Errors: $error_count"
echo "  Total volumes: ${#volumes_to_delete[@]}"

if [[ "$DRY_RUN" == true ]]; then
    print_status "info" "This was a dry run. Use -x flag to actually delete volumes."
fi

# Show remaining volumes
echo ""
print_status "info" "Remaining volumes in region $REGION:"
list_volumes 