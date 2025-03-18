#!/bin/bash
# Script for cleaning up Google Cloud TPU VM after GenCast inference

set -e

# Get parameters from command line
PROJECT_ID="$1"
TPU_NAME="$2"
ZONE="$3"

if [[ -z "$PROJECT_ID" || -z "$TPU_NAME" || -z "$ZONE" ]]; then
  echo "Error: Missing required parameters"
  echo "Usage: $0 <project_id> <tpu_name> <zone>"
  exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
  echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
  exit 1
fi

# Check if TPU exists
TPU_STATUS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "NOT_FOUND")

if [[ "$TPU_STATUS" == "NOT_FOUND" ]]; then
  echo "TPU '$TPU_NAME' does not exist or is already deleted"
  exit 0
fi

# Delete the TPU VM
echo "Deleting TPU VM '$TPU_NAME' in zone '$ZONE'..."
gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet

echo "TPU VM '$TPU_NAME' deleted successfully"
exit 0