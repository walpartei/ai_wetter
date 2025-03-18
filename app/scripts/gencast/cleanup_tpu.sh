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
  # Double-check with queued resources list to ensure it's not stuck in queue
  QUEUED_TPU=$(gcloud compute tpus queued-resources list --zone="$ZONE" --project="$PROJECT_ID" --filter="name=$TPU_NAME" --format="value(name)" 2>/dev/null || echo "")
  
  if [[ -z "$QUEUED_TPU" ]]; then
    echo "TPU '$TPU_NAME' does not exist or is already deleted"
    exit 0
  else
    echo "TPU '$TPU_NAME' found in queued resources, will be deleted"
  fi
fi

# First try deleting from queued resources
echo "Attempting to delete TPU from queued resources '$TPU_NAME' in zone '$ZONE'..."
gcloud compute tpus queued-resources delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Not in queued resources or already deleted"

# Also try normal TPU deletion in case it's been provisioned
echo "Deleting TPU VM '$TPU_NAME' in zone '$ZONE'..."
gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet 2>/dev/null || echo "Not a provisioned TPU VM or already deleted"

# Final verification that it's gone
sleep 2
TPU_FINAL_CHECK=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
QUEUED_FINAL_CHECK=$(gcloud compute tpus queued-resources list --zone="$ZONE" --project="$PROJECT_ID" --filter="name=$TPU_NAME" --format="value(name)" 2>/dev/null || echo "")

if [[ "$TPU_FINAL_CHECK" == "NOT_FOUND" && -z "$QUEUED_FINAL_CHECK" ]]; then
  echo "TPU '$TPU_NAME' deleted successfully"
else
  echo "WARNING: TPU '$TPU_NAME' may still exist in state: $TPU_FINAL_CHECK or in queue: $QUEUED_FINAL_CHECK"
  # Make one last attempt with force flag
  gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet --force 2>/dev/null || true
fi

exit 0