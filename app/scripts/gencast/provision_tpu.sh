#!/bin/bash
# Script for provisioning Google Cloud TPU VM for GenCast inference

set -e

# Get parameters from command line
PROJECT_ID="$1"
TPU_NAME="$2"
ZONE="$3"
ACCELERATOR_TYPE="$4"
RUNTIME_VERSION="$5"
SPOT="$6" # optional, set to "--spot" to request spot VM

if [[ -z "$PROJECT_ID" || -z "$TPU_NAME" || -z "$ZONE" || -z "$ACCELERATOR_TYPE" || -z "$RUNTIME_VERSION" ]]; then
  echo "Error: Missing required parameters"
  echo "Usage: $0 <project_id> <tpu_name> <zone> <accelerator_type> <runtime_version> [--spot]"
  exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
  echo "Error: gcloud CLI not found. Please install Google Cloud SDK."
  exit 1
fi

# Check if TPU already exists
TPU_STATUS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "NOT_FOUND")

if [[ "$TPU_STATUS" != "NOT_FOUND" && "$TPU_STATUS" != "TERMINATED" ]]; then
  echo "TPU '$TPU_NAME' already exists in state: $TPU_STATUS"
  exit 0
fi

# Create TPU VM
echo "Creating TPU VM '$TPU_NAME' in zone '$ZONE'..."

SPOT_FLAG=""
if [[ "$SPOT" == "--spot" ]]; then
  SPOT_FLAG="--spot"
  echo "Requesting spot TPU VM (lower cost but preemptible)"
fi

# Create TPU VM with queuing
gcloud compute tpus queued-resources create "$TPU_NAME" \
  --node-id="$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --runtime-version="$RUNTIME_VERSION" \
  $SPOT_FLAG \
  --wait

# Wait for TPU to be ready
for i in {1..60}; do
  TPU_STATUS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null || echo "NOT_FOUND")
  if [[ "$TPU_STATUS" == "READY" ]]; then
    echo "TPU VM '$TPU_NAME' is now ready"
    exit 0
  fi
  echo "Waiting for TPU VM to be ready... Current state: $TPU_STATUS"
  sleep 10
done

echo "Error: Timed out waiting for TPU VM to be ready"
exit 1