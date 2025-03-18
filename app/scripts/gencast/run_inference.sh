#!/bin/bash
# Script for running GenCast inference on a TPU VM

set -e

# Get parameters from command line
PROJECT_ID="$1"
TPU_NAME="$2"
ZONE="$3"
BUCKET_NAME="$4"
MODEL_PATH="$5"
STATS_PATH="$6"
LOCATION_LAT="$7"
LOCATION_LON="$8"
ENSEMBLE_SIZE="$9"
PREDICTION_DAYS="${10}"
OUTPUT_ID="${11}"

if [[ -z "$PROJECT_ID" || -z "$TPU_NAME" || -z "$ZONE" || -z "$BUCKET_NAME" || -z "$MODEL_PATH" || -z "$STATS_PATH" || -z "$LOCATION_LAT" || -z "$LOCATION_LON" || -z "$ENSEMBLE_SIZE" || -z "$PREDICTION_DAYS" || -z "$OUTPUT_ID" ]]; then
  echo "Error: Missing required parameters"
  echo "Usage: $0 <project_id> <tpu_name> <zone> <bucket_name> <model_path> <stats_path> <location_lat> <location_lon> <ensemble_size> <prediction_days> <output_id>"
  exit 1
fi

# Create a temporary directory for the inference script
TMP_DIR=$(mktemp -d)
SCRIPT_PATH="$TMP_DIR/inference.py"

# Create Python script for inference
cat > "$SCRIPT_PATH" << 'EOL'
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import time
import datetime

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run GenCast inference on TPU")
parser.add_argument("--bucket", required=True, help="GCS bucket for storage")
parser.add_argument("--model", required=True, help="Path to model")
parser.add_argument("--stats", required=True, help="Path to statistics directory")
parser.add_argument("--lat", required=True, type=float, help="Latitude")
parser.add_argument("--lon", required=True, type=float, help="Longitude")
parser.add_argument("--ensemble", required=True, type=int, help="Ensemble size")
parser.add_argument("--days", required=True, type=int, help="Prediction days")
parser.add_argument("--output", required=True, help="Output ID")
args = parser.parse_args()

# Print configuration for logging
print(f"\nStarting GenCast inference with:")
print(f"Bucket: {args.bucket}")
print(f"Model: {args.model}")
print(f"Stats: {args.stats}")
print(f"Location: {args.lat}, {args.lon}")
print(f"Ensemble size: {args.ensemble}")
print(f"Prediction days: {args.days}")
print(f"Output ID: {args.output}\n")

# Install required packages
sys.stdout.write("Installing required packages...\n")
sys.stdout.flush()
subprocess.run(["pip", "install", "-U", "jax[tpu]", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"], check=True)
subprocess.run(["pip", "install", "-U", "xarray", "zarr", "numpy", "matplotlib", "haiku", "optax"], check=True)
subprocess.run(["pip", "install", "-U", "https://github.com/deepmind/graphcast/archive/master.zip"], check=True)

# Import required libraries
import dataclasses
from datetime import datetime, timedelta
import math
from typing import Optional
from pathlib import Path
import haiku as hk
import jax
import numpy as np
import xarray

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

# Download model and stats files
os.makedirs("model", exist_ok=True)
os.makedirs("stats", exist_ok=True)

print("Downloading model and stats...")
subprocess.run(["gsutil", "cp", args.model, "model/"], check=True)
subprocess.run(["gsutil", "cp", "-r", f"{args.stats}/*.nc", "stats/"], check=True)

model_file = Path("model").glob("*.npz").__next__()

# Load model
print(f"Loading model from {model_file}...")
with open(model_file, "rb") as f:
    ckpt = checkpoint.load(f, gencast.CheckPoint)

params = ckpt.params
state = {}

task_config = ckpt.task_config
sampler_config = ckpt.sampler_config
noise_config = ckpt.noise_config
noise_encoder_config = ckpt.noise_encoder_config
denoiser_architecture_config = ckpt.denoiser_architecture_config

# Create synthetic input data based on location
print(f"Creating input data for location {args.lat}, {args.lon}...")

def create_synthetic_input(lat, lon, days=14):
    """Create synthetic input data for GenCast."""
    # This is a placeholder - in production, we would use actual ERA5 or HRES data
    # For now, we'll create a minimal synthetic input
    
    # Create 2D lat/lon grid for the global domain
    lat_step = 0.25 if "0p25deg" in model_file.name else 1.0
    lon_step = 0.25 if "0p25deg" in model_file.name else 1.0
    
    lats = np.arange(-90, 90+lat_step, lat_step)
    lons = np.arange(-180, 180, lon_step)
    
    # Create time coordinates (3 time steps: 2 for input, 1+ for predictions)
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    times = [start_date + timedelta(hours=12*i) for i in range(days+2)]
    time_coords = np.array([t.timestamp() * 1_000_000 for t in times], dtype=np.int64)
    
    # Create a dataset with essential variables
    ds = xarray.Dataset(
        coords={
            "time": ("time", time_coords),
            "latitude": ("latitude", lats),
            "longitude": ("longitude", lons),
        }
    )
    
    # Add pressure levels
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    ds = ds.assign_coords(level=("level", levels))
    
    # Add required variables (just filling with climatology values for now)
    required_vars = [
        "2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
        "10m_v_component_of_wind", "total_precipitation_12hr", "2m_dewpoint_temperature",
        "geopotential", "u_component_of_wind", "v_component_of_wind", "temperature",
        "specific_humidity", "sea_surface_temperature"
    ]
    
    for var in required_vars:
        if var in ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity"]:
            # 3D variables (include level dimension)
            ds[var] = (("time", "level", "latitude", "longitude"), 
                       np.zeros((len(times), len(levels), len(lats), len(lons))))
        else:
            # 2D variables
            ds[var] = (("time", "latitude", "longitude"), 
                      np.zeros((len(times), len(lats), len(lons))))
    
    return ds

# Create input data
example_batch = create_synthetic_input(args.lat, args.lon, args.days)

# Load normalization data
print("Loading normalization data...")
with open("stats/diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open("stats/mean_by_level.nc", "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open("stats/stddev_by_level.nc", "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()
with open("stats/min_by_level.nc", "rb") as f:
    min_by_level = xarray.load_dataset(f).compute()

# Extract inputs, targets, and forcings
print("Extracting inputs, targets, and forcings...")
eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", f"{args.days*12}h"),
    **dataclasses.asdict(task_config))

# Define the model
def construct_wrapped_gencast():
    """Constructs and wraps the GenCast Predictor."""
    predictor = gencast.GenCast(
        sampler_config=sampler_config,
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
    )

    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=min_by_level,
        var_to_clean='sea_surface_temperature',
    )

    return predictor

@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
    predictor = construct_wrapped_gencast()
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

# Initialize model if params is None
if params is None:
    init_jitted = jax.jit(run_forward.init)
    params, state = init_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )

# We also produce a pmapped version for running in parallel
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

# Run the model
print(f"Running GenCast with {args.ensemble} ensemble members...")
print(f"Number of local devices: {len(jax.local_devices())}")

# Create random keys for ensemble members
rng = jax.random.PRNGKey(42)
rngs = np.stack(
    [jax.random.fold_in(rng, i) for i in range(args.ensemble)], axis=0)

# Run inference with chunking for memory efficiency
print("Starting inference...")
chunks = []
for chunk in rollout.chunked_prediction_generator_multiple_runs(
    predictor_fn=run_forward_pmap,
    rngs=rngs,
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
    num_steps_per_chunk=1,
    num_samples=args.ensemble,
    pmap_devices=jax.local_devices()
    ):
    chunks.append(chunk)
    print(f"Processed chunk {len(chunks)}")

print("Combining predictions...")
predictions = xarray.combine_by_coords(chunks)

# Save predictions to output
print(f"Saving predictions to gs://{args.bucket}/{args.output}.zarr")
predictions.to_zarr(f"{args.output}.zarr")
subprocess.run(["gsutil", "cp", "-r", f"{args.output}.zarr", f"gs://{args.bucket}/"], check=True)

print("Inference completed successfully")
EOL

# Transfer script to TPU and run
echo "Transferring and running inference script on TPU VM '$TPU_NAME'..."

# Copy script to TPU VM
gcloud compute tpus tpu-vm scp "$SCRIPT_PATH" "$TPU_NAME":/tmp/inference.py --zone="$ZONE" --project="$PROJECT_ID" --quiet

# Run inference on TPU VM
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="\
  python /tmp/inference.py \
  --bucket='$BUCKET_NAME' \
  --model='$MODEL_PATH' \
  --stats='$STATS_PATH' \
  --lat='$LOCATION_LAT' \
  --lon='$LOCATION_LON' \
  --ensemble='$ENSEMBLE_SIZE' \
  --days='$PREDICTION_DAYS' \
  --output='$OUTPUT_ID' \
" --quiet

# Fetch the results
echo "Inference completed. Results saved to gs://$BUCKET_NAME/$OUTPUT_ID.zarr"

# Cleanup temp directory
rm -rf "$TMP_DIR"

exit 0