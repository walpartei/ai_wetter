import os
import json
import uuid
import tempfile
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from app.data_sources.base_source import BaseDataSource
from app.models.location import Location
from app.models.forecast import Forecast, WeatherMetrics
from app.utils.config import Config, APP_DIR
from app.utils.logging import get_logger

logger = get_logger()

class GenCastDataSource(BaseDataSource):
    """Data source for GenCast AI weather forecasts."""
    
    def __init__(self):
        super().__init__("GenCast")
        self.config = Config().get_api_config("gencast")
        self.project_id = self.config.get("project_id")
        self.bucket_name = self.config.get("bucket_name")
        self.zone = self.config.get("zone")
        self.accelerator_type = self.config.get("accelerator_type")
        self.runtime_version = self.config.get("runtime_version")
        self.model_path = self.config.get("model_path")
        self.stats_path = self.config.get("stats_path")
        self.ensemble_samples = self.config.get("ensemble_samples", 1)
        
        # Setup directories
        self.scripts_dir = APP_DIR / "app" / "scripts" / "gencast"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create helper scripts
        self._setup_helper_scripts()
    
    def is_available(self) -> bool:
        """Check if GenCast is properly configured and available."""
        if not self.config.get("enabled", False):
            logger.warning("GenCast is not enabled in configuration")
            return False
            
        if not all([self.project_id, self.bucket_name, self.zone, 
                  self.accelerator_type, self.runtime_version]):
            logger.warning("GenCast cloud configuration is incomplete")
            return False
        
        # Check if gcloud CLI is installed
        try:
            result = subprocess.run(
                ["gcloud", "--version"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode != 0:
                logger.warning("gcloud CLI is not available")
                return False
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.warning("gcloud CLI is not available")
            return False
            
        # Check if authenticated with GCP
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if "No credentialed accounts." in result.stdout:
                logger.warning("Not authenticated with Google Cloud")
                return False
        except subprocess.SubprocessError:
            logger.warning("Failed to check Google Cloud authentication")
            return False
            
        # Check if we can access the bucket
        try:
            result = subprocess.run(
                ["gcloud", "storage", "ls", f"gs://{self.bucket_name}"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode != 0:
                logger.warning(f"Cannot access bucket gs://{self.bucket_name}")
                return False
        except subprocess.SubprocessError:
            logger.warning(f"Failed to check bucket gs://{self.bucket_name}")
            return False
        
        return True
    
    def get_forecast(self, location: Location, days: int = 14) -> List[Forecast]:
        """Get weather forecast for a location using GenCast.
        
        Args:
            location: The location to get forecast for
            days: Number of days to forecast (max 14)
            
        Returns:
            List of Forecast objects for the requested days
        """
        if not self.is_available():
            logger.error("GenCast is not available")
            return []
        
        # Limit days to 14 (our model is trained for this range)
        days = min(days, 14)
        
        try:
            # Generate a unique job ID for this forecast request
            job_id = f"gencast-{str(uuid.uuid4())[:8]}"
            logger.info(f"Starting GenCast forecast job {job_id} for {location.name}")
            
            # Provision TPU VM and run forecast
            success, result_path = self._run_gencast_forecast(
                job_id=job_id,
                location=location,
                days=days
            )
            
            if not success:
                logger.error(f"GenCast forecast job {job_id} failed")
                return []
                
            # Parse results and convert to forecast objects
            forecasts = self._parse_forecast_results(
                result_path=result_path,
                location=location,
                days=days
            )
            
            logger.info(f"GenCast forecast for {location.name} completed successfully with {len(forecasts)} days")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error getting GenCast forecast: {e}", exc_info=True)
            return []
    
    def _setup_helper_scripts(self):
        """Create helper scripts for provisioning TPUs and running forecasts."""
        # Create provision_tpu.sh
        provision_script = self.scripts_dir / "provision_tpu.sh"
        with open(provision_script, "w") as f:
            f.write(f"""#!/bin/bash
# Provision a TPU VM for GenCast inference
TPU_NAME=$1
ZONE="{self.zone}"
PROJECT_ID="{self.project_id}"
ACCELERATOR_TYPE="{self.accelerator_type}"
RUNTIME_VERSION="{self.runtime_version}"

echo "Provisioning TPU VM $TPU_NAME in $ZONE..."
gcloud compute tpus queued-resources create $TPU_NAME \\
  --node-id=$TPU_NAME \\
  --zone=$ZONE \\
  --project=$PROJECT_ID \\
  --accelerator-type=$ACCELERATOR_TYPE \\
  --runtime-version=$RUNTIME_VERSION \\
  --spot
""")

        # Create run_inference.sh
        inference_script = self.scripts_dir / "run_inference.sh"
        with open(inference_script, "w") as f:
            f.write(f"""#!/bin/bash
# Run GenCast inference on a TPU VM
TPU_NAME=$1
JOB_ID=$2
LATITUDE=$3
LONGITUDE=$4
DAYS=$5
SAMPLES=$6
ZONE="{self.zone}"
PROJECT_ID="{self.project_id}"
BUCKET_NAME="{self.bucket_name}"
MODEL_PATH="{self.model_path}"
STATS_PATH="{self.stats_path}"

echo "Running GenCast inference on $TPU_NAME..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \\
  --zone=$ZONE --project=$PROJECT_ID --command="
  set -e
  
  # Install dependencies
  pip install -qq -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install -qq numpy xarray zarr netCDF4 matplotlib jsonpickle
  pip install -qq --upgrade https://github.com/deepmind/graphcast/archive/master.zip

  # Download necessary data
  gcloud storage cp $MODEL_PATH .
  gcloud storage cp --recursive $STATS_PATH ./stats/

  # Create input data from location
  python3 -c '
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os

# Create a simple input for GenCast (simplified, would need to be expanded for real use)
lat, lon = $LATITUDE, $LONGITUDE
days = $DAYS

# Create a grid (very simplified)
lat_range = np.linspace(-90, 90, 721)  # 0.25-degree resolution
lon_range = np.linspace(-180, 180, 1441)  # 0.25-degree resolution

# Create a time array for 2 input times + forecast days
times = [datetime.now() + timedelta(hours=i*12) for i in range(days+2)]

ds = xr.Dataset(
    coords={
        "time": times,
        "lat": lat_range,
        "lon": lon_range,
    }
)

# Initialize with some basic variables (would need more in real implementation)
ds["2m_temperature"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * 20.0,  # 20°C everywhere
    dims=["time", "lat", "lon"]
)

ds["10m_u_component_of_wind"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * 0.0,  # 0 m/s everywhere
    dims=["time", "lat", "lon"]
)

ds["10m_v_component_of_wind"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * 0.0,  # 0 m/s everywhere
    dims=["time", "lat", "lon"]
)

# Save to a NetCDF file
ds.to_netcdf("input_data.nc")
'

  # Create run_gencast.py script
  cat > run_gencast.py << EOL
import jax
import numpy as np
import xarray as xr
import zarr
import os
import json
import time
import dataclasses
import math
import haiku as hk
from datetime import datetime, timedelta
import statistics
import sys

print(f'Number of devices: {jax.device_count()}')
print(f'JAX devices: {jax.devices()}')

# Import GenCast dependencies
# These will be installed via the pip command in the script
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

# Extract metadata from environment
latitude = float(os.environ.get('LATITUDE'))
longitude = float(os.environ.get('LONGITUDE'))
days = int(os.environ.get('DAYS'))
ensemble_samples = int(os.environ.get('SAMPLES', 1))

# Calculate number of timesteps needed (usually in 12-hour increments)
num_steps = days * 2  # Two steps per day with 12-hour intervals

print(f"Running GenCast for location: {latitude}, {longitude}")
print(f"Generating {days} days of forecast with {ensemble_samples} ensemble members")

# Load the GenCast model
print("Loading GenCast model...")
model_path = "GenCast 0p25deg Operational <2019.npz"
try:
    with open(model_path, "rb") as f:
        print(f"Loading checkpoint from {model_path}")
        ckpt = checkpoint.load(f, gencast.CheckPoint)
        print("Checkpoint loaded successfully")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    available_files = os.listdir()
    print(f"Available files in current directory: {available_files}")
    raise

# Extract model configurations
params = ckpt.params
state = {}
task_config = ckpt.task_config
sampler_config = ckpt.sampler_config
noise_config = ckpt.noise_config
noise_encoder_config = ckpt.noise_encoder_config
denoiser_architecture_config = ckpt.denoiser_architecture_config
print("Model description:", ckpt.description)

# Load normalization data
print("Loading normalization data...")
with open("stats/diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()
with open("stats/mean_by_level.nc", "rb") as f:
    mean_by_level = xr.load_dataset(f).compute()
with open("stats/stddev_by_level.nc", "rb") as f:
    stddev_by_level = xr.load_dataset(f).compute()
with open("stats/min_by_level.nc", "rb") as f:
    min_by_level = xr.load_dataset(f).compute()
print("Normalization data loaded")

# Prepare model input data for the location
print("Creating input data...")
# This is a simplified version of input preparation
# In a full implementation, we would use actual weather data from a global source

# Create an example input for GenCast
# This replicates part of the process from the GenCast colab notebook
ds = xr.Dataset()

# Create a grid
# For 0.25 degree resolution
lat_range = np.linspace(-90, 90, 721)
lon_range = np.linspace(-180, 180, 1441)

# Create a time array with current timestamp and 12h before
now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
times = [now - timedelta(hours=12), now]

# Create input coordinates
ds.coords["time"] = times
ds.coords["lat"] = lat_range
ds.coords["lon"] = lon_range

# Create an example input for GenCast containing typical atmospheric variables
# In a real system, this would be actual global weather data

# Initialize global fields with some reasonable values
# Temperature at 2m in K
ds["2m_temperature"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * (15 + 273.15),
    dims=["time", "lat", "lon"]
)

# 10m_u_component_of_wind in m/s
ds["10m_u_component_of_wind"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * 2.0,
    dims=["time", "lat", "lon"]
)

# 10m_v_component_of_wind in m/s
ds["10m_v_component_of_wind"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * 2.0,
    dims=["time", "lat", "lon"]
)

# Mean sea level pressure in Pa
ds["mean_sea_level_pressure"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * 101325.0,
    dims=["time", "lat", "lon"]
)

# Total precipitation in m
ds["total_precipitation"] = xr.DataArray(
    np.ones((len(times), len(lat_range), len(lon_range))) * 0.0,
    dims=["time", "lat", "lon"]
)

# Find nearest grid point to our target
lat_idx = (np.abs(ds.lat.values - latitude)).argmin()
lon_idx = (np.abs(ds.lon.values - longitude)).argmin()
target_lat = float(ds.lat.values[lat_idx])
target_lon = float(ds.lon.values[lon_idx])
print(f"Target location grid point: {target_lat}, {target_lon}")

# Set up the model
print("Setting up GenCast model...")

def construct_wrapped_gencast():
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

# Create targets template
print("Creating targets template...")
# Create a template for the forecast period
forecast_times = [now + timedelta(hours=12*i) for i in range(1, num_steps+1)]
targets_template = ds.copy()
targets_template.coords["time"] = forecast_times
# Fill with NaNs to indicate these are to be predicted
for var in targets_template.data_vars:
    targets_template[var].values[:] = np.nan

# Prepare forcing data if needed (can be empty for GenCast)
forcings = None

# Extract inputs, targets template using the task configuration
print("Extracting inputs and targets...")
try:
    ds_inputs, _, ds_forcings = data_utils.extract_inputs_targets_forcings(
        ds, target_lead_times=slice("12h", f"{num_steps*12}h"),
        **dataclasses.asdict(task_config))
except Exception as e:
    print(f"Error extracting inputs and targets: {e}")
    raise

# Initialize and jit the forward function
print("Initializing model...")
if params is None:
    print("Error: Model parameters not loaded correctly")
    sys.exit(1)

# Jit the forward function
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
# Create a pmapped version for running in parallel
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

# Generate ensemble forecasts
print(f"Generating {ensemble_samples} forecast ensemble members...")
rng = jax.random.PRNGKey(int(time.time()))
# We fold-in the ensemble member, this way the first N members should always
# match across different runs which use take the same inputs
rngs = np.stack(
    [jax.random.fold_in(rng, i) for i in range(ensemble_samples)], axis=0)

# Run the model
print("Running GenCast inference...")
try:
    chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        # Use pmapped version to parallelise across devices
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=ds_inputs,
        targets_template=targets_template * np.nan,
        forcings=ds_forcings,
        num_steps_per_chunk=1,
        num_samples=ensemble_samples,
        pmap_devices=jax.local_devices()
    ):
        print(f"Generated chunk of forecast data: {chunk.dims}")
        chunks.append(chunk)
    
    # Combine the forecast chunks
    predictions = xr.combine_by_coords(chunks)
    print(f"Predictions shape: {predictions.dims}")
except Exception as e:
    print(f"Error during inference: {e}")
    raise

# Process predictions to extract forecast metrics at our target location
print("Processing predictions...")

# Extract data for our target location
target_data = predictions.sel(lat=target_lat, lon=target_lon, method="nearest")

# Function to convert the forecasts to our format
def process_ensemble_member(member_idx):
    # Extract this member's data
    member_data = target_data.isel(sample=member_idx)
    
    forecasts = []
    # Loop through each time step (by pairs to get daily min/max)
    for day in range(days):
        # Get morning and afternoon data for this day
        try:
            morning = member_data.isel(time=day*2)
            afternoon = member_data.isel(time=day*2+1) if day*2+1 < len(member_data.time) else morning
            
            # Get the forecast date (noon of the day)
            forecast_date = now + timedelta(days=day+1)
            
            # Extract metrics
            temp_morning = float(morning["2m_temperature"].values) - 273.15  # Convert K to C
            temp_afternoon = float(afternoon["2m_temperature"].values) - 273.15
            
            temp_min = min(temp_morning, temp_afternoon)
            temp_max = max(temp_morning, temp_afternoon)
            temp_mean = (temp_min + temp_max) / 2
            
            # Wind components to speed and direction
            u_wind = float(afternoon["10m_u_component_of_wind"].values)
            v_wind = float(afternoon["10m_v_component_of_wind"].values)
            
            wind_speed = math.sqrt(u_wind**2 + v_wind**2) * 3.6  # Convert m/s to km/h
            wind_direction = (math.degrees(math.atan2(v_wind, u_wind)) + 180) % 360  # 0 = from North
            
            # Check if we have MSL pressure
            pressure = float(afternoon["mean_sea_level_pressure"].values) / 100 if "mean_sea_level_pressure" in afternoon else 1013.25  # Convert Pa to hPa
            
            # Precipitation - sum up the day's total
            precip_morning = float(morning["total_precipitation"].values) * 1000 if "total_precipitation" in morning else 0  # Convert m to mm
            precip_afternoon = float(afternoon["total_precipitation"].values) * 1000 if "total_precipitation" in afternoon else 0
            precipitation = precip_morning + precip_afternoon
            
            # Calculate confidence based on lead time
            # In a real implementation, this would use ensemble spread
            confidence = max(0.2, min(0.95, 0.95 - 0.05 * day))
            
            # Create forecast dictionary
            forecast = {
                "date": forecast_date.isoformat(),
                "latitude": target_lat,
                "longitude": target_lon,
                "metrics": {
                    "temperature_min": float(temp_min),
                    "temperature_max": float(temp_max),
                    "temperature_mean": float(temp_mean),
                    "precipitation": float(precipitation),
                    "humidity": 70.0,  # Placeholder - GenCast doesn't directly predict this
                    "wind_speed": float(wind_speed),
                    "wind_direction": float(wind_direction),
                    "cloud_cover": 50.0,  # Placeholder - GenCast doesn't directly predict this
                    "soil_temperature": float(temp_mean - 2.0),  # Approximation
                    "soil_moisture": 50.0,  # Placeholder - GenCast doesn't directly predict this
                    "uv_index": None  # GenCast doesn't predict this
                },
                "confidence": confidence
            }
            forecasts.append(forecast)
        except Exception as e:
            print(f"Error processing day {day} for member {member_idx}: {e}")
            continue
    
    return forecasts

# Process each ensemble member
ensemble = []
for i in range(ensemble_samples):
    try:
        member_forecasts = process_ensemble_member(i)
        ensemble.append(member_forecasts)
        print(f"Processed ensemble member {i+1}/{ensemble_samples}")
    except Exception as e:
        print(f"Error processing ensemble member {i}: {e}")
        continue

# Calculate ensemble statistics for confidence
if len(ensemble) > 1:
    print("Calculating ensemble statistics...")
    for day in range(min(len(member_forecasts) for member_forecasts in ensemble)):
        # Extract metrics for this day across all members
        temp_values = [members[day]["metrics"]["temperature_mean"] for members in ensemble]
        precip_values = [members[day]["metrics"]["precipitation"] for members in ensemble]
        wind_values = [members[day]["metrics"]["wind_speed"] for members in ensemble]
        
        # Calculate standard deviation
        temp_std = statistics.stdev(temp_values) if len(temp_values) > 1 else 0
        precip_std = statistics.stdev(precip_values) if len(precip_values) > 1 else 0
        wind_std = statistics.stdev(wind_values) if len(wind_values) > 1 else 0
        
        # Update confidence based on ensemble spread
        temp_confidence = max(0.1, min(0.9, 1 - (temp_std / 10)))  # Normalize, assuming 10°C spread is low confidence
        precip_confidence = max(0.1, min(0.9, 1 - (precip_std / 20)))  # Normalize, 20mm spread is low confidence
        wind_confidence = max(0.1, min(0.9, 1 - (wind_std / 15)))  # Normalize, 15km/h spread is low confidence
        
        # Overall confidence is weighted average
        overall = (temp_confidence * 0.4) + (precip_confidence * 0.4) + (wind_confidence * 0.2)
        
        # Update confidence in the first ensemble member (which will be used as primary)
        if ensemble and ensemble[0] and day < len(ensemble[0]):
            ensemble[0][day]["confidence"] = overall
            # Also store confidence metrics for reference
            ensemble[0][day]["confidence_metrics"] = {
                "temperature": temp_confidence,
                "precipitation": precip_confidence,
                "wind": wind_confidence,
                "overall": overall,
                "temp_std": temp_std,
                "precip_std": precip_std,
                "wind_std": wind_std
            }

# Save forecasts as JSON
print("Saving forecast results...")
with open('gencast_forecast.json', 'w') as f:
    json.dump({
        "metadata": {
            "latitude": target_lat,
            "longitude": target_lon,
            "generated_at": datetime.now().isoformat(),
            "model": "GenCast 0.25deg Operational",
            "ensemble_size": len(ensemble)
        },
        "ensemble": ensemble
    }, f, indent=2)

print("GenCast forecast generation complete")
EOL

  # Run the GenCast inference
  export LATITUDE=$LATITUDE
  export LONGITUDE=$LONGITUDE
  export DAYS=$DAYS
  export SAMPLES=$SAMPLES
  python3 run_gencast.py
  
  # Copy results to Cloud Storage
  echo 'Copying results to Cloud Storage...'
  gcloud storage cp gencast_forecast.json gs://$BUCKET_NAME/$JOB_ID/
"

# Check if command succeeded
if [ $? -eq 0 ]; then
  echo "GenCast inference successful"
  exit 0
else
  echo "GenCast inference failed"
  exit 1
fi
""")

        # Create cleanup_tpu.sh
        cleanup_script = self.scripts_dir / "cleanup_tpu.sh"
        with open(cleanup_script, "w") as f:
            f.write(f"""#!/bin/bash
# Clean up TPU resources
TPU_NAME=$1
ZONE="{self.zone}"
PROJECT_ID="{self.project_id}"

echo "Cleaning up TPU VM $TPU_NAME..."
gcloud compute tpus queued-resources delete $TPU_NAME \\
  --zone=$ZONE \\
  --project=$PROJECT_ID \\
  --quiet
""")

        # Make scripts executable
        for script in [provision_script, inference_script, cleanup_script]:
            os.chmod(script, 0o755)
    
    def _run_gencast_forecast(self, job_id: str, location: Location, days: int) -> Tuple[bool, Optional[str]]:
        """Run GenCast forecast job on Google Cloud TPU.
        
        Args:
            job_id: Unique job identifier
            location: Location for forecast
            days: Number of days to forecast
            
        Returns:
            Tuple of (success, result_path)
        """
        tpu_name = f"tpu-{job_id}"
        
        try:
            # Step 1: Provision TPU VM
            logger.info(f"Provisioning TPU VM {tpu_name}...")
            provision_cmd = [
                str(self.scripts_dir / "provision_tpu.sh"),
                tpu_name
            ]
            
            result = subprocess.run(
                provision_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to provision TPU VM: {result.stderr}")
                return False, None
            
            # Step 2: Wait for TPU VM to be ready (this can take a while)
            logger.info(f"Waiting for TPU VM {tpu_name} to be ready...")
            max_wait_time = 600  # 10 minutes
            wait_interval = 30   # 30 seconds
            
            for _ in range(0, max_wait_time, wait_interval):
                status_cmd = [
                    "gcloud", "compute", "tpus", "queued-resources", "describe",
                    tpu_name, "--zone", self.zone, "--project", self.project_id,
                ]
                
                status_result = subprocess.run(
                    status_cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if "state: RUNNING" in status_result.stdout:
                    logger.info(f"TPU VM {tpu_name} is now running")
                    break
                    
                logger.info(f"TPU VM {tpu_name} not ready yet, waiting...")
                time.sleep(wait_interval)
            else:
                logger.error(f"Timed out waiting for TPU VM {tpu_name} to be ready")
                return False, None
            
            # Step 3: Run GenCast inference
            logger.info(f"Running GenCast inference on TPU VM {tpu_name}...")
            inference_cmd = [
                str(self.scripts_dir / "run_inference.sh"),
                tpu_name,
                job_id,
                str(location.latitude),
                str(location.longitude),
                str(days),
                str(self.ensemble_samples)
            ]
            
            inference_result = subprocess.run(
                inference_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if inference_result.returncode != 0:
                logger.error(f"GenCast inference failed: {inference_result.stderr}")
                return False, None
                
            # Step 4: Download results from Cloud Storage
            logger.info(f"Downloading GenCast forecast results for job {job_id}...")
            result_path = self._download_forecast_results(job_id)
            
            if not result_path:
                logger.error(f"Failed to download GenCast forecast results for job {job_id}")
                return False, None
                
            return True, result_path
            
        finally:
            # Clean up TPU resources
            logger.info(f"Cleaning up TPU VM {tpu_name}...")
            cleanup_cmd = [
                str(self.scripts_dir / "cleanup_tpu.sh"),
                tpu_name
            ]
            
            cleanup_result = subprocess.run(
                cleanup_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if cleanup_result.returncode != 0:
                logger.warning(f"Failed to clean up TPU VM {tpu_name}: {cleanup_result.stderr}")
    
    def _download_forecast_results(self, job_id: str) -> Optional[str]:
        """Download forecast results from Cloud Storage.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Path to downloaded results or None if failed
        """
        # Create temporary directory for results
        temp_dir = tempfile.mkdtemp(prefix="gencast_")
        result_path = os.path.join(temp_dir, "gencast_forecast.json")
        
        try:
            # Download results from Cloud Storage
            download_cmd = [
                "gcloud", "storage", "cp",
                f"gs://{self.bucket_name}/{job_id}/gencast_forecast.json",
                result_path
            ]
            
            result = subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to download results: {result.stderr}")
                return None
                
            return result_path
            
        except Exception as e:
            logger.error(f"Error downloading forecast results: {e}")
            return None
    
    def _parse_forecast_results(self, result_path: str, location: Location, days: int) -> List[Forecast]:
        """Parse forecast results and convert to Forecast objects.
        
        Args:
            result_path: Path to results file
            location: Location for forecast
            days: Number of days in forecast
            
        Returns:
            List of Forecast objects
        """
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        
        try:
            # Load forecast data
            with open(result_path, 'r') as f:
                data = json.load(f)
                
            ensemble = data.get("ensemble", [])
            metadata = data.get("metadata", {})
            
            if not ensemble:
                logger.error("No forecast data found in results")
                return []
                
            # Use the first ensemble member by default, which has confidence metrics from all members
            forecast_data = ensemble[0]
            ensemble_size = metadata.get("ensemble_size", len(ensemble))
            
            # Convert to Forecast objects
            forecasts = []
            
            for day_forecast in forecast_data:
                forecast_date = datetime.fromisoformat(day_forecast["date"])
                
                # Extract metrics
                metrics = WeatherMetrics(
                    temperature_min=day_forecast["metrics"]["temperature_min"],
                    temperature_max=day_forecast["metrics"]["temperature_max"],
                    temperature_mean=day_forecast["metrics"]["temperature_mean"],
                    precipitation=day_forecast["metrics"]["precipitation"],
                    humidity=day_forecast["metrics"]["humidity"],
                    wind_speed=day_forecast["metrics"]["wind_speed"],
                    wind_direction=day_forecast["metrics"]["wind_direction"],
                    cloud_cover=day_forecast["metrics"]["cloud_cover"],
                    soil_temperature=day_forecast["metrics"]["soil_temperature"],
                    soil_moisture=day_forecast["metrics"]["soil_moisture"],
                    uv_index=day_forecast["metrics"].get("uv_index")
                )
                
                # Get confidence metrics
                confidence = day_forecast.get("confidence", 0.5)
                confidence_metrics = day_forecast.get("confidence_metrics", {})
                
                # Create raw data with enhanced confidence metrics
                raw_data = {
                    "source": self.name,
                    "date": forecast_date.isoformat(),
                    "location": {
                        "name": location.name,
                        "lat": location.latitude,
                        "lon": location.longitude
                    },
                    "metrics": day_forecast["metrics"],
                    "confidence": confidence,
                    "ensemble_size": ensemble_size
                }
                
                # Add confidence metrics if available (from ensemble)
                if confidence_metrics:
                    raw_data["confidence_metrics"] = confidence_metrics
                
                # Create Forecast object
                forecast = Forecast(
                    source=self.name,
                    location_id=location_id,
                    date=forecast_date,
                    metrics=metrics,
                    raw_data=raw_data
                )
                
                forecasts.append(forecast)
            
            logger.info(f"Successfully processed {len(forecasts)} days of forecast from GenCast with {ensemble_size} ensemble members")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error parsing forecast results: {e}", exc_info=True)
            return []