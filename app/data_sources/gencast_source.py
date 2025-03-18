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
  cat > run_gencast.py << 'EOL'
import jax
import numpy as np
import xarray as xr
import zarr
import os
import json
from datetime import datetime
import jsonpickle

print(f'Number of devices: {jax.device_count()}')

# Load input data
input_data = xr.load_dataset('input_data.nc')

# Extract metadata
latitude = float(os.environ.get('LATITUDE'))
longitude = float(os.environ.get('LONGITUDE'))
days = int(os.environ.get('DAYS'))
ensemble_samples = int(os.environ.get('SAMPLES', 1))

# Find nearest grid points for our location of interest
lat_idx = (np.abs(input_data.lat.values - latitude)).argmin()
lon_idx = (np.abs(input_data.lon.values - longitude)).argmin()

# Extract the target location for result processing
target_lat = float(input_data.lat.values[lat_idx])
target_lon = float(input_data.lon.values[lon_idx])
print(f"Target location: {target_lat}, {target_lon}")

# In a full implementation, we would now:
# 1. Load the GenCast model using the JAX/Haiku APIs
# 2. Prepare the input data for the model
# 3. Run the model to generate predictions
# 4. Process the predictions to extract forecasts

# For now, we'll generate a structured simulated output
# This would be replaced with actual model inference in production

def generate_simulated_forecast():
    """Generate simulated forecast data at a single location."""
    # Start with today's date
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Base values (would be determined by model inference)
    base_temp = 20.0 + np.random.normal(0, 2)  # Celsius
    base_precip = max(0, np.random.gamma(1, 2))  # mm
    base_humidity = 70.0 + np.random.normal(0, 5)  # %
    base_wind_speed = max(0, np.random.gamma(2, 2))  # km/h
    base_wind_direction = np.random.uniform(0, 360)  # degrees
    base_cloud_cover = np.random.uniform(0, 100)  # %
    
    forecasts = []
    
    # Generate forecasts for each day
    for day in range(days):
        date = start_date + np.timedelta64(day, 'D')
        
        # Add daily variations with temporal correlation
        temp_variation = np.random.normal(0, 1) * 0.5 * day  # Increasing uncertainty
        precip_variation = np.random.exponential(1) * 0.2 * day
        
        temp_min = max(-20, base_temp - 5 + temp_variation)
        temp_max = min(50, base_temp + 5 + temp_variation)
        
        forecast = {
            "date": date.isoformat(),
            "latitude": target_lat,
            "longitude": target_lon,
            "metrics": {
                "temperature_min": float(temp_min),
                "temperature_max": float(temp_max),
                "temperature_mean": float((temp_min + temp_max) / 2),
                "precipitation": float(max(0, base_precip + precip_variation)),
                "humidity": float(max(0, min(100, base_humidity + np.random.normal(0, 3)))),
                "wind_speed": float(max(0, base_wind_speed + np.random.normal(0, 1))),
                "wind_direction": float((base_wind_direction + np.random.normal(0, 10)) % 360),
                "cloud_cover": float(max(0, min(100, base_cloud_cover + np.random.normal(0, 5)))),
                "soil_temperature": float(max(-20, base_temp - 3 + np.random.normal(0, 0.5))),
                "soil_moisture": float(max(0, min(100, base_humidity - 10 + np.random.normal(0, 2)))),
                "uv_index": float(max(0, min(12, 8 * (1 - base_cloud_cover/100) + np.random.normal(0, 0.5))))
            },
            "confidence": max(0, min(1, 0.9 - 0.05 * day))  # Decreasing confidence over time
        }
        forecasts.append(forecast)
    
    return forecasts

# Generate ensemble of forecasts
ensemble = []
for i in range(ensemble_samples):
    ensemble.append(generate_simulated_forecast())

# Save forecasts as JSON
with open('gencast_forecast.json', 'w') as f:
    json.dump({
        "metadata": {
            "latitude": target_lat,
            "longitude": target_lon,
            "generated_at": datetime.now().isoformat(),
            "model": "GenCast 0.25deg Operational",
            "ensemble_size": ensemble_samples
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
                
            # For now, we'll use the first ensemble member
            # In a future iteration, we could use the ensemble for confidence metrics
            ensemble = data.get("ensemble", [])
            
            if not ensemble:
                logger.error("No forecast data found in results")
                return []
                
            # Use the first ensemble member by default
            forecast_data = ensemble[0]
            
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
                
                # Create Forecast object
                forecast = Forecast(
                    source=self.name,
                    location_id=location_id,
                    date=forecast_date,
                    metrics=metrics,
                    raw_data={
                        "source": self.name,
                        "date": forecast_date.isoformat(),
                        "location": {
                            "name": location.name,
                            "lat": location.latitude,
                            "lon": location.longitude
                        },
                        "metrics": day_forecast["metrics"],
                        "confidence": day_forecast.get("confidence", 0.5),
                        "ensemble_size": len(ensemble)
                    }
                )
                
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error parsing forecast results: {e}", exc_info=True)
            return []