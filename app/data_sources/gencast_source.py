import json
import uuid
import tempfile
import subprocess
import random
import math
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

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
        self.use_tpu = self.config.get("use_tpu", False)
        self.use_spot = self.config.get("use_spot", True)
        self.cache_forecasts = self.config.get("cache_forecasts", True)
        self.cache_duration_hours = self.config.get("cache_duration_hours", 6)

        # Setup directories
        self.scripts_dir = APP_DIR / "app" / "scripts" / "gencast"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure scripts are executable
        for script in self.scripts_dir.glob("*.sh"):
            script.chmod(0o755)
        for script in self.scripts_dir.glob("*.py"):
            script.chmod(0o755)
        
        # Create cache directory
        self.cache_dir = APP_DIR / "data" / "cache" / "gencast"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if GenCast is properly configured and available."""
        if not self.config.get("enabled", False):
            logger.warning("GenCast is not enabled in configuration")
            return False

        if not all([self.project_id, self.bucket_name]):
            logger.warning("GenCast cloud configuration is incomplete")
            return False
            
        # Check if required scripts exist
        required_scripts = [
            "provision_tpu.sh",
            "run_inference.sh",
            "cleanup_tpu.sh",
            "process_results.py"
        ]
        
        for script in required_scripts:
            if not (self.scripts_dir / script).exists():
                logger.warning(f"Required script {script} not found")
                return False
                
        # If not using TPU, we can still provide simulated forecasts
        if not self.use_tpu:
            logger.info("GenCast TPU usage is disabled, will use simulation")
            return True
                
        # Check if gcloud is installed when using TPU
        try:
            subprocess.run(["gcloud", "--version"], 
                          check=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Google Cloud SDK (gcloud) not found but required for GenCast TPU usage")
            return False
            
        return True

    def get_forecast(
        self, location: Location, days: int = 14
    ) -> List[Forecast]:
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
            # Check cache first if enabled
            if self.cache_forecasts:
                cached_forecasts = self._get_cached_forecast(location, days)
                if cached_forecasts:
                    logger.info(f"Using cached GenCast forecast for {location.name}")
                    return cached_forecasts
            
            # If TPU usage is enabled and properly configured, use it
            if self.use_tpu and self._check_cloud_credentials():
                logger.info(f"Generating TPU-based GenCast forecast for {location.name}")
                forecasts = self._generate_tpu_forecast(location, days)
                
                # Cache the results if successful
                if forecasts and self.cache_forecasts:
                    self._cache_forecast(location, forecasts)
            else:
                # Fall back to simulation
                logger.info(f"Generating simulated GenCast forecast for {location.name}")
                forecasts = self._generate_simulated_forecast(location, days)

            logger.info(f"Generated GenCast forecast with {len(forecasts)} days")
            return forecasts

        except Exception as e:
            logger.error(f"Error getting GenCast forecast: {e}", exc_info=True)
            # Fall back to simulation if TPU fails
            if self.use_tpu:
                logger.info("Falling back to simulated forecast due to TPU error")
                try:
                    return self._generate_simulated_forecast(location, days)
                except Exception as sim_error:
                    logger.error(f"Error generating fallback simulation: {sim_error}")
            return []
            
    def _check_cloud_credentials(self) -> bool:
        """Check if Google Cloud credentials are properly configured."""
        try:
            # Check if gcloud is authenticated
            result = subprocess.run(
                ["gcloud", "auth", "list", "--format=json"], 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Parse JSON output to check if any accounts are active
            accounts = json.loads(result.stdout)
            if not accounts:
                logger.warning("No authenticated Google Cloud accounts found")
                return False
                
            # Check if we can access the project
            result = subprocess.run(
                ["gcloud", "projects", "describe", self.project_id, "--format=json"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return True
            
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to verify Google Cloud credentials: {e}")
            return False
            
    def _generate_tpu_forecast(self, location: Location, days: int) -> List[Forecast]:
        """Generate forecast using Google Cloud TPU."""
        # Create a unique identifier for this forecast request
        forecast_id = f"gencast_{location.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        tpu_name = f"gencast-{uuid.uuid4().hex[:8]}"
        
        try:
            # Create temporary directory for outputs
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                output_path = temp_dir_path / f"{forecast_id}.json"
                
                # Step 1: Provision TPU
                logger.info(f"Provisioning TPU VM {tpu_name} for GenCast inference")
                spot_flag = "--spot" if self.use_spot else ""
                provision_cmd = [
                    str(self.scripts_dir / "provision_tpu.sh"),
                    self.project_id,
                    tpu_name,
                    self.zone,
                    self.accelerator_type,
                    self.runtime_version,
                    spot_flag
                ]
                
                subprocess.run(provision_cmd, check=True, text=True)
                
                # Step 2: Run inference
                logger.info(f"Running GenCast inference on TPU VM {tpu_name}")
                inference_cmd = [
                    str(self.scripts_dir / "run_inference.sh"),
                    self.project_id,
                    tpu_name,
                    self.zone,
                    self.bucket_name,
                    self.model_path,
                    self.stats_path,
                    str(location.latitude),
                    str(location.longitude),
                    str(self.ensemble_samples),
                    str(days),
                    forecast_id
                ]
                
                subprocess.run(inference_cmd, check=True, text=True)
                
                # Step 3: Process results
                logger.info(f"Processing GenCast inference results for {location.name}")
                location_json = json.dumps({
                    "id": location.id,
                    "name": location.name,
                    "latitude": location.latitude,
                    "longitude": location.longitude
                })
                
                process_cmd = [
                    str(self.scripts_dir / "process_results.py"),
                    "--bucket", self.bucket_name,
                    "--output-id", forecast_id,
                    "--location", location_json,
                    "--output-file", str(output_path)
                ]
                
                subprocess.run(process_cmd, check=True, text=True)
                
                # Step 4: Cleanup TPU
                logger.info(f"Cleaning up TPU VM {tpu_name}")
                cleanup_cmd = [
                    str(self.scripts_dir / "cleanup_tpu.sh"),
                    self.project_id,
                    tpu_name,
                    self.zone
                ]
                
                # Run cleanup in the background to avoid waiting
                subprocess.Popen(cleanup_cmd)
                
                # Step 5: Parse results
                if output_path.exists():
                    with open(output_path, "r") as f:
                        forecast_data = json.load(f)
                        
                    # Convert to Forecast objects
                    forecasts = []
                    for f_data in forecast_data:
                        metrics = WeatherMetrics(
                            temperature_min=f_data["metrics"]["temperature_min"],
                            temperature_max=f_data["metrics"]["temperature_max"],
                            temperature_mean=f_data["metrics"]["temperature_mean"],
                            precipitation=f_data["metrics"]["precipitation"],
                            humidity=f_data["metrics"]["humidity"],
                            wind_speed=f_data["metrics"]["wind_speed"],
                            wind_direction=f_data["metrics"]["wind_direction"],
                            cloud_cover=f_data["metrics"]["cloud_cover"],
                            soil_temperature=f_data["metrics"]["soil_temperature"],
                            soil_moisture=f_data["metrics"]["soil_moisture"],
                            uv_index=None
                        )
                        
                        forecast = Forecast(
                            source=self.name,
                            location_id=f_data["location_id"],
                            date=datetime.fromisoformat(f_data["date"]),
                            metrics=metrics,
                            raw_data=f_data["raw_data"],
                            retrieved_at=datetime.fromisoformat(f_data["retrieved_at"])
                        )
                        
                        forecasts.append(forecast)
                        
                    return forecasts
                else:
                    logger.error(f"Output file not found: {output_path}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in TPU forecast generation: {e}", exc_info=True)
            # Ensure TPU is cleaned up even if there's an error
            try:
                cleanup_cmd = [
                    str(self.scripts_dir / "cleanup_tpu.sh"),
                    self.project_id,
                    tpu_name,
                    self.zone
                ]
                subprocess.run(cleanup_cmd, check=True)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up TPU: {cleanup_error}")
                
            # Re-raise to let caller handle
            raise
        
    def _get_cached_forecast(self, location: Location, days: int) -> Optional[List[Forecast]]:
        """Get forecast from cache if available and not expired."""
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        cache_file = self.cache_dir / f"{location_id}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                
            # Check if cache is expired
            cached_at = datetime.fromisoformat(cache_data["cached_at"])
            cache_age = datetime.now() - cached_at
            
            if cache_age.total_seconds() > (self.cache_duration_hours * 3600):
                logger.info(f"Cache expired for {location.name}")
                return None
                
            # Check if we have enough days in the cache
            forecasts = []
            for f_data in cache_data["forecasts"]:
                forecast = Forecast.from_dict(f_data)
                forecasts.append(forecast)
                
            # Sort forecasts by date and limit to requested days
            forecasts.sort(key=lambda f: f.date)
            return forecasts[:days]
            
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None
            
    def _cache_forecast(self, location: Location, forecasts: List[Forecast]) -> None:
        """Cache forecast for future use."""
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        cache_file = self.cache_dir / f"{location_id}.json"
        
        try:
            cache_data = {
                "location_id": location_id,
                "cached_at": datetime.now().isoformat(),
                "forecasts": [f.to_dict() for f in forecasts]
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"Cached forecast for {location.name}")
            
        except Exception as e:
            logger.warning(f"Error writing cache: {e}")

    def _generate_simulated_forecast(
        self, location: Location, days: int
    ) -> List[Forecast]:
        """Generate a simulated forecast for testing.

        This is a placeholder for the actual TPU-based forecast.
        """
        location_id = (
            location.id or str(location.name).lower().replace(" ", "_")
        )
        forecasts = []
        start_date = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Base values for the location
        base_temp_min = 15.0
        base_temp_max = 25.0
        base_precipitation = 2.0
        base_humidity = 70.0
        base_wind_speed = 10.0
        base_wind_direction = 180.0
        base_cloud_cover = 30.0

        # For ensemble effect, generate multiple forecasts
        # Limit to 8 for simplicity
        ensemble_size = min(self.ensemble_samples, 8)
        ensemble_forecasts = []

        for ensemble_idx in range(ensemble_size):
            member_forecasts = []

            # Different random seed for each ensemble member
            random.seed(42 + ensemble_idx)

            # Generate forecasts for each day
            for day in range(days):
                forecast_date = start_date + timedelta(days=day)

                # Add some randomness to simulate weather changes with temporal
                # correlation
                temp_trend = random.uniform(-3.0, 3.0) * math.sqrt(day + 1) / 2
                precip_trend = (
                    max(0, random.uniform(-1.0, 3.0) * math.sqrt(day + 1) / 2)
                )

                # Daily variations
                temp_variation = random.uniform(-2.0, 2.0)
                precip_variation = (
                    random.uniform(0, 1.0) if random.random() > 0.7 else 0
                )

                # Calculate metrics
                temp_min = max(0, base_temp_min + temp_trend + 
                                temp_variation - 5)
                temp_max = max(
                    temp_min + 3, base_temp_max + temp_trend + temp_variation
                )
                temp_mean = (temp_min + temp_max) / 2
                precipitation = max(
                    0, base_precipitation + precip_trend + precip_variation
                )
                humidity = max(
                    0, min(100, base_humidity + random.uniform(-10, 10))
                )
                wind_speed = max(0, base_wind_speed + random.uniform(-5, 5))
                wind_direction = (
                    base_wind_direction + random.uniform(-45, 45)
                ) % 360
                cloud_cover = max(
                    0, min(100, base_cloud_cover + random.uniform(-15, 15))
                )

                # Create forecast object with GenCast as source
                member_forecasts.append(
                    {
                        "date": forecast_date.isoformat(),
                        "metrics": {
                            "temperature_min": temp_min,
                            "temperature_max": temp_max,
                            "temperature_mean": temp_mean,
                            "precipitation": precipitation,
                            "humidity": humidity,
                            "wind_speed": wind_speed,
                            "wind_direction": wind_direction,
                            "cloud_cover": cloud_cover,
                            "soil_temperature": temp_mean - 2,
                            "soil_moisture": max(0, min(100, humidity - 10)),
                        },
                        "confidence": max(0.2, min(0.9, 0.9 - 0.05 * day)),
                    }
                )

            ensemble_forecasts.append(member_forecasts)

        # Calculate ensemble statistics and update confidence
        if len(ensemble_forecasts) > 1:
            for day in range(days):
                # Extract metrics for this day across all members
                temp_values = [
                    member[day]["metrics"]["temperature_mean"]
                    for member in ensemble_forecasts
                ]
                precip_values = [
                    member[day]["metrics"]["precipitation"]
                    for member in ensemble_forecasts
                ]
                wind_values = [
                    member[day]["metrics"]["wind_speed"]
                    for member in ensemble_forecasts
                ]

                # Calculate standard deviation
                temp_std = self._stddev(temp_values)
                precip_std = self._stddev(precip_values)
                wind_std = self._stddev(wind_values)

                # Calculate confidence based on ensemble spread
                temp_confidence = max(0.1, min(0.9, 1 - (temp_std / 10)))
                precip_confidence = max(0.1, min(0.9, 1 - (precip_std / 20)))
                wind_confidence = max(0.1, min(0.9, 1 - (wind_std / 15)))

                # Overall confidence
                overall = (
                    (temp_confidence * 0.4)
                    + (precip_confidence * 0.4)
                    + (wind_confidence * 0.2)
                )

                # Update confidence in first member's forecast
                ensemble_forecasts[0][day]["confidence"] = overall
                # Add confidence metrics
                ensemble_forecasts[0][day]["confidence_metrics"] = {
                    "temperature": temp_confidence,
                    "precipitation": precip_confidence,
                    "wind": wind_confidence,
                    "overall": overall,
                    "ensemble_size": len(ensemble_forecasts),
                }

        # Use first ensemble member (which has updated confidence metrics)
        member_data = ensemble_forecasts[0]

        # Convert to Forecast objects
        for day_forecast in member_data:
            forecast_date = datetime.fromisoformat(day_forecast["date"])

            # Create metrics object
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
                uv_index=None,
            )

            # Get confidence metrics
            confidence = day_forecast.get("confidence", 0.5)
            confidence_metrics = day_forecast.get("confidence_metrics", {})

            # Create the forecast object
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
                        "lon": location.longitude,
                    },
                    "metrics": day_forecast["metrics"],
                    "confidence": confidence,
                    "confidence_metrics": confidence_metrics,
                    "ensemble_size": ensemble_size,
                },
            )

            forecasts.append(forecast)

        return forecasts

    def _stddev(self, values):
        """Calculate standard deviation."""
        if not values or len(values) <= 1:
            return 0
        mean = sum(values) / len(values)
        return (
            sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        ) ** 0.5
