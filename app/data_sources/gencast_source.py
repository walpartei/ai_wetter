import random
import math
from datetime import datetime, timedelta
from typing import List

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

        # For now, we'll skip script generation and use a simplified approach

    def is_available(self) -> bool:
        """Check if GenCast is properly configured and available."""
        if not self.config.get("enabled", False):
            logger.warning("GenCast is not enabled in configuration")
            return False

        if not all([self.project_id, self.bucket_name]):
            logger.warning("GenCast cloud configuration is incomplete")
            return False

        # For now, let's assume it's available if enabled in config
        # We'll check actual availability when we try to use it
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
            # For now, use a simulated forecast since we're focusing on
            # integration
            logger.info(
                f"Generating simulated GenCast forecast for {location.name}"
            )

            forecasts = self._generate_simulated_forecast(location, days)

            logger.info(
                f"Generated GenCast forecast with {len(forecasts)} days"
            )
            return forecasts

        except Exception as e:
            logger.error(f"Error getting GenCast forecast: {e}", exc_info=True)
            return []

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