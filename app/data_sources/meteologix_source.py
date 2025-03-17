import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from app.data_sources.base_source import BaseDataSource
from app.models.location import Location
from app.models.forecast import Forecast, WeatherMetrics
from app.utils.config import Config
from app.utils.logging import get_logger

logger = get_logger()


class MeteologixDataSource(BaseDataSource):
    """Data source for Meteologix weather forecasts."""
    
    def __init__(self):
        super().__init__("Meteologix")
        self.config = Config().get_api_config("meteologix")
        self.enabled = self.config.get("enabled", False)
    
    def is_available(self) -> bool:
        """Check if Meteologix API is properly configured and available."""
        # Since this is a placeholder for future implementation,
        # we'll just check if it's enabled in the config
        return self.enabled
    
    def get_forecast(self, location: Location, days: int = 14) -> List[Forecast]:
        """Get weather forecast for a location using Meteologix data.
        
        Args:
            location: The location to get forecast for
            days: Number of days to forecast (max 14)
            
        Returns:
            List of Forecast objects for the requested days
        """
        if not self.is_available():
            logger.info("Meteologix data source is not enabled")
            return []
        
        # Limit days to 14
        days = min(days, 14)
        
        try:
            # In a real implementation, we would fetch data from Meteologix
            # However, since this is marked as "skip for now" in the requirements,
            # we'll just return a simulated forecast for testing purposes
            forecasts = self._get_simulated_forecast(location, days)
            return forecasts
        except Exception as e:
            logger.error(f"Error getting Meteologix forecast: {e}")
            return []
    
    def _get_simulated_forecast(self, location: Location, days: int) -> List[Forecast]:
        """Generate simulated forecast data for testing.
        
        In a real implementation, this would be replaced with actual API calls.
        """
        forecasts = []
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Base values for the location (would be determined by actual data)
        # Different values to simulate another source
        base_temp_min = 13.5  # Celsius
        base_temp_max = 24.5  # Celsius
        base_precipitation = 2.2  # mm
        base_humidity = 68.0  # %
        base_wind_speed = 11.0  # km/h
        base_wind_direction = 190.0  # degrees
        base_cloud_cover = 35.0  # %
        
        # Generate forecasts for each day
        for day in range(days):
            forecast_date = start_date + timedelta(days=day)
            
            # Add some randomness to simulate weather changes
            import random
            temp_variation = random.uniform(-3.0, 3.0)
            precip_variation = random.uniform(-1.2, 2.2)
            humid_variation = random.uniform(-10.0, 10.0)
            wind_variation = random.uniform(-5.0, 5.0)
            wind_dir_variation = random.uniform(-45.0, 45.0)
            cloud_variation = random.uniform(-15.0, 15.0)
            
            # Create metrics for this forecast
            temp_min = max(0, base_temp_min + temp_variation - 4.5)
            temp_max = max(temp_min + 4.5, base_temp_max + temp_variation)
            temp_mean = (temp_min + temp_max) / 2
            
            metrics = WeatherMetrics(
                temperature_min=temp_min,
                temperature_max=temp_max,
                temperature_mean=temp_mean,
                precipitation=max(0, base_precipitation + precip_variation),
                humidity=max(0, min(100, base_humidity + humid_variation)),
                wind_speed=max(0, base_wind_speed + wind_variation),
                wind_direction=(base_wind_direction + wind_dir_variation) % 360,
                cloud_cover=max(0, min(100, base_cloud_cover + cloud_variation)),
                soil_temperature=temp_mean - 3,  # Simulated soil temperature
                soil_moisture=max(0, min(100, base_humidity + humid_variation - 12))
            )
            
            # Create the forecast object
            forecast = Forecast(
                source=self.name,
                location_id=location.id or str(location.name).lower().replace(" ", "_"),
                date=forecast_date,
                metrics=metrics,
                raw_data={  # Simulated raw data
                    "source": "Meteologix",
                    "date": forecast_date.isoformat(),
                    "location": {
                        "name": location.name,
                        "lat": location.latitude,
                        "lon": location.longitude
                    },
                    "temperature": {
                        "min": temp_min,
                        "max": temp_max,
                        "mean": temp_mean
                    },
                    "precipitation": max(0, base_precipitation + precip_variation),
                    "humidity": max(0, min(100, base_humidity + humid_variation)),
                    "wind": {
                        "speed": max(0, base_wind_speed + wind_variation),
                        "direction": (base_wind_direction + wind_dir_variation) % 360
                    },
                    "cloud_cover": max(0, min(100, base_cloud_cover + cloud_variation))
                }
            )
            
            forecasts.append(forecast)
        
        return forecasts
