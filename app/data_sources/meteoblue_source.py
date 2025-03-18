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


class MeteoblueDataSource(BaseDataSource):
    """Data source for Meteoblue weather forecasts."""
    
    def __init__(self):
        super().__init__("Meteoblue")
        self.config = Config().get_api_config("meteoblue")
        self.api_key = self.config.get("key")
        self.api_url = "https://my.meteoblue.com/packages/basic-1h"  # Basic package endpoint
    
    def is_available(self) -> bool:
        """Check if Meteoblue API is properly configured and available."""
        # First check if enabled in config
        if not self.config.get("enabled", False):
            logger.warning("Meteoblue API is disabled in configuration")
            return False
            
        if not self.api_key:
            logger.warning("Meteoblue API key not configured")
            return False
            
        # Could add a ping to the API here to check availability
        return True
    
    def get_forecast(self, location: Location, days: int = 14) -> List[Forecast]:
        """Get weather forecast for a location using Meteoblue API.
        
        Args:
            location: The location to get forecast for
            days: Number of days to forecast (max 14)
            
        Returns:
            List of Forecast objects for the requested days
        """
        if not self.is_available():
            logger.error("Meteoblue API not available")
            return []
        
        # Limit days to 14 (or whatever Meteoblue's limit is)
        days = min(days, 14)
        
        try:
            # In a real implementation, we would make a request to the Meteoblue API
            # For example:
            # params = {
            #     "apikey": self.api_key,
            #     "lat": location.latitude,
            #     "lon": location.longitude,
            #     "format": "json",
            #     "temperature": "C",
            #     "windspeed": "kmh",
            #     "precipitationamount": "mm",
            #     "timeformat": "iso8601",
            #     "timeresolution": "daily",
            #     "days": days
            # }
            # response = requests.get(self.api_url, params=params)
            # data = response.json()
            
            # For now, we'll simulate a response
            forecasts = self._get_simulated_forecast(location, days)
            return forecasts
        except Exception as e:
            logger.error(f"Error getting Meteoblue forecast: {e}")
            return []
    
    def _get_simulated_forecast(self, location: Location, days: int) -> List[Forecast]:
        """Generate simulated forecast data for testing.
        
        In a real implementation, this would be replaced with actual API calls.
        """
        forecasts = []
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Base values for the location (would be determined by actual API response)
        # Slightly different from ECMWF to simulate differences between sources
        base_temp_min = 14.0  # Celsius
        base_temp_max = 26.0  # Celsius
        base_precipitation = 1.5  # mm
        base_humidity = 65.0  # %
        base_wind_speed = 12.0  # km/h
        base_wind_direction = 200.0  # degrees
        base_cloud_cover = 25.0  # %
        
        # Generate forecasts for each day
        for day in range(days):
            forecast_date = start_date + timedelta(days=day)
            
            # Add some randomness to simulate weather changes
            import random
            temp_variation = random.uniform(-2.5, 2.5)
            precip_variation = random.uniform(-0.8, 1.5)
            humid_variation = random.uniform(-8.0, 8.0)
            wind_variation = random.uniform(-4.0, 4.0)
            wind_dir_variation = random.uniform(-35.0, 35.0)
            cloud_variation = random.uniform(-12.0, 12.0)
            
            # Create metrics for this forecast
            temp_min = max(0, base_temp_min + temp_variation - 4)
            temp_max = max(temp_min + 4, base_temp_max + temp_variation)
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
                soil_temperature=temp_mean - 2.5,  # Simulated soil temperature
                soil_moisture=max(0, min(100, base_humidity + humid_variation - 8)),
                uv_index=random.uniform(0, 10)  # Meteoblue might provide UV index
            )
            
            # Create the forecast object
            forecast = Forecast(
                source=self.name,
                location_id=location.id or str(location.name).lower().replace(" ", "_"),
                date=forecast_date,
                metrics=metrics,
                raw_data={  # Simulated raw data
                    "source": "Meteoblue",
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
                    "cloud_cover": max(0, min(100, base_cloud_cover + cloud_variation)),
                    "uv_index": metrics.uv_index
                }
            )
            
            forecasts.append(forecast)
        
        return forecasts
