from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import statistics
import math


@dataclass
class WeatherMetrics:
    """Weather metrics relevant for agricultural purposes."""
    temperature_min: float  # Celsius
    temperature_max: float  # Celsius
    temperature_mean: float  # Celsius
    precipitation: float  # mm
    humidity: float  # %
    wind_speed: float  # km/h
    wind_direction: float  # degrees
    cloud_cover: float  # %
    soil_temperature: Optional[float] = None  # Celsius
    soil_moisture: Optional[float] = None  # %
    uv_index: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "temperature_mean": self.temperature_mean,
            "precipitation": self.precipitation,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "cloud_cover": self.cloud_cover,
            "soil_temperature": self.soil_temperature,
            "soil_moisture": self.soil_moisture,
            "uv_index": self.uv_index
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WeatherMetrics':
        return cls(**data)


@dataclass
class Forecast:
    """A weather forecast for a specific date and location."""
    source: str  # Name of the data source (ECMWF, Meteoblue, etc.)
    location_id: str  # ID of the location
    date: datetime  # Date of the forecast
    metrics: WeatherMetrics  # Weather metrics
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Original data from source
    retrieved_at: datetime = field(default_factory=datetime.now)  # When the forecast was retrieved
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "location_id": self.location_id,
            "date": self.date.isoformat(),
            "metrics": self.metrics.to_dict(),
            "raw_data": self.raw_data,
            "retrieved_at": self.retrieved_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Forecast':
        metrics = WeatherMetrics.from_dict(data["metrics"])
        return cls(
            source=data["source"],
            location_id=data["location_id"],
            date=datetime.fromisoformat(data["date"]),
            metrics=metrics,
            raw_data=data["raw_data"],
            retrieved_at=datetime.fromisoformat(data["retrieved_at"])
        )


@dataclass
class CombinedForecast:
    """Combined forecast from multiple sources for a specific date and location."""
    location_id: str
    date: datetime
    forecasts: List[Forecast]  # Forecasts from different sources
    combined_metrics: Optional[WeatherMetrics] = None  # Aggregated metrics
    confidence_levels: Dict[str, float] = field(default_factory=dict)  # Confidence in each metric
    
    def calculate_combined_metrics(self) -> None:
        """Calculate combined metrics from all forecasts."""
        if not self.forecasts:
            return
            
        # Simple averaging for now
        temp_min = sum(f.metrics.temperature_min for f in self.forecasts) / len(self.forecasts)
        temp_max = sum(f.metrics.temperature_max for f in self.forecasts) / len(self.forecasts)
        temp_mean = sum(f.metrics.temperature_mean for f in self.forecasts) / len(self.forecasts)
        precip = sum(f.metrics.precipitation for f in self.forecasts) / len(self.forecasts)
        humidity = sum(f.metrics.humidity for f in self.forecasts) / len(self.forecasts)
        wind_speed = sum(f.metrics.wind_speed for f in self.forecasts) / len(self.forecasts)
        
        # For wind direction, we need to handle the circular nature
        sin_sum = sum(math.sin(math.radians(f.metrics.wind_direction)) for f in self.forecasts)
        cos_sum = sum(math.cos(math.radians(f.metrics.wind_direction)) for f in self.forecasts)
        wind_dir = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        
        cloud_cover = sum(f.metrics.cloud_cover for f in self.forecasts) / len(self.forecasts)
        
        # Optional metrics
        soil_temps = [f.metrics.soil_temperature for f in self.forecasts if f.metrics.soil_temperature is not None]
        soil_temp = sum(soil_temps) / len(soil_temps) if soil_temps else None
        
        soil_moist = [f.metrics.soil_moisture for f in self.forecasts if f.metrics.soil_moisture is not None]
        soil_moisture = sum(soil_moist) / len(soil_moist) if soil_moist else None
        
        uv_indices = [f.metrics.uv_index for f in self.forecasts if f.metrics.uv_index is not None]
        uv_index = sum(uv_indices) / len(uv_indices) if uv_indices else None
        
        self.combined_metrics = WeatherMetrics(
            temperature_min=temp_min,
            temperature_max=temp_max,
            temperature_mean=temp_mean,
            precipitation=precip,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            cloud_cover=cloud_cover,
            soil_temperature=soil_temp,
            soil_moisture=soil_moisture,
            uv_index=uv_index
        )
        
        # Calculate confidence levels
        self._calculate_confidence_levels()
    
    def _calculate_confidence_levels(self) -> None:
        """Calculate confidence levels based on agreement between sources."""
        if len(self.forecasts) < 2:
            # Not enough data to calculate confidence
            self.confidence_levels = {
                "temperature": 0.5,
                "precipitation": 0.5,
                "wind": 0.5,
                "overall": 0.5
            }
            return
            
        # Temperature confidence based on standard deviation
        temp_values = [f.metrics.temperature_mean for f in self.forecasts]
        temp_std = statistics.stdev(temp_values) if len(temp_values) > 1 else 0
        # Lower std deviation means higher confidence
        temp_confidence = max(0, min(1, 1 - (temp_std / 10)))  # Normalize, assuming 10°C diff is low confidence
        
        # Precipitation confidence
        precip_values = [f.metrics.precipitation for f in self.forecasts]
        precip_std = statistics.stdev(precip_values) if len(precip_values) > 1 else 0
        precip_confidence = max(0, min(1, 1 - (precip_std / 20)))  # Normalize, 20mm diff is low confidence
        
        # Wind confidence
        wind_values = [f.metrics.wind_speed for f in self.forecasts]
        wind_std = statistics.stdev(wind_values) if len(wind_values) > 1 else 0
        wind_confidence = max(0, min(1, 1 - (wind_std / 15)))  # Normalize, 15km/h diff is low confidence
        
        # Overall confidence is weighted average
        overall = (temp_confidence * 0.4) + (precip_confidence * 0.4) + (wind_confidence * 0.2)
        
        self.confidence_levels = {
            "temperature": temp_confidence,
            "precipitation": precip_confidence,
            "wind": wind_confidence,
            "overall": overall
        }
    
    def to_dict(self) -> dict:
        return {
            "location_id": self.location_id,
            "date": self.date.isoformat(),
            "forecasts": [f.to_dict() for f in self.forecasts],
            "combined_metrics": self.combined_metrics.to_dict() if self.combined_metrics else None,
            "confidence_levels": self.confidence_levels
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CombinedForecast':
        forecasts = [Forecast.from_dict(f) for f in data["forecasts"]]
        combined_metrics = WeatherMetrics.from_dict(data["combined_metrics"]) if data.get("combined_metrics") else None
        
        return cls(
            location_id=data["location_id"],
            date=datetime.fromisoformat(data["date"]),
            forecasts=forecasts,
            combined_metrics=combined_metrics,
            confidence_levels=data.get("confidence_levels", {})
        )
