import unittest
from pathlib import Path

from app.models.location import Location
from app.models.forecast import WeatherMetrics, Forecast, CombinedForecast
from app.data_sources.ecmwf_source import ECMWFDataSource
from app.data_sources.meteoblue_source import MeteoblueDataSource
from app.utils.config import Config


class TestBasicFunctionality(unittest.TestCase):
    """Basic tests for core functionality."""

    def test_location_model(self):
        """Test Location model."""
        location = Location(
            name="Sofia",
            latitude=42.6977,
            longitude=23.3219,
            region="Sofia-Capital",
            id="sofia"
        )
        
        self.assertEqual(location.name, "Sofia")
        self.assertEqual(location.id, "sofia")
        
        # Test to_dict and from_dict
        location_dict = location.to_dict()
        reconstructed = Location.from_dict(location_dict)
        self.assertEqual(location.name, reconstructed.name)
        self.assertEqual(location.id, reconstructed.id)
    
    def test_weather_metrics(self):
        """Test WeatherMetrics model."""
        metrics = WeatherMetrics(
            temperature_min=15.0,
            temperature_max=25.0,
            temperature_mean=20.0,
            precipitation=2.0,
            humidity=70.0,
            wind_speed=10.0,
            wind_direction=180.0,
            cloud_cover=30.0
        )
        
        self.assertEqual(metrics.temperature_min, 15.0)
        self.assertEqual(metrics.temperature_max, 25.0)
        
        # Test to_dict and from_dict
        metrics_dict = metrics.to_dict()
        reconstructed = WeatherMetrics.from_dict(metrics_dict)
        self.assertEqual(metrics.temperature_min, reconstructed.temperature_min)
        self.assertEqual(metrics.precipitation, reconstructed.precipitation)
    
    def test_config(self):
        """Test configuration loading."""
        config = Config()
        api_config = config.get_api_config("ecmwf")
        
        self.assertIsNotNone(api_config)
        self.assertIn("key", api_config)
        self.assertIn("email", api_config)
    
    def test_data_sources_initialization(self):
        """Test data sources can be initialized."""
        ecmwf = ECMWFDataSource()
        meteoblue = MeteoblueDataSource()
        
        self.assertEqual(ecmwf.name, "ECMWF")
        self.assertEqual(meteoblue.name, "Meteoblue")


if __name__ == '__main__':
    unittest.main()