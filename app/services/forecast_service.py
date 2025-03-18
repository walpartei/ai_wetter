from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from app.data_sources import ECMWFDataSource, MeteoblueDataSource, MeteologixDataSource, GenCastDataSource
from app.models import Location, Forecast, CombinedForecast
from app.utils.logging import get_logger
from app.utils.storage import save_forecast_history

logger = get_logger()


class ForecastService:
    """Service for fetching and processing weather forecasts."""
    
    def __init__(self):
        # Initialize data sources
        self.data_sources = [
            ECMWFDataSource(),
            MeteoblueDataSource(),
            MeteologixDataSource(),
            GenCastDataSource()
        ]
        # Filter only available data sources
        self.data_sources = [ds for ds in self.data_sources if ds.is_available()]
        
        if not self.data_sources:
            logger.warning("No data sources available. Please check API configuration.")
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data source names."""
        return [ds.name for ds in self.data_sources]
    
    def get_forecasts(self, location: Location, days: int = 14) -> Dict[str, List[Forecast]]:
        """Get forecasts from all available sources for a location.
        
        Args:
            location: The location to get forecasts for
            days: Number of days to forecast (max 14)
            
        Returns:
            Dictionary mapping source names to lists of forecasts
        """
        forecasts = {}
        
        for source in self.data_sources:
            try:
                logger.info(f"Fetching forecast from {source.name} for {location.name}")
                source_forecasts = source.get_forecast(location, days)
                forecasts[source.name] = source_forecasts
                logger.info(f"Received {len(source_forecasts)} days of forecast from {source.name}")
            except Exception as e:
                logger.error(f"Error getting forecast from {source.name}: {e}")
                forecasts[source.name] = []
        
        # Save to history
        self._save_forecasts_to_history(location, forecasts)
        
        return forecasts
    
    def get_combined_forecasts(self, location: Location, days: int = 14) -> List[CombinedForecast]:
        """Get combined forecasts from all sources for a location.
        
        Args:
            location: The location to get forecasts for
            days: Number of days to forecast (max 14)
            
        Returns:
            List of CombinedForecast objects, one for each day
        """
        all_forecasts = self.get_forecasts(location, days)
        combined_forecasts = []
        
        if not all_forecasts:
            logger.warning(f"No forecasts available for {location.name}")
            return []
        
        # Get the start date from the first forecast of the first source
        # (assuming all sources start forecasts from the same date)
        first_source = list(all_forecasts.keys())[0] if all_forecasts else None
        if not first_source or not all_forecasts[first_source]:
            logger.warning(f"No forecasts available from {first_source}")
            return []
            
        start_date = all_forecasts[first_source][0].date
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        
        # Create a combined forecast for each day
        for day in range(days):
            forecast_date = start_date + timedelta(days=day)
            day_forecasts = []
            
            # Collect forecasts for this day from all sources
            for source_name, source_forecasts in all_forecasts.items():
                if day < len(source_forecasts):
                    day_forecasts.append(source_forecasts[day])
            
            if not day_forecasts:
                logger.warning(f"No forecasts available for {location.name} on {forecast_date}")
                continue
                
            # Create combined forecast
            combined = CombinedForecast(
                location_id=location_id,
                date=forecast_date,
                forecasts=day_forecasts
            )
            
            # Calculate combined metrics and confidence levels
            combined.calculate_combined_metrics()
            
            combined_forecasts.append(combined)
        
        return combined_forecasts
    
    def convert_history_to_combined(self, forecast_data: Dict, location: Location) -> List[CombinedForecast]:
        """Convert historical forecast data to combined forecasts.
        
        Args:
            forecast_data: Dictionary of historical forecast data by source
            location: The location for the forecasts
            
        Returns:
            List of CombinedForecast objects created from historical data
        """
        combined_forecasts = []
        
        # Convert dictionary data back to Forecast objects
        all_forecasts = {}
        for source_name, source_forecasts in forecast_data.items():
            forecasts = []
            for f_data in source_forecasts:
                try:
                    forecast = Forecast.from_dict(f_data)
                    forecasts.append(forecast)
                except Exception as e:
                    logger.error(f"Error converting historical forecast: {e}")
                    continue
            all_forecasts[source_name] = forecasts
            
        # Now process similar to get_combined_forecasts
        if not all_forecasts:
            logger.warning(f"No historical forecasts available for {location.name}")
            return []
            
        # Try to determine the number of days based on the forecasts
        max_days = 0
        for source_forecasts in all_forecasts.values():
            max_days = max(max_days, len(source_forecasts))
            
        # Get the start date from the first forecast of the first source
        first_source = list(all_forecasts.keys())[0] if all_forecasts else None
        if not first_source or not all_forecasts[first_source]:
            logger.warning(f"No historical forecasts available from {first_source}")
            return []
            
        start_date = all_forecasts[first_source][0].date
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        
        # Create a combined forecast for each day
        for day in range(max_days):
            forecast_date = start_date + timedelta(days=day)
            day_forecasts = []
            
            # Collect forecasts for this day from all sources
            for source_name, source_forecasts in all_forecasts.items():
                if day < len(source_forecasts):
                    day_forecasts.append(source_forecasts[day])
            
            if not day_forecasts:
                logger.warning(f"No historical forecasts available for {location.name} on {forecast_date}")
                continue
                
            # Create combined forecast
            combined = CombinedForecast(
                location_id=location_id,
                date=forecast_date,
                forecasts=day_forecasts
            )
            
            # Calculate combined metrics and confidence levels
            combined.calculate_combined_metrics()
            
            combined_forecasts.append(combined)
        
        return combined_forecasts
    
    def _save_forecasts_to_history(self, location: Location, forecasts: Dict[str, List[Forecast]]) -> bool:
        """Save forecasts to history.
        
        Args:
            location: The location
            forecasts: Dictionary of forecasts by source
            
        Returns:
            True if saved successfully, False otherwise
        """
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        
        # Convert forecasts to dictionaries for storage
        forecast_dicts = {}
        for source_name, source_forecasts in forecasts.items():
            forecast_dicts[source_name] = [f.to_dict() for f in source_forecasts]
        
        # Save to history
        try:
            save_forecast_history(location_id, forecast_dicts)
            return True
        except Exception as e:
            logger.error(f"Error saving forecasts to history: {e}")
            return False