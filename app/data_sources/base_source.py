from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from app.models.location import Location
from app.models.forecast import Forecast


class BaseDataSource(ABC):
    """Abstract base class for all weather data sources."""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def get_forecast(self, location: Location, days: int = 14) -> List[Forecast]:
        """Fetches weather forecast for the given location.
        
        Args:
            location: The location to get forecast for
            days: Number of days to forecast (max 14)
            
        Returns:
            List of Forecast objects for the requested days
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Checks if this data source is properly configured and available."""
        pass
        
    def __str__(self) -> str:
        return f"{self.name} Data Source"
