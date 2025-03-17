import json
import os
from typing import List, Optional
from pathlib import Path

from app.models import Location
from app.utils.config import RESOURCES_DIR
from app.utils.logging import get_logger

logger = get_logger()


class LocationSelector:
    """Component for selecting and managing locations."""
    
    def __init__(self):
        self.locations = self._load_locations()
        
    def _load_locations(self) -> List[Location]:
        """Load locations from file."""
        try:
            locations_file = RESOURCES_DIR / "locations.json"
            if not locations_file.exists():
                logger.warning(f"Locations file not found at {locations_file}")
                return []
                
            with open(locations_file, "r") as f:
                locations_data = json.load(f)
                
            return [Location.from_dict(loc) for loc in locations_data]
        except Exception as e:
            logger.error(f"Error loading locations: {e}")
            return []
    
    def get_locations(self) -> List[Location]:
        """Get all available locations."""
        return self.locations
    
    def get_location_by_id(self, location_id: str) -> Optional[Location]:
        """Get location by ID."""
        return next((loc for loc in self.locations if loc.id == location_id), None)
    
    def get_location_by_name(self, name: str) -> Optional[Location]:
        """Get location by name."""
        return next((loc for loc in self.locations if loc.name.lower() == name.lower()), None)
    
    def get_locations_by_region(self, region: str) -> List[Location]:
        """Get locations in a specific region."""
        return [loc for loc in self.locations if loc.region.lower() == region.lower()]
    
    def render_dropdown(self, selected_id: Optional[str] = None) -> str:
        """Render HTML dropdown for selecting a location."""
        if not self.locations:
            return '<select id="location-select" name="location"><option value="">No locations available</option></select>'
        
        options = []
        for location in sorted(self.locations, key=lambda x: x.name):
            selected = 'selected' if location.id == selected_id else ''
            options.append(f'<option value="{location.id}" {selected}>{location.name}, {location.region}</option>')
            
        return f'<select id="location-select" name="location" class="form-select">{"\n".join(options)}</select>'
