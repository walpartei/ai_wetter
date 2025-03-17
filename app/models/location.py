from dataclasses import dataclass
from typing import Optional


@dataclass
class Location:
    """Represents a geographical location in Bulgaria."""
    name: str
    latitude: float
    longitude: float
    region: str
    id: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.name}, {self.region}"
    
    def to_dict(self) -> dict:
        """Convert location to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "region": self.region
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Location':
        """Create location from dictionary."""
        return cls(
            id=data.get("id"),
            name=data["name"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            region=data["region"]
        )
