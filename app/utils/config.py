import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Base directories
APP_DIR = Path(__file__).parent.parent.parent
DATA_DIR = APP_DIR / "data"
RESOURCES_DIR = APP_DIR / "app" / "resources"

# Ensure directories exist
HISTORY_DIR = DATA_DIR / "history"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"

for directory in [HISTORY_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API configuration
API_CONFIG = {
    "ecmwf": {
        "url": "https://api.ecmwf.int/v1",
        "key": "b64cad57c8c084de0242df1110ef67e3",
        "email": "l@lll.uno"
    },
    "meteoblue": {
        "key": "rv1lLgwaCJGiuKoF"
    },
    "meteologix": {
        # To be configured later
        "enabled": False
    }
}


class Config:
    """Handles application configuration."""
    
    _instance = None
    _config_data: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        config_path = APP_DIR / "config.json"
        
        if config_path.exists():
            with open(config_path, "r") as f:
                self._config_data = json.load(f)
        else:
            # Use defaults
            self._config_data = {
                "api": API_CONFIG,
                "app": {
                    "default_days": 14,
                    "default_location": "Varna",
                    "theme": "light"
                }
            }
            # Save default config
            self.save_config()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        config_path = APP_DIR / "config.json"
        
        with open(config_path, "w") as f:
            json.dump(self._config_data, f, indent=4)
    
    def get_api_config(self, source: str) -> Dict[str, Any]:
        """Get API configuration for a specific data source."""
        return self._config_data.get("api", {}).get(source, {})
    
    def get_app_setting(self, key: str, default: Any = None) -> Any:
        """Get an application setting."""
        return self._config_data.get("app", {}).get(key, default)
    
    def set_app_setting(self, key: str, value: Any) -> None:
        """Set an application setting."""
        if "app" not in self._config_data:
            self._config_data["app"] = {}
        
        self._config_data["app"][key] = value
        self.save_config()
