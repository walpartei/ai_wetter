import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Browser-use imports
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI

# Local imports
from app.data_sources.base_source import BaseDataSource
from app.models.location import Location
from app.models.forecast import Forecast, WeatherMetrics
from app.utils.config import Config
from app.utils.logging import get_logger

logger = get_logger()

# Dictionary mapping our location IDs to Meteologix location IDs
METEOLOGIX_LOCATION_MAPPING = {
    "balchik": "733515-baltchik",
    "blagoevgrad": "733191-blagoevgrad",
    "burgas": "732770-burgas",
    "devnya": "732280-devnya",
    "dobrich": "726418-dobrich",
    "gabrovo": "731549-gabrovo",
    "general_toshevo": "731464-general-toshevo",
    "haskovo": "730435-haskovo",
    "kardzhali": "729794-kardzhali",
    "kyustendil": "729730-kyustendil",
    "lovech": "729559-lovech",
    "montana": "729114-montana",
    "pazardzhik": "728378-pazardzhik",
    "pernik": "728330-pernik",
    "pleven": "728203-pleven",
    "plovdiv": "728193-plovdiv",
    "razgrad": "727696-razgrad",
    "ruse": "727523-rousse",
    "shumen": "727233-shumen",
    "silistra": "727221-silistra",
    "sliven": "727079-sliven",
    "smolyan": "727030-smolyan",
    "sofia": "727011-sofia",
    "stara_zagora": "726848-stara-zagora",
    "targovishte": "726174-targovishte",
    "varna": "726050-varna",
    "veliko_tarnovo": "725993-veliko-tarnovo",
    "vidin": "725905-vidin",
    "vratsa": "725712-vratsa",
    "yambol": "725578-yambol",
}


class MeteologixDataSource(BaseDataSource):
    """Data source for Meteologix weather forecasts using browser automation."""
    
    def __init__(self):
        super().__init__("Meteologix")
        self.config = Config().get_api_config("meteologix")
        self.enabled = self.config.get("enabled", False)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    def is_available(self) -> bool:
        """Check if Meteologix data source is properly configured and available."""
        # Check if the source is enabled in config and OpenAI API key is set
        has_config = self.enabled and bool(self.openai_api_key)
        
        # Try to check if Playwright is properly installed
        try:
            import playwright
            return has_config
        except Exception as e:
            logger.warning(f"Playwright not properly installed: {e}")
            return False
    
    def get_forecast(self, location: Location, days: int = 14) -> List[Forecast]:
        """Get weather forecast for a location using Meteologix data.
        
        Args:
            location: The location to get forecast for
            days: Number of days to forecast (max 14)
            
        Returns:
            List of Forecast objects for the requested days
        """
        if not self.is_available():
            logger.warning("Meteologix data source is not available: enabled=%s, has_api_key=%s", 
                         self.enabled, bool(self.openai_api_key))
            return []
        
        # Limit days to 14
        days = min(days, 14)
        
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        
        # Get the Meteologix location ID
        meteologix_location_id = METEOLOGIX_LOCATION_MAPPING.get(location_id)
        if not meteologix_location_id:
            logger.warning(f"No Meteologix mapping found for location ID: {location_id}")
            return []
            
        try:
            # Use asyncio to run the browser scraping
            logger.info(f"Fetching Meteologix forecast for {location.name}")
            forecast_data = asyncio.run(self._fetch_meteologix_data(meteologix_location_id))
            
            if not forecast_data or "forecast" not in forecast_data:
                logger.error(f"Invalid or empty response from Meteologix for {location.name}")
                return []
                
            # Convert the scraped data to our forecast model
            forecasts = self._convert_to_forecasts(forecast_data, location, days)
            logger.info(f"Successfully parsed {len(forecasts)} days of Meteologix forecast for {location.name}")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error getting Meteologix forecast: {e}")
            return []
    
    async def _fetch_meteologix_data(self, meteologix_location_id: str) -> Dict[str, Any]:
        """Fetch weather data from Meteologix using browser-use."""
        logger.info(f"Starting browser-use agent for Meteologix, location ID: {meteologix_location_id}")
        
        # Try to install browser if it's not already installed
        # But with a short timeout to prevent worker hanging
        try:
            import subprocess
            import threading
            
            def install_browser():
                try:
                    logger.info("Attempting to install Playwright browser (with timeout)")
                    result = subprocess.run(
                        ["python", "-m", "playwright", "install", "chromium"],
                        check=False, capture_output=True, timeout=5
                    )
                    logger.info(f"Browser installation finished with code: {result.returncode}")
                except subprocess.TimeoutExpired:
                    logger.warning("Browser installation timed out")
                except Exception as e:
                    logger.warning(f"Failed to install Playwright browser: {e}")
                    
            # Start installation in a separate thread with a timeout
            logger.info("Starting browser installation in background thread")
            install_thread = threading.Thread(target=install_browser)
            install_thread.daemon = True
            install_thread.start()
            
            # Wait for up to 5 seconds
            install_thread.join(5.0)
            
            if install_thread.is_alive():
                logger.warning("Browser installation is taking too long, continuing without waiting")
                
        except Exception as e:
            logger.warning(f"Failed to start browser installation: {e}")
        
        # Define a timeout for the entire browser operation (20 seconds)
        browser_timeout = 20  # seconds
        
        # Try to initialize the browser with a short timeout
        try:
            import asyncio
            import concurrent.futures
            from concurrent.futures import TimeoutError
            
            # Configure the browser to run headless
            browser = Browser(
                config=BrowserConfig(
                    headless=True,
                )
            )
            
            # Define the initial actions (navigating to the 14-day forecast page)
            initial_actions = [
                {'open_tab': {'url': f'https://meteologix.com/bg/forecast/{meteologix_location_id}/14-day-trend'}}
            ]
            
            # Create the task for extracting weather data
            task = """On the current site, please check the weather forecast for the next 14 days, and create a JSON summary including temperature, precipitation, and wind. Please ONLY output json in the following format:

{
  "forecast": [
    {
      "date": "2025-03-17",
      "temperature": {
        "min": "5°C",
        "max": "14°C"
      },
      "precipitation": {
        "probability": "0%",
        "total": "0mm"
      },
      "wind": "15 kph"
    },
    ...
  ]
}
"""
            
            # Create a future to run the agent with a timeout
            async def run_agent_with_timeout():
                try:
                    # Create the browser-use agent
                    logger.info("Creating browser-use agent")
                    agent = Agent(
                        task=task,
                        llm=ChatOpenAI(model="gpt-4o", api_key=self.openai_api_key),
                        initial_actions=initial_actions,
                        browser=browser,
                    )
                    
                    # Run the agent with a timeout
                    logger.info("Running browser-use agent with timeout")
                    await asyncio.wait_for(agent.run(), timeout=browser_timeout)
                    
                    # Parse the agent's response to extract JSON
                    response = agent.messages[-1].content
                    logger.info(f"Agent response received, length: {len(response)} chars")
                    
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # If no JSON code block is found, try the entire response
                        json_str = response
                        
                    # Parse the JSON
                    data = json.loads(json_str)
                    return data
                except asyncio.TimeoutError:
                    logger.error(f"Browser-use operation timed out after {browser_timeout} seconds")
                    raise
                except Exception as e:
                    logger.error(f"Error in browser automation: {e}")
                    raise
                finally:
                    try:
                        await browser.close()
                    except:
                        pass
            
            # Execute with timeout
            logger.info("Starting browser operations with timeout protection")
            data = await asyncio.wait_for(run_agent_with_timeout(), timeout=browser_timeout)
            return data
            
        except Exception as e:
            logger.error(f"Error in browser-use agent: {e}")
            try:
                await browser.close()
            except:
                pass
            return {}
    
    def _convert_to_forecasts(self, data: Dict[str, Any], location: Location, days: int) -> List[Forecast]:
        """Convert Meteologix forecast data to our Forecast model."""
        forecasts = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get the forecast items, limit to the requested number of days
        forecast_items = data.get("forecast", [])[:days]
        
        for i, item in enumerate(forecast_items):
            try:
                # Parse date
                date_str = item.get("date")
                try:
                    forecast_date = datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    # If date parsing fails, use today + days offset
                    forecast_date = today + timedelta(days=i)
                
                # Parse temperature
                temp_data = item.get("temperature", {})
                try:
                    temp_min = float(temp_data.get("min", "0").replace("°C", ""))
                except:
                    temp_min = 0.0
                    
                try:
                    temp_max = float(temp_data.get("max", "0").replace("°C", ""))
                except:
                    temp_max = 0.0
                    
                temp_mean = (temp_min + temp_max) / 2
                
                # Parse precipitation
                precip_data = item.get("precipitation", {})
                try:
                    precip_prob = float(precip_data.get("probability", "0%").replace("%", "")) / 100
                except:
                    precip_prob = 0.0
                    
                try:
                    precip_total = float(precip_data.get("total", "0mm").replace("mm", ""))
                except:
                    precip_total = 0.0
                
                # Parse wind
                wind_str = item.get("wind", "0 kph")
                try:
                    wind_speed = float(wind_str.replace("kph", "").strip())
                except:
                    wind_speed = 0.0
                
                # Create weather metrics
                metrics = WeatherMetrics(
                    temperature_min=temp_min,
                    temperature_max=temp_max,
                    temperature_mean=temp_mean,
                    precipitation=precip_total,
                    precipitation_probability=precip_prob,
                    humidity=70.0,  # Estimated value, not provided by the API
                    wind_speed=wind_speed,
                    wind_direction=0.0,  # Not provided in the example data
                    cloud_cover=50.0,  # Estimated value, not provided by the API
                    # Soil data not available from this source
                    soil_temperature=None,
                    soil_moisture=None
                )
                
                # Create forecast object
                forecast = Forecast(
                    source=self.name,
                    location_id=location.id or str(location.name).lower().replace(" ", "_"),
                    date=forecast_date,
                    metrics=metrics,
                    raw_data=item  # Store the original data as raw_data
                )
                
                forecasts.append(forecast)
                
            except Exception as e:
                logger.error(f"Error parsing Meteologix forecast item: {e}")
                continue
                
        return forecasts
