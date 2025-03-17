import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Browser-use imports
from browser_use import Agent, Browser, BrowserConfig, Controller
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Local imports
from app.data_sources.base_source import BaseDataSource
from app.models.location import Location
from app.models.forecast import Forecast, WeatherMetrics
from app.utils.config import Config
from app.utils.logging import get_logger

logger = get_logger()

# Pydantic models for structured output from browser-use


class ForecastDay(BaseModel):
    """Single day forecast data from Meteologix."""
    date: str
    temperature: Dict[str, str]
    precipitation: Dict[str, str]
    wind: str


class MeteologixForecast(BaseModel):
    """Complete forecast data from Meteologix."""
    forecast: List[ForecastDay]


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
        """Fetch weather data from Meteologix using browser-use with structured output."""
        logger.info(
            f"Starting browser-use agent for Meteologix, location ID: {meteologix_location_id}")

        # Try to install browser if it's not already installed
        try:
            import subprocess
            logger.info("Checking if Playwright browser needs to be installed")
            subprocess.run(
                ["python", "-m", "playwright", "install", "chromium"],
                check=False, capture_output=True
            )
        except Exception as e:
            logger.warning(f"Failed to install Playwright browser: {e}")

        # Configure the browser to run headless
        browser = Browser(
            config=BrowserConfig(
                headless=True,
            )
        )

        # Define the initial actions (navigating to the 14-day forecast page)
        url = f"https://meteologix.com/bg/forecast/{meteologix_location_id}/14-day-trend"
        initial_actions = [{'open_tab': {'url': url}}]

        # Set up the controller with our Pydantic model for structured output
        controller = Controller(output_model=MeteologixForecast)

        # Create the task for extracting weather data
        task = """Go to the current page and extract the 14-day weather forecast data.
For each day, collect:
1. The date (YYYY-MM-DD format)
2. Min and max temperature in Celsius (e.g., "5°C" and "14°C")
3. Precipitation probability percentage and total amount in mm (e.g., "20%" and "5mm")
4. Wind speed in kph (e.g., "15 kph")

Structure the data exactly as shown in the output format."""

        # Create the browser-use agent with structured output
        try:
            logger.info("Creating browser-use agent with structured output controller")
            agent = Agent(
                task=task,
                llm=ChatOpenAI(model="gpt-4o", api_key=self.openai_api_key),
                initial_actions=initial_actions,
                browser=browser,
                controller=controller
            )

            # Run the agent and get the result
            logger.info("Running browser-use agent")
            history = await agent.run()

            # Log information about the result
            logger.info(f"Agent run completed with history type: {type(history)}")

            # Extract the structured result using the Pydantic model
            result = history.final_result()
            if result:
                logger.info(f"Got structured result from agent: {result[:100]}...")
                try:
                    # Parse the result as our Pydantic model
                    parsed_forecast = MeteologixForecast.model_validate_json(result)
                    forecast_days = len(parsed_forecast.forecast)
                    logger.info(f"Successfully parsed structured data with {forecast_days} days")

                    # Convert to dictionary for compatibility with existing code
                    return parsed_forecast.model_dump()
                except Exception as e:
                    logger.error(f"Error parsing structured output: {e}")
            else:
                logger.error("No result returned from browser-use agent")

            # If structured output failed, return empty dict
            await browser.close()
            return {}

        except Exception as e:
            logger.error(f"Error in browser-use agent: {e}")
            try:
                await browser.close()
            except Exception:
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
                # Note: We store precipitation probability in raw_data, as it's not in WeatherMetrics
                metrics = WeatherMetrics(
                    temperature_min=temp_min,
                    temperature_max=temp_max,
                    temperature_mean=temp_mean,
                    precipitation=precip_total,
                    humidity=70.0,  # Estimated value, not provided by the API
                    wind_speed=wind_speed,
                    wind_direction=0.0,  # Not provided in the example data
                    cloud_cover=50.0,  # Estimated value, not provided by the API
                    # Soil data not available from this source
                    soil_temperature=None,
                    soil_moisture=None
                )
                
                # Create forecast object with enhanced raw_data
                raw_data = item.copy()  # Start with original data
                # Add processed data that's not in the metrics model
                raw_data["precipitation_probability"] = precip_prob
                
                forecast = Forecast(
                    source=self.name,
                    location_id=location.id or str(location.name).lower().replace(" ", "_"),
                    date=forecast_date,
                    metrics=metrics,
                    raw_data=raw_data
                )
                
                forecasts.append(forecast)
                
            except Exception as e:
                logger.error(f"Error parsing Meteologix forecast item: {e}")
                continue
                
        return forecasts
