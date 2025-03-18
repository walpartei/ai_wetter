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
    max_temperature: str
    min_temperature: str
    precipitation_probability: str
    precipitation_total: str
    wind_peak_gust: str


class MeteologixLocation(BaseModel):
    """Forecast for a specific location from Meteologix."""
    # Use a model with extra=True to allow for dynamic location fields
    class Config:
        extra = "allow"
        
    # We'll access the forecast data dynamically in the code


# Alternative format as a fallback
class ClassicForecastDay(BaseModel):
    """Classic format single day forecast data."""
    date: str
    temperature: Dict[str, str]
    precipitation: Dict[str, str]
    wind: str


class ClassicMeteologixForecast(BaseModel):
    """Classic format complete forecast data."""
    forecast: List[ClassicForecastDay]


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
        
    def _convert_date_format(self, date_str: str) -> str:
        """Convert from MM/DD format to YYYY-MM-DD format.
        
        Args:
            date_str: Date in MM/DD format (e.g., "03/18")
            
        Returns:
            Date in YYYY-MM-DD format
        """
        try:
            # Extract month and day
            parts = date_str.split('/')
            if len(parts) != 2:
                raise ValueError(f"Invalid date format: {date_str}")
            
            month = int(parts[0])
            day = int(parts[1])
            
            # Get current year
            current_year = datetime.now().year
            
            # Create date string
            return f"{current_year}-{month:02d}-{day:02d}"
        except Exception as e:
            logger.warning(f"Error converting date format {date_str}: {e}")
            # Fallback: just return the original with current year
            return f"{datetime.now().year}-{date_str.replace('/', '-')}"
    
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
        logger.info(f"Starting Meteologix forecast fetch for {location.name}")
        if not self.is_available():
            logger.warning("Meteologix data source is not available: enabled=%s, has_api_key=%s", 
                         self.enabled, bool(self.openai_api_key))
            return []
        
        # Limit days to 14
        days = min(days, 14)
        logger.info(f"Fetching {days} days of Meteologix forecast data")
        
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        
        # Get the Meteologix location ID
        meteologix_location_id = METEOLOGIX_LOCATION_MAPPING.get(location_id)
        if meteologix_location_id:
            logger.info(f"Using Meteologix location ID: {meteologix_location_id} for {location.name}")
        if not meteologix_location_id:
            logger.warning(f"No Meteologix mapping found for location ID: {location_id}")
            return []
            
        try:
            # Use asyncio to run the browser scraping
            logger.info(f"Starting browser automation for Meteologix forecasts")
            logger.info(f"Fetching Meteologix forecast data for {location.name}")
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
        controller = Controller(output_model=MeteologixLocation)

        # Use location name in the request
        location_name = "Devnya"  # Default to Devnya as fallback
        location_parts = meteologix_location_id.split('-')
        if len(location_parts) > 1:
            # Try to get a better location name from the ID
            location_name = location_parts[1].capitalize()
        
        # Create the task for extracting weather data
        task = f"""Go to the current page and extract the 14-day weather forecast data for {location_name}.
For each day, collect:
1. The date (in MM/DD format, e.g. "03/18")
2. Min and max temperature in Celsius (e.g., "-2°C" and "7°C")
3. Precipitation probability percentage and total amount in mm (e.g., "20%" and "5mm")
4. Wind peak gust in kph (e.g., "46 kph")

Structure the data in exactly the following format:
{{
  "{location_name}_14_day_weather_forecast": [
    {{
      "date": "03/18",
      "max_temperature": "7°C",
      "min_temperature": "-2°C",
      "precipitation_probability": "0%",
      "precipitation_total": "0mm",
      "wind_peak_gust": "46 kph"
    }},
    // more days...
  ]
}}"""

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
            logger.info("Running browser-use agent to extract Meteologix forecast data")
            history = await agent.run()

            # Log information about the result
            logger.info("Browser automation completed successfully")

            # Extract the structured result using the Pydantic model
            logger.info("Extracting structured weather data from browser automation result")
            result = history.final_result()
            if result:
                logger.info("Got structured result from browser agent")
                try:
                    # Try to parse with the new format first
                    logger.info(f"Parsing structured output with Pydantic model")
                    parsed_forecast = MeteologixLocation.model_validate_json(result)
                    
                    # Find the forecast field (it should be the only field ending with _14_day_weather_forecast)
                    forecast_field = None
                    forecast_data = []
                    
                    # Convert to dict to work with dynamic field name
                    forecast_dict = parsed_forecast.model_dump()
                    
                    # Look for the forecast field with dynamic name
                    for field_name, field_value in forecast_dict.items():
                        if field_name.endswith('_14_day_weather_forecast'):
                            forecast_field = field_name
                            forecast_data = field_value
                            break
                            
                    if not forecast_field or not forecast_data:
                        raise ValueError("Could not find forecast data in the parsed result")
                        
                    forecast_days = len(forecast_data)
                    logger.info(f"Successfully parsed structured data with {forecast_days} days of forecast")

                    # Convert to the format expected by our application
                    logger.info("Converting forecast data to internal format")
                    # Create a dictionary with the expected structure
                    converted_forecast = {
                        "forecast": []
                    }
                    
                    # Convert each day's data to the format our application expects
                    for day in forecast_data:
                        # Check if we're working with a dict or object
                        if isinstance(day, dict):
                            # Dictionary access
                            converted_day = {
                                "date": self._convert_date_format(day.get("date", "")),
                                "temperature": {
                                    "min": day.get("min_temperature", "N/A"),
                                    "max": day.get("max_temperature", "N/A")
                                },
                                "precipitation": {
                                    "probability": day.get("precipitation_probability", "0%"),
                                    "total": day.get("precipitation_total", "0mm")
                                },
                                "wind": day.get("wind_peak_gust", "0 kph")
                            }
                        else:
                            # Object attribute access
                            converted_day = {
                                "date": self._convert_date_format(getattr(day, "date", "")),
                                "temperature": {
                                    "min": getattr(day, "min_temperature", "N/A"),
                                    "max": getattr(day, "max_temperature", "N/A")
                                },
                                "precipitation": {
                                    "probability": getattr(day, "precipitation_probability", "0%"),
                                    "total": getattr(day, "precipitation_total", "0mm")
                                },
                                "wind": getattr(day, "wind_peak_gust", "0 kph")
                            }
                        converted_forecast["forecast"].append(converted_day)
                    
                    logger.info("Finalizing Meteologix forecast data")
                    return converted_forecast
                except Exception as e:
                    logger.warning(f"Error parsing with primary format: {e}")
                    
                    # Try fallback format
                    try:
                        logger.info("Trying fallback format parse")
                        # Try to parse with the classic format
                        parsed_forecast = ClassicMeteologixForecast.model_validate_json(result)
                        forecast_days = len(parsed_forecast.forecast)
                        logger.info(f"Successfully parsed with fallback format: {forecast_days} days")
                        
                        # Return the data in the expected format (already correct)
                        return parsed_forecast.model_dump()
                    except Exception as fallback_error:
                        logger.error(f"Error parsing structured output (all formats failed): {fallback_error}")
                        
                        # Last resort: try to convert from raw JSON if possible
                        try:
                            logger.info("Attempting to parse raw JSON")
                            raw_data = json.loads(result)
                            
                            # Look for any field ending with _14_day_weather_forecast
                            forecast_field = None
                            for field in raw_data:
                                if field.endswith('_14_day_weather_forecast'):
                                    forecast_field = field
                                    break
                                    
                            if forecast_field:
                                logger.info(f"Found forecast data in field: {forecast_field}")
                                converted_forecast = {"forecast": []}
                                
                                for day in raw_data[forecast_field]:
                                    # Already using dictionary access in this case
                                    converted_day = {
                                        "date": self._convert_date_format(day.get("date", "")),
                                        "temperature": {
                                            "min": day.get("min_temperature", "N/A"),
                                            "max": day.get("max_temperature", "N/A")
                                        },
                                        "precipitation": {
                                            "probability": day.get("precipitation_probability", "0%"),
                                            "total": day.get("precipitation_total", "0mm")
                                        },
                                        "wind": day.get("wind_peak_gust", "0 kph")
                                    }
                                    converted_forecast["forecast"].append(converted_day)
                                
                                logger.info("Successfully converted raw JSON format")
                                return converted_forecast
                        except Exception as json_error:
                            logger.error(f"All parsing methods failed: {json_error}")
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
