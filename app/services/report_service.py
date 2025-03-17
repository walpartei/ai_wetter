from datetime import datetime
from typing import List, Dict, Any, Optional

from app.models import Location, CombinedForecast
from app.services.forecast_service import ForecastService
from app.services.history_service import HistoryService
from app.utils.logging import get_logger
from app.utils.storage import save_report, load_reports

logger = get_logger()


class ReportService:
    """Service for generating and managing weather reports."""
    
    def __init__(self):
        self.forecast_service = ForecastService()
        self.history_service = HistoryService()
    
    def generate_report(self, location: Location, days: int = 14) -> Dict[str, Any]:
        """Generate a comprehensive weather report for the location.
        
        Args:
            location: The location to generate report for
            days: Number of days to include in the report
            
        Returns:
            Report data dictionary
        """
        logger.info(f"Generating report for {location.name} for {days} days")
        
        # Get forecasts from all sources
        combined_forecasts = self.forecast_service.get_combined_forecasts(location, days)
        
        if not combined_forecasts:
            logger.warning(f"No forecasts available for {location.name}")
            return self._get_empty_report(location)
        
        # Get historical accuracy data
        accuracy_data = self.history_service.get_forecast_accuracy(location)
        
        # Create report data
        report = {
            "location": location.to_dict(),
            "generated_at": datetime.now().isoformat(),
            "days": days,
            "forecasts": [cf.to_dict() for cf in combined_forecasts],
            "accuracy": accuracy_data,
            "recommendations": self._generate_recommendations(combined_forecasts, accuracy_data)
        }
        
        # Save report
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        report_path = save_report(location_id, report)
        
        if report_path:
            logger.info(f"Report saved to {report_path}")
            report["file_path"] = report_path
        
        return report
    
    def get_saved_reports(self, location: Location, limit: int = 10) -> List[Dict[str, Any]]:
        """Get previously saved reports for a location.
        
        Args:
            location: The location to get reports for
            limit: Maximum number of reports to return
            
        Returns:
            List of report dictionaries
        """
        location_id = location.id or str(location.name).lower().replace(" ", "_")
        
        try:
            reports = load_reports(location_id, limit)
            logger.info(f"Loaded {len(reports)} reports for {location.name}")
            return reports
        except Exception as e:
            logger.error(f"Error loading reports: {e}")
            return []
    
    def _get_empty_report(self, location: Location) -> Dict[str, Any]:
        """Generate an empty report when no forecasts are available.
        
        Args:
            location: The location
            
        Returns:
            Empty report data dictionary
        """
        return {
            "location": location.to_dict(),
            "generated_at": datetime.now().isoformat(),
            "days": 0,
            "forecasts": [],
            "accuracy": {},
            "recommendations": {
                "error": "No forecast data available. Please check API configuration or try again later."
            }
        }
    
    def _generate_recommendations(self, forecasts: List[CombinedForecast], accuracy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on the forecast data and historical accuracy.
        
        Args:
            forecasts: List of combined forecasts
            accuracy: Historical accuracy data
            
        Returns:
            Recommendations dictionary
        """
        # Extract best sources from accuracy data
        best_sources = accuracy.get("best_source", {})
        
        # Get available sources
        available_sources = self.forecast_service.get_available_sources()
        
        # Generate general recommendations
        recommendations = {
            "summary": self._generate_summary(forecasts),
            "best_sources": {
                "daily_planning": best_sources.get("short_term", "Meteoblue"),
                "medium_term": best_sources.get("medium_term", "ECMWF"),
                "cross_checking": "Meteologix" if "Meteologix" in available_sources else "Meteoblue"
            },
            "agricultural_advice": self._generate_agricultural_advice(forecasts),
            "confidence_levels": self._calculate_overall_confidence(forecasts)
        }
        
        return recommendations
    
    def _generate_summary(self, forecasts: List[CombinedForecast]) -> str:
        """Generate a text summary of the forecast.
        
        Args:
            forecasts: List of combined forecasts
            
        Returns:
            Summary text
        """
        if not forecasts:
            return "No forecast data available."
        
        # Group forecasts into periods
        short_term = forecasts[:3] if len(forecasts) >= 3 else forecasts
        medium_term = forecasts[3:7] if len(forecasts) >= 7 else forecasts[3:] if len(forecasts) >= 3 else []
        long_term = forecasts[7:] if len(forecasts) >= 7 else []
        
        # Generate summary for short term (1-3 days)
        summary = "Weather Forecast Summary:\n\n"
        
        if short_term:
            avg_temp = sum(f.combined_metrics.temperature_mean for f in short_term) / len(short_term)
            has_rain = any(f.combined_metrics.precipitation > 1.0 for f in short_term)
            high_wind = any(f.combined_metrics.wind_speed > 20.0 for f in short_term)
            
            summary += f"Short-term (1-3 days): Average temperature around {avg_temp:.1f}°C. "
            summary += "Expect some rainfall. " if has_rain else "Mostly dry. "
            summary += "Windy conditions expected. " if high_wind else ""
        
        # Generate summary for medium term (4-7 days)
        if medium_term:
            avg_temp = sum(f.combined_metrics.temperature_mean for f in medium_term) / len(medium_term)
            has_rain = any(f.combined_metrics.precipitation > 1.0 for f in medium_term)
            
            summary += f"\n\nMedium-term (4-7 days): Temperatures around {avg_temp:.1f}°C. "
            summary += "Some precipitation expected. " if has_rain else "Generally dry conditions. "
        
        # Generate summary for long term (8-14 days)
        if long_term:
            avg_temp = sum(f.combined_metrics.temperature_mean for f in long_term) / len(long_term)
            trend = "warming" if avg_temp > (short_term[0].combined_metrics.temperature_mean if short_term else 20) else "cooling"
            
            summary += f"\n\nLong-term (8-14 days): Generally {trend} trend with average temperatures around {avg_temp:.1f}°C. "
            summary += "Long-range forecasts have lower confidence, check updated forecasts regularly."
        
        return summary
    
    def _generate_agricultural_advice(self, forecasts: List[CombinedForecast]) -> Dict[str, str]:
        """Generate agricultural advice based on the forecast data.
        
        Args:
            forecasts: List of combined forecasts
            
        Returns:
            Dictionary of agricultural advice by category
        """
        if not forecasts:
            return {"error": "No forecast data available for agricultural advice."}
        
        # Check for significant weather events
        has_heavy_rain = any(f.combined_metrics.precipitation > 10.0 for f in forecasts)
        has_frost = any(f.combined_metrics.temperature_min < 2.0 for f in forecasts)
        has_heat = any(f.combined_metrics.temperature_max > 30.0 for f in forecasts)
        has_strong_wind = any(f.combined_metrics.wind_speed > 30.0 for f in forecasts)
        
        # Generate advice for different categories
        irrigation_advice = "Consider reducing irrigation" if has_heavy_rain else "Regular irrigation recommended"
        
        planting_advice = "Delay planting due to expected frost" if has_frost else \
                         "Avoid planting during peak heat" if has_heat else \
                         "Favorable conditions for planting"
        
        harvesting_advice = "Unfavorable conditions for harvesting due to precipitation" if has_heavy_rain else \
                           "Secure crops due to expected strong winds" if has_strong_wind else \
                           "Good conditions for harvesting"
        
        # Check soil conditions if available
        soil_temp_available = any(f.combined_metrics.soil_temperature is not None for f in forecasts)
        soil_moisture_available = any(f.combined_metrics.soil_moisture is not None for f in forecasts)
        
        soil_advice = ""
        if soil_temp_available and soil_moisture_available:
            avg_soil_temp = sum(f.combined_metrics.soil_temperature for f in forecasts if f.combined_metrics.soil_temperature is not None) / \
                         sum(1 for f in forecasts if f.combined_metrics.soil_temperature is not None)
            
            avg_soil_moisture = sum(f.combined_metrics.soil_moisture for f in forecasts if f.combined_metrics.soil_moisture is not None) / \
                              sum(1 for f in forecasts if f.combined_metrics.soil_moisture is not None)
            
            soil_advice = f"Average soil temperature: {avg_soil_temp:.1f}°C, soil moisture: {avg_soil_moisture:.1f}%. "
            soil_advice += "Soil conditions are favorable for field work." if (5 < avg_soil_temp < 25 and 30 < avg_soil_moisture < 70) else \
                          "Soil conditions may not be optimal for field work."
        
        return {
            "irrigation": irrigation_advice,
            "planting": planting_advice,
            "harvesting": harvesting_advice,
            "soil": soil_advice if soil_advice else "Soil data not available."
        }
    
    def _calculate_overall_confidence(self, forecasts: List[CombinedForecast]) -> Dict[str, Any]:
        """Calculate overall confidence levels for the forecast.
        
        Args:
            forecasts: List of combined forecasts
            
        Returns:
            Dictionary of confidence levels
        """
        if not forecasts:
            return {"overall": 0.0}
        
        # Group forecasts into periods
        short_term = forecasts[:3] if len(forecasts) >= 3 else forecasts
        medium_term = forecasts[3:7] if len(forecasts) >= 7 else forecasts[3:] if len(forecasts) >= 3 else []
        long_term = forecasts[7:] if len(forecasts) >= 7 else []
        
        # Calculate confidence for each period
        short_term_conf = sum(f.confidence_levels.get("overall", 0.5) for f in short_term) / len(short_term) if short_term else 0.0
        medium_term_conf = sum(f.confidence_levels.get("overall", 0.5) for f in medium_term) / len(medium_term) if medium_term else 0.0
        long_term_conf = sum(f.confidence_levels.get("overall", 0.5) for f in long_term) / len(long_term) if long_term else 0.0
        
        # Overall confidence weighted by period
        overall = 0.5 * short_term_conf + 0.3 * medium_term_conf + 0.2 * long_term_conf if long_term else \
                 0.7 * short_term_conf + 0.3 * medium_term_conf if medium_term else \
                 short_term_conf
        
        return {
            "short_term": short_term_conf,
            "medium_term": medium_term_conf,
            "long_term": long_term_conf,
            "overall": overall
        }
