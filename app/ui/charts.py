import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from app.models import CombinedForecast


class ChartGenerator:
    """Generates charts for weather visualization."""
    
    @staticmethod
    def temperature_chart(forecasts: List[CombinedForecast], width: int = 10, height: int = 6) -> str:
        """Generate a temperature chart for the forecasts.
        
        Args:
            forecasts: List of combined forecasts
            width: Chart width in inches
            height: Chart height in inches
            
        Returns:
            Base64 encoded PNG image
        """
        if not forecasts:
            return ""
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Extract dates and temperatures
        dates = [f.date for f in forecasts]
        temp_min = [f.combined_metrics.temperature_min for f in forecasts]
        temp_max = [f.combined_metrics.temperature_max for f in forecasts]
        temp_mean = [f.combined_metrics.temperature_mean for f in forecasts]
        
        # Plot temperature data
        ax.plot(dates, temp_mean, 'o-', color='blue', label='Average')
        ax.fill_between(dates, temp_min, temp_max, alpha=0.2, color='blue')
        ax.plot(dates, temp_min, '--', color='lightblue', label='Min')
        ax.plot(dates, temp_max, '--', color='darkblue', label='Max')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Forecast')
        
        # Format x-axis to show dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add confidence annotations
        for i, forecast in enumerate(forecasts):
            confidence = forecast.confidence_levels.get('temperature', 0.5)
            color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
            ax.annotate(f"{int(confidence*100)}%", 
                        (dates[i], temp_max[i] + 1),
                        ha='center',
                        color=color,
                        fontsize=8)
        
        # Ensure layout looks good
        plt.tight_layout()
        
        # Convert to base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        # Encode and return
        encoded = base64.b64encode(image_png).decode('utf-8')
        return encoded
    
    @staticmethod
    def precipitation_chart(forecasts: List[CombinedForecast], width: int = 10, height: int = 6) -> str:
        """Generate a precipitation chart for the forecasts.
        
        Args:
            forecasts: List of combined forecasts
            width: Chart width in inches
            height: Chart height in inches
            
        Returns:
            Base64 encoded PNG image
        """
        if not forecasts:
            return ""
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Extract dates and precipitation
        dates = [f.date for f in forecasts]
        precip = [f.combined_metrics.precipitation for f in forecasts]
        humidity = [f.combined_metrics.humidity for f in forecasts]
        
        # Create a bar chart for precipitation
        bars = ax.bar(dates, precip, width=0.7, color='blue', alpha=0.7, label='Precipitation (mm)')
        
        # Create a second y-axis for humidity
        ax2 = ax.twinx()
        ax2.plot(dates, humidity, 'o-', color='green', label='Humidity (%)')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Humidity (%)')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Precipitation (mm)')
        ax.set_title('Precipitation and Humidity Forecast')
        
        # Format x-axis to show dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add confidence annotations
        for i, forecast in enumerate(forecasts):
            confidence = forecast.confidence_levels.get('precipitation', 0.5)
            color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
            if precip[i] > 0.5:  # Only annotate if there's precipitation
                ax.annotate(f"{int(confidence*100)}%", 
                           (dates[i], precip[i] + 0.5),
                           ha='center',
                           color=color,
                           fontsize=8)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Ensure layout looks good
        plt.tight_layout()
        
        # Convert to base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        # Encode and return
        encoded = base64.b64encode(image_png).decode('utf-8')
        return encoded
    
    @staticmethod
    def wind_chart(forecasts: List[CombinedForecast], width: int = 10, height: int = 6) -> str:
        """Generate a wind chart for the forecasts.
        
        Args:
            forecasts: List of combined forecasts
            width: Chart width in inches
            height: Chart height in inches
            
        Returns:
            Base64 encoded PNG image
        """
        if not forecasts:
            return ""
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Extract dates and wind data
        dates = [f.date for f in forecasts]
        wind_speed = [f.combined_metrics.wind_speed for f in forecasts]
        wind_direction = [f.combined_metrics.wind_direction for f in forecasts]
        
        # Plot wind speed
        ax.plot(dates, wind_speed, 'o-', color='blue', label='Wind Speed (km/h)')
        
        # Add wind direction arrows
        for i, (date, speed, direction) in enumerate(zip(dates, wind_speed, wind_direction)):
            # Convert direction to radians (meteorological angle: 0° = North, 90° = East)
            rad = (270 - direction) * (3.14159 / 180)
            dx = 0.3 * speed * abs(abs(rad)) * 0.04
            dy = 0.3 * speed * abs(abs(rad)) * 0.04
            ax.arrow(mdates.date2num(date), speed, dx, dy, 
                     head_width=0.5, head_length=0.7, fc='k', ec='k')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Wind Speed (km/h)')
        ax.set_title('Wind Forecast')
        
        # Format x-axis to show dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add confidence annotations
        for i, forecast in enumerate(forecasts):
            confidence = forecast.confidence_levels.get('wind', 0.5)
            color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
            ax.annotate(f"{int(confidence*100)}%", 
                        (dates[i], wind_speed[i] + 1),
                        ha='center',
                        color=color,
                        fontsize=8)
        
        # Add threshold lines for wind categories
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Moderate Wind')
        ax.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Strong Wind')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ensure layout looks good
        plt.tight_layout()
        
        # Convert to base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        # Encode and return
        encoded = base64.b64encode(image_png).decode('utf-8')
        return encoded
    
    @staticmethod
    def comparison_chart(forecasts: List[CombinedForecast], metric: str = 'temperature_mean', 
                         width: int = 10, height: int = 6) -> str:
        """Generate a chart comparing forecasts from different sources.
        
        Args:
            forecasts: List of combined forecasts
            metric: The metric to compare (temperature_mean, precipitation, etc.)
            width: Chart width in inches
            height: Chart height in inches
            
        Returns:
            Base64 encoded PNG image
        """
        if not forecasts or not forecasts[0].forecasts:
            return ""
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Extract dates
        dates = [f.date for f in forecasts]
        
        # Get available sources
        sources = set()
        for cf in forecasts:
            for f in cf.forecasts:
                sources.add(f.source)
        
        # Plot data for each source
        colors = {'ECMWF': 'blue', 'Meteoblue': 'green', 'Meteologix': 'red'}
        for source in sources:
            # Extract values for this source
            values = []
            for cf in forecasts:
                # Find forecast for this source and day
                source_forecast = next((f for f in cf.forecasts if f.source == source), None)
                if source_forecast:
                    # Get the metric value
                    if metric == 'temperature_mean':
                        values.append(source_forecast.metrics.temperature_mean)
                    elif metric == 'temperature_min':
                        values.append(source_forecast.metrics.temperature_min)
                    elif metric == 'temperature_max':
                        values.append(source_forecast.metrics.temperature_max)
                    elif metric == 'precipitation':
                        values.append(source_forecast.metrics.precipitation)
                    elif metric == 'humidity':
                        values.append(source_forecast.metrics.humidity)
                    elif metric == 'wind_speed':
                        values.append(source_forecast.metrics.wind_speed)
                    else:
                        values.append(0)
                else:
                    values.append(None)
            
            # Plot this source
            color = colors.get(source, 'gray')
            ax.plot(dates, values, 'o-', color=color, label=source)
        
        # Add combined forecast
        combined_values = []
        for cf in forecasts:
            if metric == 'temperature_mean':
                combined_values.append(cf.combined_metrics.temperature_mean)
            elif metric == 'temperature_min':
                combined_values.append(cf.combined_metrics.temperature_min)
            elif metric == 'temperature_max':
                combined_values.append(cf.combined_metrics.temperature_max)
            elif metric == 'precipitation':
                combined_values.append(cf.combined_metrics.precipitation)
            elif metric == 'humidity':
                combined_values.append(cf.combined_metrics.humidity)
            elif metric == 'wind_speed':
                combined_values.append(cf.combined_metrics.wind_speed)
            else:
                combined_values.append(0)
                
        ax.plot(dates, combined_values, 'o-', color='black', linewidth=2, label='Combined')
        
        # Add labels and title
        ax.set_xlabel('Date')
        metric_name = metric.replace('_', ' ').title()
        unit = 'mm' if metric == 'precipitation' else '°C' if 'temperature' in metric else '%' if metric == 'humidity' else 'km/h' if metric == 'wind_speed' else ''
        ax.set_ylabel(f'{metric_name} ({unit})')
        ax.set_title(f'Source Comparison: {metric_name}')
        
        # Format x-axis to show dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ensure layout looks good
        plt.tight_layout()
        
        # Convert to base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        # Encode and return
        encoded = base64.b64encode(image_png).decode('utf-8')
        return encoded
    
    @staticmethod
    def confidence_chart(forecasts: List[CombinedForecast], width: int = 10, height: int = 6) -> str:
        """Generate a chart showing confidence levels for the forecasts.
        
        Args:
            forecasts: List of combined forecasts
            width: Chart width in inches
            height: Chart height in inches
            
        Returns:
            Base64 encoded PNG image
        """
        if not forecasts:
            return ""
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Extract dates and confidence levels
        dates = [f.date for f in forecasts]
        temp_conf = [f.confidence_levels.get('temperature', 0.5) for f in forecasts]
        precip_conf = [f.confidence_levels.get('precipitation', 0.5) for f in forecasts]
        wind_conf = [f.confidence_levels.get('wind', 0.5) for f in forecasts]
        overall_conf = [f.confidence_levels.get('overall', 0.5) for f in forecasts]
        
        # Plot confidence levels
        ax.plot(dates, temp_conf, 'o-', color='red', label='Temperature')
        ax.plot(dates, precip_conf, 'o-', color='blue', label='Precipitation')
        ax.plot(dates, wind_conf, 'o-', color='green', label='Wind')
        ax.plot(dates, overall_conf, 'o-', color='black', linewidth=2, label='Overall')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Confidence Level')
        ax.set_title('Forecast Confidence')
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # Add threshold lines
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='High Confidence')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Low Confidence')
        
        # Format x-axis to show dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ensure layout looks good
        plt.tight_layout()
        
        # Convert to base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        # Encode and return
        encoded = base64.b64encode(image_png).decode('utf-8')
        return encoded