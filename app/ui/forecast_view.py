from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models import CombinedForecast
from app.ui.charts import ChartGenerator


class ForecastView:
    """View component for displaying forecast data."""
    
    @staticmethod
    def render_forecast_table(forecasts: List[CombinedForecast]) -> str:
        """Render an HTML table with forecast data."""
        if not forecasts:
            return '<div class="alert alert-warning">No forecast data available.</div>'
        
        html = '''
        <div class="table-responsive">
            <table class="table table-bordered table-hover">
                <thead class="table-light">
                    <tr>
                        <th>Date</th>
                        <th>Temp Min (°C)</th>
                        <th>Temp Max (°C)</th>
                        <th>Precipitation (mm)</th>
                        <th>Humidity (%)</th>
                        <th>Wind (km/h)</th>
                        <th>Cloud Cover (%)</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for forecast in forecasts:
            date_str = forecast.date.strftime("%a, %b %d")
            confidence = forecast.confidence_levels.get("overall", 0.5)
            confidence_class = "success" if confidence > 0.7 else "warning" if confidence > 0.5 else "danger"
            
            html += f'''
            <tr>
                <td>{date_str}</td>
                <td>{forecast.combined_metrics.temperature_min:.1f}</td>
                <td>{forecast.combined_metrics.temperature_max:.1f}</td>
                <td>{forecast.combined_metrics.precipitation:.1f}</td>
                <td>{forecast.combined_metrics.humidity:.0f}</td>
                <td>{forecast.combined_metrics.wind_speed:.1f}</td>
                <td>{forecast.combined_metrics.cloud_cover:.0f}</td>
                <td><span class="badge bg-{confidence_class}">{int(confidence * 100)}%</span></td>
            </tr>
            '''
            
        html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return html
    
    @staticmethod
    def render_forecast_charts(forecasts: List[CombinedForecast]) -> Dict[str, str]:
        """Generate all forecast charts."""
        charts = {}
        
        # Only generate charts if we have forecast data
        if forecasts:
            charts["temperature"] = ChartGenerator.temperature_chart(forecasts)
            charts["precipitation"] = ChartGenerator.precipitation_chart(forecasts)
            charts["wind"] = ChartGenerator.wind_chart(forecasts)
            charts["confidence"] = ChartGenerator.confidence_chart(forecasts)
            
            # Source comparison charts
            charts["comparison_temp"] = ChartGenerator.comparison_chart(forecasts, "temperature_mean")
            charts["comparison_precip"] = ChartGenerator.comparison_chart(forecasts, "precipitation")
            charts["comparison_wind"] = ChartGenerator.comparison_chart(forecasts, "wind_speed")
        
        return charts
    
    @staticmethod
    def render_sources_table(forecasts: List[CombinedForecast]) -> str:
        """Render a table comparing data from different sources."""
        if not forecasts:
            return '<div class="alert alert-warning">No forecast data available.</div>'
        
        # Get all unique sources
        sources = set()
        for cf in forecasts:
            for f in cf.forecasts:
                sources.add(f.source)
        
        if not sources:
            return '<div class="alert alert-warning">No source data available.</div>'
        
        # Create the table header
        html = '''
        <div class="table-responsive">
            <table class="table table-bordered table-sm">
                <thead class="table-light">
                    <tr>
                        <th>Date</th>
        '''
        
        for source in sorted(sources):
            html += f'<th colspan="3">{source}</th>'
        
        html += '''
                    </tr>
                    <tr>
                        <th></th>
        '''
        
        for _ in sources:
            html += '<th>Min (°C)</th><th>Max (°C)</th><th>Precip (mm)</th>'
        
        html += '''
                    </tr>
                </thead>
                <tbody>
        '''
        
        # Create the table rows
        for cf in forecasts[:7]:  # Limit to first 7 days for readability
            date_str = cf.date.strftime("%a, %b %d")
            html += f'<tr><td>{date_str}</td>'
            
            for source in sorted(sources):
                # Find forecast for this source
                source_forecast = next((f for f in cf.forecasts if f.source == source), None)
                
                if source_forecast:
                    html += f'''
                    <td>{source_forecast.metrics.temperature_min:.1f}</td>
                    <td>{source_forecast.metrics.temperature_max:.1f}</td>
                    <td>{source_forecast.metrics.precipitation:.1f}</td>
                    '''
                else:
                    html += '<td>-</td><td>-</td><td>-</td>'
            
            html += '</tr>'
        
        html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return html
