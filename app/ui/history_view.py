import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models import Location


class HistoryView:
    """View component for displaying historical forecast data."""
    
    @staticmethod
    def render_history_list(history: List[Dict[str, Any]], location: Location) -> str:
        """Render an HTML list of historical forecasts."""
        if not history:
            return f'<div class="alert alert-info">No historical forecast data for {location.name}.</div>'
        
        html = f'''
        <h4>Historical Forecasts for {location.name}</h4>
        <div class="list-group mb-4">
        '''
        
        for entry in history:
            try:
                date_str = datetime.fromisoformat(entry["date"]).strftime("%Y-%m-%d %H:%M")
                source_count = len(entry["forecasts"])
                forecast_count = sum(len(forecasts) for source, forecasts in entry["forecasts"].items())
                
                html += f'''
                <a href="#" class="list-group-item list-group-item-action" data-bs-toggle="modal" data-bs-target="#historyModal" data-history-id="{date_str}">
                    <div class="d-flex w-100 justify-content-between">
                        <h5 class="mb-1">Forecast from {date_str}</h5>
                    </div>
                    <p class="mb-1">{source_count} weather sources, {forecast_count} day forecasts</p>
                </a>
                '''
            except (KeyError, ValueError) as e:
                # Skip invalid entries
                continue
                
        html += '</div>'
        return html
    
    @staticmethod
    def render_history_details(history_entry: Dict[str, Any], location: Location) -> str:
        """Render HTML details for a specific historical forecast entry."""
        if not history_entry:
            return '<div class="alert alert-warning">Historical forecast data not found.</div>'
        
        try:
            date_str = datetime.fromisoformat(history_entry["date"]).strftime("%Y-%m-%d %H:%M")
            forecasts = history_entry["forecasts"]
            
            html = f'''
            <div class="modal-header">
                <h5 class="modal-title">Forecast from {date_str}</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
                <h6>Location: {location.name}, {location.region}</h6>
            '''
            
            # For each source
            for source, source_forecasts in forecasts.items():
                html += f'<h5 class="mt-4">{source}</h5>'
                
                # Create a table for this source
                html += '''
                <div class="table-responsive">
                    <table class="table table-sm table-bordered">
                        <thead class="table-light">
                            <tr>
                                <th>Date</th>
                                <th>Temp Min (°C)</th>
                                <th>Temp Max (°C)</th>
                                <th>Precip (mm)</th>
                                <th>Wind (km/h)</th>
                            </tr>
                        </thead>
                        <tbody>
                '''
                
                for forecast in source_forecasts[:7]:  # Limit to 7 days for readability
                    try:
                        forecast_date = datetime.fromisoformat(forecast["date"]).strftime("%a, %b %d")
                        metrics = forecast["metrics"]
                        
                        html += f'''
                        <tr>
                            <td>{forecast_date}</td>
                            <td>{metrics["temperature_min"]:.1f}</td>
                            <td>{metrics["temperature_max"]:.1f}</td>
                            <td>{metrics["precipitation"]:.1f}</td>
                            <td>{metrics["wind_speed"]:.1f}</td>
                        </tr>
                        '''
                    except (KeyError, ValueError):
                        # Skip invalid entries
                        continue
                        
                html += '''
                        </tbody>
                    </table>
                </div>
                '''
            
            html += '''
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
            '''
            
            return html
            
        except (KeyError, ValueError) as e:
            return f'<div class="alert alert-danger">Error parsing historical forecast: {str(e)}</div>'
