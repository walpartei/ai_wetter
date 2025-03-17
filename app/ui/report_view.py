from typing import Dict, Any, Optional
from datetime import datetime

from app.models import Location


class ReportView:
    """View component for displaying weather reports."""
    
    @staticmethod
    def render_report(report: Dict[str, Any], charts: Dict[str, str]) -> str:
        """Render HTML for a weather report."""
        if not report:
            return '<div class="alert alert-warning">No report data available.</div>'
        
        # Basic report information
        try:
            location = report.get("location", {})
            location_name = f"{location.get('name', 'Unknown')}, {location.get('region', 'Unknown')}"
            generated_at = datetime.fromisoformat(report.get("generated_at", datetime.now().isoformat()))
            date_str = generated_at.strftime("%Y-%m-%d %H:%M")
            days = report.get("days", 0)
            
            # Start building the HTML
            html = (
                f'<div class="card mb-4">'
                f'<div class="card-header bg-primary text-white">'
                f'<h4>Weather Report for {location_name}</h4>'
                f'<p class="mb-0">Generated on {date_str}, {days} day forecast</p>'
                f'</div>'
                f'<div class="card-body">'
            )
            
            # Recommendations
            recommendations = report.get("recommendations", {})
            if "error" in recommendations:
                html += f'<div class="alert alert-warning">{recommendations["error"]}</div>'
            else:
                # Summary
                summary = recommendations.get("summary", "")
                if summary:
                    html += (
                        f'<div class="card mb-4">'
                        f'<div class="card-header bg-info text-white">'
                        f'<h5>Summary</h5>'
                        f'</div>'
                        f'<div class="card-body">'
                        f'<p>{summary.replace(chr(10), "<br>")}</p>'
                        f'</div>'
                        f'</div>'
                    )
                
                # Best sources recommendation
                best_sources = recommendations.get("best_sources", {})
                if best_sources:
                    html += (
                        '<div class="card mb-4">'
                        '<div class="card-header bg-success text-white">'
                        '<h5>Recommended Weather Sources</h5>'
                        '</div>'
                        '<div class="card-body">'
                        '<ul class="list-group">'
                    )
                    
                    for use, source in best_sources.items():
                        use_display = use.replace("_", " ").title()
                        html += f'<li class="list-group-item"><strong>{use_display}:</strong> {source}</li>'
                    
                    html += (
                            '</ul>'
                            '</div>'
                            '</div>'
                    )
                
                # Agricultural advice
                ag_advice = recommendations.get("agricultural_advice", {})
                if ag_advice and "error" not in ag_advice:
                    html += (
                        '<div class="card mb-4">'
                        '<div class="card-header bg-success text-white">'
                        '<h5>Agricultural Recommendations</h5>'
                        '</div>'
                        '<div class="card-body">'
                        '<ul class="list-group">'
                    )
                    
                    for category, advice in ag_advice.items():
                        category_display = category.title()
                        html += f'<li class="list-group-item"><strong>{category_display}:</strong> {advice}</li>'
                    
                    html += (
                            '</ul>'
                            '</div>'
                            '</div>'
                    )
                
                # Confidence levels
                confidence = recommendations.get("confidence_levels", {})
                if confidence:
                    html += (
                        '<div class="card mb-4">'
                        '<div class="card-header bg-info text-white">'
                        '<h5>Forecast Confidence</h5>'
                        '</div>'
                        '<div class="card-body">'
                        '<div class="row">'
                    )
                    
                    for period, level in confidence.items():
                        period_display = period.replace("_", " ").title()
                        confidence_class = "success" if level > 0.7 else "warning" if level > 0.5 else "danger"
                        html += (
                            f'<div class="col-md-3 mb-3">'
                            f'<div class="card">'
                            f'<div class="card-body text-center">'
                            f'<h5 class="card-title">{period_display}</h5>'
                            f'<div class="display-4 text-{confidence_class}">{int(level * 100)}%</div>'
                            f'</div>'
                            f'</div>'
                            f'</div>'
                        )
                    
                    html += (
                        '</div>'
                        '</div>'
                        '</div>'
                    )
            
            # Charts
            if charts:
                html += '<h4 class="mt-4 mb-3">Weather Forecasts</h4>'
                
                # Temperature chart
                if "temperature" in charts and charts["temperature"]:
                    html += (
                        f'<div class="card mb-4">'
                        f'<div class="card-header">'
                        f'<h5>Temperature Forecast</h5>'
                        f'</div>'
                        f'<div class="card-body text-center">'
                        f'<img src="data:image/png;base64,{charts["temperature"]}" class="img-fluid" alt="Temperature Chart">'
                        f'</div>'
                        f'</div>'
                    )
                
                # Precipitation chart
                if "precipitation" in charts and charts["precipitation"]:
                    html += (
                        f'<div class="card mb-4">'
                        f'<div class="card-header">'
                        f'<h5>Precipitation and Humidity Forecast</h5>'
                        f'</div>'
                        f'<div class="card-body text-center">'
                        f'<img src="data:image/png;base64,{charts["precipitation"]}" class="img-fluid" alt="Precipitation Chart">'
                        f'</div>'
                        f'</div>'
                    )
                
                # Wind chart
                if "wind" in charts and charts["wind"]:
                    html += (
                        f'<div class="card mb-4">'
                        f'<div class="card-header">'
                        f'<h5>Wind Forecast</h5>'
                        f'</div>'
                        f'<div class="card-body text-center">'
                        f'<img src="data:image/png;base64,{charts["wind"]}" class="img-fluid" alt="Wind Chart">'
                        f'</div>'
                        f'</div>'
                    )
                
                # Confidence chart
                if "confidence" in charts and charts["confidence"]:
                    html += (
                        f'<div class="card mb-4">'
                        f'<div class="card-header">'
                        f'<h5>Forecast Confidence</h5>'
                        f'</div>'
                        f'<div class="card-body text-center">'
                        f'<img src="data:image/png;base64,{charts["confidence"]}" class="img-fluid" alt="Confidence Chart">'
                        f'</div>'
                        f'</div>'
                    )
                
                # Source comparison charts
                html += '<h4 class="mt-4 mb-3">Source Comparisons</h4>'
                
                comparison_charts = [
                    ("comparison_temp", "Temperature Comparison"),
                    ("comparison_precip", "Precipitation Comparison"),
                    ("comparison_wind", "Wind Speed Comparison")
                ]
                
                for chart_key, chart_title in comparison_charts:
                    if chart_key in charts and charts[chart_key]:
                        html += (
                            f'<div class="card mb-4">'
                            f'<div class="card-header">'
                            f'<h5>{chart_title}</h5>'
                            f'</div>'
                            f'<div class="card-body text-center">'
                            f'<img src="data:image/png;base64,{charts[chart_key]}" class="img-fluid" alt="{chart_title}">'
                            f'</div>'
                            f'</div>'
                        )
            
            # Source accuracy
            accuracy = report.get("accuracy", {})
            if accuracy and "sources" in accuracy:
                html += (
                '<h4 class="mt-4 mb-3">Historical Source Accuracy</h4>'
                '<div class="card mb-4">'
                '    <div class="card-header bg-info text-white">'
                '        <h5>Source Accuracy Comparison</h5>'
                '    </div>'
                '    <div class="card-body">'
                '        <div class="table-responsive">'
                '            <table class="table table-bordered">'
                '                <thead class="table-light">'
                '                    <tr>'
                '                        <th>Source</th>'
                '                        <th>Temperature</th>'
                '                        <th>Precipitation</th>'
                '                        <th>Wind</th>'
                '                        <th>Overall</th>'
                '                    </tr>'
                '                </thead>'
                '                <tbody>'
                )
                
                for source, metrics in accuracy["sources"].items():
                    html += (
                    f'<tr>'
                    f'    <td><strong>{source}</strong></td>'
                    f'    <td>{int(metrics.get("temperature", 0) * 100)}%</td>'
                    f'    <td>{int(metrics.get("precipitation", 0) * 100)}%</td>'
                    f'    <td>{int(metrics.get("wind", 0) * 100)}%</td>'
                    f'    <td><strong>{int(metrics.get("overall", 0) * 100)}%</strong></td>'
                    f'</tr>'
                    )
                
                html += (
                '                </tbody>'
                '            </table>'
                '        </div>'
                '    </div>'
                '</div>'
                )
                
                # Timeframe accuracy
                if "timeframes" in accuracy:
                    html += (
                    '<div class="card mb-4">'
                    '    <div class="card-header bg-info text-white">'
                    '        <h5>Accuracy by Timeframe</h5>'
                    '    </div>'
                    '    <div class="card-body">'
                    '        <div class="table-responsive">'
                    '            <table class="table table-bordered">'
                    '                <thead class="table-light">'
                    '                    <tr>'
                    '                        <th>Timeframe</th>'
                    )
                    
                    sources = list(next(iter(accuracy["timeframes"].values())).keys())
                    for source in sources:
                        html += f'<th>{source}</th>'
                    
                    html += (
                    '                    </tr>'
                    '                </thead>'
                    '                <tbody>'
                    )
                    
                    for timeframe, sources_data in accuracy["timeframes"].items():
                        html += f'<tr><td><strong>{timeframe}</strong></td>'
                        
                        for source in sources:
                            accuracy_val = sources_data.get(source, 0)
                            html += f'<td>{int(accuracy_val * 100)}%</td>'
                        
                        html += '</tr>'
                    
                    html += (
                    '                </tbody>'
                    '            </table>'
                    '        </div>'
                    '    </div>'
                    '</div>'
                    )
            
            # Close the main card
            html += (
                '</div>'
                '</div>'
            )
            
            return html
        
        except Exception as e:
            return f'<div class="alert alert-danger">Error rendering report: {str(e)}</div>'
    
    @staticmethod
    def render_report_list(reports: list, location: Location) -> str:
        """Render a list of saved reports."""
        if not reports:
            return f'<div class="alert alert-info">No saved reports for {location.name}.</div>'
        
        html = (
            f'<h4>Saved Reports for {location.name}</h4>'
            f'<div class="list-group mb-4">'
        )
        
        for report in reports:
            try:
                date_str = datetime.fromisoformat(report["generated_at"]).strftime("%Y-%m-%d %H:%M")
                days = report.get("days", 0)
                file_path = report.get("file_path", "#")
                
                html += (
                    f'<a href="/report/view/{location.id}/{date_str}" class="list-group-item list-group-item-action">'
                    f'<div class="d-flex w-100 justify-content-between">'
                    f'<h5 class="mb-1">Report from {date_str}</h5>'
                    f'</div>'
                    f'<p class="mb-1">{days} day forecast</p>'
                    f'</a>'
                )
            except (KeyError, ValueError):
                # Skip invalid entries
                continue
                
        html += '</div>'
        return html
