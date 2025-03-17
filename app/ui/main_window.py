import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session

from app.models import Location
from app.services import ForecastService, HistoryService, ReportService
from app.ui.location_selector import LocationSelector
from app.ui.forecast_view import ForecastView
from app.ui.history_view import HistoryView
from app.ui.report_view import ReportView
from app.utils.config import Config
from app.utils.logging import get_logger
from app.utils.auth import login_required, get_site_password, is_authenticated

logger = get_logger()


class MainWindow:
    """Main application window and controller."""
    
    def __init__(self):
        self.app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"),
                         static_folder=str(Path(__file__).parent.parent / "static"))
        self.config = Config()
        
        # Initialize services
        self.forecast_service = ForecastService()
        self.history_service = HistoryService()
        self.report_service = ReportService()
        
        # Initialize UI components
        self.location_selector = LocationSelector()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login page."""
            if request.method == 'POST':
                password = request.form.get('password')
                if password == get_site_password():
                    session['authenticated'] = True
                    next_url = request.args.get('next', url_for('index'))
                    flash('Login successful', 'success')
                    return redirect(next_url)
                else:
                    flash('Invalid password', 'danger')
            
            return render_template('login.html')
            
        @self.app.route('/logout')
        def logout():
            """Logout the user."""
            session.pop('authenticated', None)
            flash('You have been logged out', 'info')
            return redirect(url_for('login'))
        
        @self.app.route('/')
        @login_required
        def index():
            """Home page."""
            locations = self.location_selector.get_locations()
            default_location_id = self.config.get_app_setting("default_location", "devnya" if locations else None)
            location_dropdown = self.location_selector.render_dropdown(default_location_id)
            
            return render_template('index.html', 
                                   location_dropdown=location_dropdown,
                                   default_location=default_location_id,
                                   available_sources=self.forecast_service.get_available_sources())
        
        @self.app.route('/forecast', methods=['GET'])
        @login_required
        def get_forecast():
            """Get forecast for a location."""
            location_id = request.args.get('location', '')
            days = int(request.args.get('days', 14))
            
            location = self.location_selector.get_location_by_id(location_id)
            if not location:
                flash('Invalid location selected', 'error')
                return redirect(url_for('index'))
            
            # Get combined forecasts
            combined_forecasts = self.forecast_service.get_combined_forecasts(location, days)
            
            # Generate forecast table HTML
            forecast_table = ForecastView.render_forecast_table(combined_forecasts)
            
            # Generate sources comparison table
            sources_table = ForecastView.render_sources_table(combined_forecasts)
            
            # Generate charts
            charts = ForecastView.render_forecast_charts(combined_forecasts)
            
            return render_template('forecast.html',
                                   location=location,
                                   days=days,
                                   forecast_table=forecast_table,
                                   sources_table=sources_table,
                                   temperature_chart=charts.get('temperature', ''),
                                   precipitation_chart=charts.get('precipitation', ''),
                                   wind_chart=charts.get('wind', ''),
                                   confidence_chart=charts.get('confidence', ''),
                                   comparison_temp=charts.get('comparison_temp', ''),
                                   comparison_precip=charts.get('comparison_precip', ''),
                                   comparison_wind=charts.get('comparison_wind', ''))
        
        @self.app.route('/history', methods=['GET'])
        @login_required
        def get_history():
            """View history for a location."""
            location_id = request.args.get('location', '')
            
            location = self.location_selector.get_location_by_id(location_id)
            if not location:
                flash('Invalid location selected', 'error')
                return redirect(url_for('index'))
                
            # Get forecast history
            history = self.history_service.get_forecast_history(location)
            
            # Render history list
            history_list = HistoryView.render_history_list(history, location)
            
            return render_template('history.html',
                                   location=location,
                                   history_list=history_list,
                                   history_data=json.dumps(history))
        
        @self.app.route('/report/generate', methods=['GET'])
        @login_required
        def generate_report():
            """Generate a new report for a location."""
            location_id = request.args.get('location', '')
            days = int(request.args.get('days', 14))
            
            location = self.location_selector.get_location_by_id(location_id)
            if not location:
                flash('Invalid location selected', 'error')
                return redirect(url_for('index'))
                
            # Generate the report
            report = self.report_service.generate_report(location, days)
            
            # Get combined forecasts for charts
            combined_forecasts = self.forecast_service.get_combined_forecasts(location, days)
            
            # Generate charts
            charts = ForecastView.render_forecast_charts(combined_forecasts)
            
            # Render report view
            report_html = ReportView.render_report(report, charts)
            
            return render_template('report.html',
                                   location=location,
                                   report_html=report_html)
        
        @self.app.route('/report/list', methods=['GET'])
        @login_required
        def list_reports():
            """List saved reports for a location."""
            location_id = request.args.get('location', '')
            
            location = self.location_selector.get_location_by_id(location_id)
            if not location:
                flash('Invalid location selected', 'error')
                return redirect(url_for('index'))
                
            # Get saved reports
            reports = self.report_service.get_saved_reports(location)
            
            # Render report list
            report_list = ReportView.render_report_list(reports, location)
            
            return render_template('report_list.html',
                                   location=location,
                                   report_list=report_list)
        
        @self.app.route('/api/history/details', methods=['GET'])
        @login_required
        def get_history_details():
            """API endpoint to get historical forecast details."""
            location_id = request.args.get('location', '')
            date_str = request.args.get('date', '')
            
            location = self.location_selector.get_location_by_id(location_id)
            if not location:
                return jsonify({'error': 'Invalid location'}), 400
                
            # Get forecast history
            history = self.history_service.get_forecast_history(location)
            
            # Find the requested history entry
            history_entry = next((h for h in history if h["date"].startswith(date_str)), None)
            
            if not history_entry:
                return jsonify({'error': 'History entry not found'}), 404
                
            # Render history details
            history_html = HistoryView.render_history_details(history_entry, location)
            
            return jsonify({'html': history_html})
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the Flask application."""
        self.app.secret_key = 'ai_wetter_secret_key'  # For flash messages
        self.app.run(host=host, port=port, debug=debug)
