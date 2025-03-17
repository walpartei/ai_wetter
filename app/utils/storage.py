import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from app.utils.config import HISTORY_DIR, REPORTS_DIR
from app.utils.logging import get_logger

logger = get_logger()


def save_forecast_history(location_id: str, forecasts: List[Dict[str, Any]]) -> bool:
    """Save forecast history to file.
    
    Args:
        location_id: The location ID
        forecasts: List of forecast dictionaries to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory for location if it doesn't exist
        location_dir = HISTORY_DIR / location_id
        location_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_{timestamp}.json"
        filepath = location_dir / filename
        
        # Save forecasts to file
        with open(filepath, "w") as f:
            json.dump(forecasts, f, indent=4)
            
        logger.info(f"Saved forecast history for {location_id} to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving forecast history: {e}")
        return False


def load_forecast_history(location_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Load forecast history for a location.
    
    Args:
        location_id: The location ID
        limit: Maximum number of historical forecasts to return
        
    Returns:
        List of historical forecast dictionaries
    """
    try:
        location_dir = HISTORY_DIR / location_id
        
        if not location_dir.exists():
            logger.info(f"No history found for location {location_id}")
            return []
        
        # Get all forecast files and sort by modification time (newest first)
        files = sorted(
            [f for f in location_dir.glob("forecast_*.json")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Load the most recent files up to the limit
        history = []
        for file in files[:limit]:
            with open(file, "r") as f:
                history.append({
                    "date": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "forecasts": json.load(f)
                })
        
        return history
    except Exception as e:
        logger.error(f"Error loading forecast history: {e}")
        return []


def save_report(location_id: str, report_data: Dict[str, Any]) -> Optional[str]:
    """Save a generated report to file.
    
    Args:
        location_id: The location ID
        report_data: The report data to save
        
    Returns:
        The path to the saved report file, or None if failed
    """
    try:
        # Create directory for location if it doesn't exist
        location_dir = REPORTS_DIR / location_id
        location_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.json"
        filepath = location_dir / filename
        
        # Save report to file
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=4)
            
        logger.info(f"Saved report for {location_id} to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return None


def load_reports(location_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Load reports for a location.
    
    Args:
        location_id: The location ID
        limit: Maximum number of reports to return
        
    Returns:
        List of report dictionaries
    """
    try:
        location_dir = REPORTS_DIR / location_id
        
        if not location_dir.exists():
            logger.info(f"No reports found for location {location_id}")
            return []
        
        # Get all report files and sort by modification time (newest first)
        files = sorted(
            [f for f in location_dir.glob("report_*.json")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Load the most recent files up to the limit
        reports = []
        for file in files[:limit]:
            with open(file, "r") as f:
                report_data = json.load(f)
                report_data["date"] = datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                report_data["file_path"] = str(file)
                reports.append(report_data)
        
        return reports
    except Exception as e:
        logger.error(f"Error loading reports: {e}")
        return []
