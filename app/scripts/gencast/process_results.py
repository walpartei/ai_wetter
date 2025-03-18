#!/usr/bin/env python3
"""
Process GenCast inference results and convert to Forecast objects.
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess
from datetime import datetime, timedelta
import numpy as np
import xarray
from pathlib import Path
import statistics

# Need to add the app directory to the path to import from app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from app.models.forecast import Forecast, WeatherMetrics


def download_zarr(bucket_name, output_id, local_path):
    """Download the zarr dataset from GCS."""
    print(f"Downloading results from gs://{bucket_name}/{output_id}.zarr")
    cmd = [
        "gsutil", "-m", "cp", 
        f"gs://{bucket_name}/{output_id}.zarr/**", 
        local_path
    ]
    subprocess.run(cmd, check=True)
    return f"{local_path}/{output_id}.zarr"


def process_predictions(predictions, location, source_name="GenCast"):
    """
    Process GenCast predictions and convert to Forecast objects.
    
    Args:
        predictions: xarray Dataset containing GenCast predictions
        location: Location object containing location information
        source_name: Name of the data source
        
    Returns:
        List of Forecast objects
    """
    print(f"Processing predictions for {location['name']}")
    
    # Extract ensemble members and time steps
    ensemble_size = predictions.dims.get("sample", 1)
    forecast_steps = predictions.dims.get("time", 0) - 1  # Exclude the initial condition
    
    # Create a location ID
    location_id = location.get("id") or location["name"].lower().replace(" ", "_")
    
    # Get coordinates for the location
    lat = location["latitude"]
    lon = location["longitude"]
    
    # Find the closest grid point
    lat_idx = abs(predictions.latitude.values - lat).argmin()
    lon_idx = abs(predictions.longitude.values - lon).argmin()
    
    print(f"Closest grid point: lat={predictions.latitude.values[lat_idx]}, lon={predictions.longitude.values[lon_idx]}")
    
    # Extract data for that grid point for each ensemble member
    # Start from the second time step (first is initial condition)
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Create forecasts for each day
    forecasts = []
    
    for step in range(1, forecast_steps + 1):
        forecast_date = start_date + timedelta(hours=12 * step)
        
        # Extract metrics for all ensemble members at this time step and grid point
        ensemble_metrics = []
        
        for sample in range(ensemble_size):
            # Extract data for this ensemble member
            sample_data = predictions.isel(sample=sample, time=step)
            
            # Extract variables at the grid point
            temp_2m = float(sample_data["2m_temperature"].isel(latitude=lat_idx, longitude=lon_idx).values)
            dewpoint_2m = float(sample_data["2m_dewpoint_temperature"].isel(latitude=lat_idx, longitude=lon_idx).values)
            mslp = float(sample_data["mean_sea_level_pressure"].isel(latitude=lat_idx, longitude=lon_idx).values)
            u10 = float(sample_data["10m_u_component_of_wind"].isel(latitude=lat_idx, longitude=lon_idx).values)
            v10 = float(sample_data["10m_v_component_of_wind"].isel(latitude=lat_idx, longitude=lon_idx).values)
            precip = float(sample_data["total_precipitation_12hr"].isel(latitude=lat_idx, longitude=lon_idx).values)
            
            # Temperature min/max estimation (simplified)
            temp_min = temp_2m - 2.0
            temp_max = temp_2m + 2.0
            
            # Calculate derived metrics
            wind_speed = np.sqrt(u10**2 + v10**2) * 3.6  # Convert m/s to km/h
            wind_direction = (270 - np.degrees(np.arctan2(v10, u10))) % 360  # Meteorological convention
            
            # Estimate humidity from dewpoint and temperature
            humidity = 100 * (np.exp((17.625 * dewpoint_2m) / (243.04 + dewpoint_2m)) / 
                              np.exp((17.625 * temp_2m) / (243.04 + temp_2m)))
            humidity = min(100, max(0, humidity))
            
            # Estimate cloud cover (very simplified)
            cloud_cover = min(100, max(0, humidity - 30 + np.random.normal(0, 10)))
            
            # Calculate soil metrics (simplified estimation)
            soil_temp = temp_2m - 3.0 + np.random.normal(0, 1.0)
            soil_moisture = min(100, max(0, humidity - 10 + np.random.normal(0, 5)))
            
            # Create metrics object for this ensemble member
            metrics = {
                "temperature_min": float(temp_min),
                "temperature_max": float(temp_max),
                "temperature_mean": float(temp_2m),
                "precipitation": float(precip),
                "humidity": float(humidity),
                "wind_speed": float(wind_speed),
                "wind_direction": float(wind_direction),
                "cloud_cover": float(cloud_cover),
                "soil_temperature": float(soil_temp),
                "soil_moisture": float(soil_moisture),
                "uv_index": None,
            }
            
            ensemble_metrics.append(metrics)
        
        # Calculate ensemble statistics and confidence levels
        # Extract metrics for easier processing
        temp_values = [m["temperature_mean"] for m in ensemble_metrics]
        precip_values = [m["precipitation"] for m in ensemble_metrics]
        wind_values = [m["wind_speed"] for m in ensemble_metrics]
        
        # Calculate standard deviation
        temp_std = statistics.stdev(temp_values) if len(temp_values) > 1 else 0
        precip_std = statistics.stdev(precip_values) if len(precip_values) > 1 else 0
        wind_std = statistics.stdev(wind_values) if len(wind_values) > 1 else 0
        
        # Calculate confidence based on ensemble spread
        temp_confidence = max(0.1, min(0.9, 1 - (temp_std / 10)))
        precip_confidence = max(0.1, min(0.9, 1 - (precip_std / 20)))
        wind_confidence = max(0.1, min(0.9, 1 - (wind_std / 15)))
        
        # Overall confidence
        overall_confidence = (temp_confidence * 0.4) + (precip_confidence * 0.4) + (wind_confidence * 0.2)
        
        # Use the first ensemble member's metrics, but with confidence from the ensemble
        metrics = ensemble_metrics[0]
        
        # Create a Forecast object with the ensemble data
        forecast = {
            "source": source_name,
            "location_id": location_id,
            "date": forecast_date.isoformat(),
            "metrics": metrics,
            "raw_data": {
                "source": source_name,
                "date": forecast_date.isoformat(),
                "location": {
                    "name": location["name"],
                    "lat": location["latitude"],
                    "lon": location["longitude"]
                },
                "metrics": metrics,
                "confidence": overall_confidence,
                "confidence_metrics": {
                    "temperature": temp_confidence,
                    "precipitation": precip_confidence,
                    "wind": wind_confidence,
                    "overall": overall_confidence,
                    "ensemble_size": ensemble_size
                },
                "ensemble_size": ensemble_size,
                "ensemble_data": ensemble_metrics[:10]  # Store up to 10 ensemble members
            },
            "retrieved_at": datetime.now().isoformat()
        }
        
        forecasts.append(forecast)
    
    print(f"Created {len(forecasts)} forecasts")
    return forecasts


def main():
    parser = argparse.ArgumentParser(description="Process GenCast inference results")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--output-id", required=True, help="Output ID")
    parser.add_argument("--location", required=True, help="Location data in JSON format")
    parser.add_argument("--output-file", required=True, help="Output file to write results")
    
    args = parser.parse_args()
    
    # Create temp directory for zarr data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the zarr dataset
        zarr_path = download_zarr(args.bucket, args.output_id, temp_dir)
        
        # Load the predictions
        predictions = xarray.open_zarr(zarr_path)
        
        # Parse location
        location = json.loads(args.location)
        
        # Process predictions
        forecasts = process_predictions(predictions, location)
        
        # Write results to output file
        with open(args.output_file, 'w') as f:
            json.dump(forecasts, f, indent=2)
        
        print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()