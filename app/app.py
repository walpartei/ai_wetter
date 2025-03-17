#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

# Create necessary directories if they don't exist
for dir_path in ['templates', 'static', 'static/css', 'static/js']:
    os.makedirs(Path(__file__).parent / dir_path, exist_ok=True)

# Make sure all modules are imported properly
from app.ui import MainWindow
from app.utils.logging import get_logger

logger = get_logger()


def create_app():
    """Create the Flask application for production."""
    window = MainWindow()
    return window.app


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="AI Wetter - Bulgarian Weather Forecast Tool")
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    logger.info(f"Starting AI Wetter on {args.host}:{args.port} (debug: {args.debug})")
    
    # Create and run the main window
    window = MainWindow()
    window.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
