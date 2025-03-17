#!/usr/bin/env bash
# Build script for Render.com

# Exit on error
set -e

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Skip Playwright browser installation in build phase
# We'll modify the app to handle missing browser more gracefully
echo "Skipping Playwright browser installation (will be handled in the application)"