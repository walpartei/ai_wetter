#!/usr/bin/env bash
# Build script for Render.com

# Exit on error
set -e

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright with Chromium only to save space
python -m playwright install --with-deps chromium