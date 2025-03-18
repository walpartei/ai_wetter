#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
gunicorn "app.app:create_app()" --bind=127.0.0.1:8000 --workers=2 --timeout=300