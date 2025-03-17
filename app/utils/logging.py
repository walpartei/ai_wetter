import logging
import os
from datetime import datetime
from pathlib import Path

from app.utils.config import LOGS_DIR

# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
LOG_FILE = LOGS_DIR / "ai_wetter.log"

# Configure logger
logger = logging.getLogger("ai_wetter")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def get_logger():
    """Returns the configured logger."""
    return logger
