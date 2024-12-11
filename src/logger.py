import logging
import os
from datetime import datetime

# Create log file name with timestamp
LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Create logs directory path
LOGS_DIR = os.path.join(os.getcwd(), 'logs')

# Ensure the logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Add a console logger for better debugging during development
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
logging.getLogger().addHandler(console_handler)

