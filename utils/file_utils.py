# utils/file_utils.py

import os
import sys
import logging
from pathlib import Path
import config # Import config to get default format and level

logger = logging.getLogger(__name__)

# --- THIS IS THE FIX ---
# Update the function to accept 'log_file_path' directly
def setup_logging(log_file_path):
    """
    Sets up logging for the entire application, saving to a specific file path.
    """
    # Use settings from the config file for level and format
    log_level = getattr(config, 'LOG_LEVEL', logging.INFO)
    log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Ensure the log directory exists
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # This is crucial to override any existing logging configurations
    )
    
    # Quieten down noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("astropy").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Log file: {log_file_path}")




# You can include your other utility functions (save_pickle, etc.) below