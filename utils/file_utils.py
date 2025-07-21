# utils/file_utils.py

import os
import sys
import logging
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(config):
    """
    Sets up logging for the entire application using settings from the config object.
    """
    log_level = getattr(config, 'LOG_LEVEL', logging.INFO)
    log_file = getattr(config, 'DEFAULT_LOG_FILE', 'results/pipeline_general.log')
    log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Ensure the log directory exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing basicConfig
    )
    
    # Quieten down noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("astropy").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Log file: {log_file}")


def ensure_directory(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    """
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def save_pickle(data, file_path):
    """
    Save data to a pickle file.
    """
    ensure_directory(os.path.dirname(file_path))
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Data saved to pickle file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {e}", exc_info=True)


def load_pickle(file_path):
    """
    Load data from a pickle file.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle file {file_path}: {e}", exc_info=True)
    return None

# ... (include other functions like save_json, load_json, etc., from your original file)