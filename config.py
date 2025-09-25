# config.py

import logging
import os
from pathlib import Path

# --- Project Structure & Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_files"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models_trained"
METADATA_DIR = DATA_DIR / "metadata"
LIGHT_CURVE_DIR = DATA_DIR / "light_curves"

# --- Logging Configuration ---
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
DEFAULT_LOG_FILE = RESULTS_DIR / "pipeline_general.log"

# --- Data Fetching ---
EXOPLANET_ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP"

# --- OPTIMIZATION 1: More Sensitive Transit Detection ---
TRANSIT_SENSITIVITY = 2.0
PROMINENCE_FACTOR = 0.3
MIN_TRANSIT_DURATION = 0.04
MAX_TRANSIT_DURATION = 0.6
BLS_POWER_THRESHOLD = 0.1 # Added this line

# --- Feature Extraction for Multimodal Model ---
WINDOW_SIZE = 1.0
IMAGE_SIZE = (64, 64)

FIXED_LENGTH = 2048 # Define a fixed length for time-series segments
FEATURE_VECTOR_LENGTH = 512 # New parameter for feature vector length

# --- File Type Constants ---
FILE_TYPE_CONFIRMED_PLANET = 'confirmed_planet'
FILE_TYPE_FALSE_POSITIVE = 'false_positive'

# --- OPTIMIZATION 2: Model Training Hyperparameters (Updated with your results) ---
EPOCHS = 35                # From your optimization results
BATCH_SIZE = 32            # Increased for potential speedup
EARLY_STOPPING_PATIENCE = 10 # Kept as a sensible default

LEARNING_RATE = 0.01      # From your optimization results

# Focal loss parameters (can still be useful)
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.5

# --- Pipeline Execution Parameters ---
DEFAULT_MAX_WORKERS = os.cpu_count() if os.cpu_count() else 4
USE_MULTIMODAL = True
USE_CACHE = True
AUGMENTATION_FACTOR = 1 # Reduced for potential speedup

MIN_SNR = 3