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
TRANSIT_SENSITIVITY = 2.0      # Lowered from 3.0 to find more subtle signals
PROMINENCE_FACTOR = 0.3        # Lowered from 0.5
MIN_TRANSIT_DURATION = 0.04    # Lowered from 0.05
MAX_TRANSIT_DURATION = 0.6     # Increased from 0.5

# --- Feature Extraction for Multimodal Model ---
WINDOW_SIZE = 1.0
IMAGE_SIZE = (64, 64) # Define the size for the 2D image representation

# --- File Type Constants ---
FILE_TYPE_CONFIRMED_PLANET = 'confirmed_planet'
FILE_TYPE_FALSE_POSITIVE = 'false_positive'

# --- OPTIMIZATION 2: Model Training Hyperparameters ---
EPOCHS = 75 # Increased epochs for more training time
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15

LEARNING_RATE = 1e-4 # A stable learning rate

# Focal loss parameters to prioritize hard-to-classify examples
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# --- Pipeline Execution Parameters ---
DEFAULT_MAX_WORKERS = os.cpu_count() if os.cpu_count() else 4
USE_MULTIMODAL = True
USE_CACHE = True
AUGMENTATION_FACTOR = 3

MIN_SNR = 3.0