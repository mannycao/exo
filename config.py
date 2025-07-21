# config.py

"""
Configuration settings for the Exoplanet Detection Pipeline.
Review and adjust these parameters as needed for your specific datasets and goals.
"""

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
MAST_DOWNLOAD_MAX_RECORDS_OBS_TABLE = 200
MAST_DOWNLOAD_MAX_FILES_PER_RUN = 50

# --- Synthetic Data Generation ---
SYNTHETIC_TRANSIT_PROBABILITY = 0.5
SYNTHETIC_TIME_STEPS = 2000
SYNTHETIC_OBSERVATION_DURATION_DAYS = 100.0
SYNTHETIC_NOISE_LEVEL_MIN = 0.0005
SYNTHETIC_NOISE_LEVEL_MAX = 0.002
SYNTHETIC_VAR_PROB = 0.7
SYNTHETIC_VAR_PERIOD_MIN_DAYS = SYNTHETIC_OBSERVATION_DURATION_DAYS / 10
SYNTHETIC_VAR_PERIOD_MAX_DAYS = SYNTHETIC_OBSERVATION_DURATION_DAYS / 2
SYNTHETIC_VAR_AMPLITUDE_MIN = 0.001
SYNTHETIC_VAR_AMPLITUDE_MAX = 0.005
SYNTHETIC_TRANSIT_PERIOD_MIN_DAYS = 1.0
SYNTHETIC_TRANSIT_PERIOD_MAX_DAYS = SYNTHETIC_OBSERVATION_DURATION_DAYS / 3
SYNTHETIC_TRANSIT_DURATION_MIN_HOURS = 1.0
SYNTHETIC_TRANSIT_DURATION_MAX_HOURS = 6.0
SYNTHETIC_TRANSIT_DEPTH_MIN = 0.0005
SYNTHETIC_TRANSIT_DEPTH_MAX = 0.01

# --- Light Curve Preprocessing ---
DEFAULT_STELLAR_RADIUS_SOL = 1.0
DEFAULT_STELLAR_MASS_SOL = 1.0
DEFAULT_STELLAR_TEFF_K = 5778.0
DETRENDING_WINDOW_DAYS = 2.0
SAVGOL_POLYORDER = 2

# --- Transit Detection ---
TRANSIT_SENSITIVITY = 3.0
PROMINENCE_FACTOR = 0.5
MIN_TRANSIT_DURATION = 0.05
MAX_TRANSIT_DURATION = 0.5

# --- Feature Extraction ---
WINDOW_SIZE = 1.0
IMAGE_SIZE = (64, 64)

# --- File Type Constants ---
FILE_TYPE_CONFIRMED_PLANET = 'confirmed_planet'
FILE_TYPE_FALSE_POSITIVE = 'false_positive'
FILE_TYPE_UNKNOWN = 'unknown'
FILE_TYPE_SYNTHETIC_PLANET = 'synthetic_planet'
FILE_TYPE_SYNTHETIC_NOISE = 'synthetic_noise'

# --- Model Training Parameters ---
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15

LEARNING_RATE = 1e-4 # Keeping the safer learning rate

# VVVVVVVVVVVVVVVVVVVVVV MODIFICATION VVVVVVVVVVVVVVVVVVVVVV
# Since the model is predicting all negatives, let's heavily penalize it
# for missing positives by increasing the alpha weight for the positive class.
FOCAL_LOSS_ALPHA = 0.75 # Changed from 0.5 to 0.75
# ^^^^^^^^^^^^^^^^^^^^^^ MODIFICATION ^^^^^^^^^^^^^^^^^^^^^^

FOCAL_LOSS_GAMMA = 2.0

# --- Pipeline Execution Parameters ---
DEFAULT_MAX_WORKERS = os.cpu_count() if os.cpu_count() else 4
USE_MULTIMODAL = True
USE_CACHE = True
AUGMENTATION_FACTOR = 2 # Keeping augmentation factor at 2 for now