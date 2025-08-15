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

# --- Enhanced Transit Detection and Processing ---
TRANSIT_SENSITIVITY = 1.5  # More sensitive detection
PROMINENCE_FACTOR = 0.25   # Lower to catch smaller transits
MIN_TRANSIT_DURATION = 0.03  # Catch shorter transits
MAX_TRANSIT_DURATION = 0.8   # Allow longer transits
TRANSIT_SNR_THRESHOLD = 3.0  # Minimum SNR for valid transit detection

# --- Enhanced Feature Extraction ---
WINDOW_SIZE = 2.0  # Larger window to capture more context
IMAGE_SIZE = (128, 128)  # Higher resolution images
FEATURE_EXTRACTION_PARAMS = {
    'rv_snr_threshold': 2.0,
    'rv_outlier_zscore': 3.0,
    'rv_min_periodogram_power': 0.1,
    'imaging_min_separation': 0.1,
    'imaging_max_separation': 10.0,
    'imaging_min_contrast': 0.1
}

# --- File Type Constants ---
FILE_TYPE_CONFIRMED_PLANET = 'confirmed_planet'
FILE_TYPE_FALSE_POSITIVE = 'false_positive'

# --- Enhanced Model Training Parameters ---
EPOCHS = 50                # More epochs for better convergence
BATCH_SIZE = 32           # Larger batch size for better gradient estimates
EARLY_STOPPING_PATIENCE = 15  # More patience for complex model

# Learning rate schedule
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00001
LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 5

# Regularization
DROPOUT_RATE = 0.3
L2_LAMBDA = 0.001

# Attention Parameters
NUM_ATTENTION_HEADS = 8
ATTENTION_DROPOUT = 0.1
ATTENTION_TEMPERATURE = 0.1

# Bayesian Parameters
NUM_MC_SAMPLES = 50
PRIOR_SIGMA = 1.0
KL_WEIGHT = 1.0 / 50000  # Adjust based on dataset size

# Focal loss parameters (can still be useful)
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.5

# --- Pipeline Execution Parameters ---
DEFAULT_MAX_WORKERS = os.cpu_count() if os.cpu_count() else 4
USE_MULTIMODAL = True
USE_CACHE = True
AUGMENTATION_FACTOR = 6 # You can experiment with increasing this for more data

MIN_SNR = 3
