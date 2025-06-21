# FILE: config.py (New, Complete, and Corrected Version)

import logging
from pathlib import Path
from tensorflow import keras

# --- Directory and File Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data_files"
LIGHT_CURVE_DIR = DATA_DIR / "light_curves"
METADATA_DIR = DATA_DIR / "metadata"
MODEL_DIR = RESULTS_DIR / "trained_models"

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'

# --- Data Processing & Feature Extraction ---
# Maximum expected transit duration in days. Used for setting the detrending window.
MAX_TRANSIT_DURATION = 1.0
# Window size for the Savitzky-Golay filter used in detrending.
DETRENDING_WINDOW_DAYS = 5.0 

# --- Transit Detection Parameters ---
# How many standard deviations below the mean a dip must be to be considered a potential transit.
TRANSIT_SENSITIVITY = 2.5
# The minimum number of consecutive data points (cadences) for a valid transit.
MIN_TRANSIT_DURATION_CADENCES = 2
# The maximum number of consecutive data points for a valid transit.
MAX_TRANSIT_DURATION_CADENCES = 50

# --- File Type Constants ---
FILE_TYPE_CONFIRMED_PLANET = 'confirmed_planet'
FILE_TYPE_FALSE_POSITIVE = 'false_positive'
FILE_TYPE_UNKNOWN = 'unknown'
FILE_TYPE_SYNTHETIC_PLANET = 'synthetic_planet'
FILE_TYPE_SYNTHETIC_NOISE = 'synthetic_noise'

# --- AI Model Training Hyperparameters ---
MAX_EPOCHS = 50
BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 1e-4

# Callbacks Configuration
EARLY_STOPPING_PATIENCE = 5
LR_REDUCTION_PATIENCE = 3
LR_REDUCTION_FACTOR = 0.5

# Model Metrics
MODEL_METRICS = [
    'accuracy',
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='roc_auc'),
    keras.metrics.AUC(name='pr_auc', curve='PR')
]

# --- Data Augmentation ---
AUGMENTATION_FACTOR = 2
