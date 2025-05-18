"""
Configuration settings for the exoplanet detection pipeline.
"""

import os
import logging
from pathlib import Path

# Set up base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data_files")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models_trained")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
LIGHT_CURVE_DIR = os.path.join(DATA_DIR, "light_curves")

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, MODEL_DIR, METADATA_DIR, LIGHT_CURVE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline.log")

# Transit detection parameters
WINDOW_SIZE = 100  # Size of window for transit feature extraction
IMAGE_SIZE = (64, 64)  # Size of image representation for CNN
TRANSIT_SENSITIVITY = 3.0  # Number of standard deviations for transit detection
MIN_TRANSIT_DURATION = 0.1  # Minimum transit duration in days
MAX_TRANSIT_DURATION = 1.0  # Maximum transit duration in days

# Model training parameters
EPOCHS = 50
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data augmentation parameters
AUGMENTATION_FACTOR = 3
NOISE_LEVELS = [0.05, 0.1, 0.2]
TRANSIT_DEPTH_RANGE = (0.005, 0.05)

# Pipeline parameters
DEFAULT_MAX_WORKERS = 4
USE_MULTIMODAL = True
USE_CACHE = True
