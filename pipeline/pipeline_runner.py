# pipeline/pipeline_runner.py

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data.data_fetcher import download_light_curves, get_kepler_target_lists
from data.dataset_generator import create_dataset
from models.cnn_model import build_transit_detection_model
from models.model_trainer import train_enhanced_model

def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

def run_pipeline():
    """Main function to run the exoplanet detection pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting the exoplanet detection pipeline.")

    # Step 1: Fetch exoplanet target lists
    logger.info("Fetching exoplanet target lists...")
    confirmed_planets, false_positives = get_kepler_target_lists()

    if not confirmed_planets or not false_positives:
        logger.error("Failed to fetch exoplanet target lists. Aborting.")
        return

    # For development, let's limit the number of downloads
    confirmed_planets = confirmed_planets[:10]
    false_positives = false_positives[:10]

    # Step 2: Download light curves
    logger.info("Downloading light curves...")
    all_downloaded_files = []
    all_downloaded_files.extend(download_light_curves(confirmed_planets, 'confirmed'))
    all_downloaded_files.extend(download_light_curves(false_positives, 'false_positive'))

    if not all_downloaded_files:
        logger.error("No light curves were downloaded. Aborting.")
        return

    # Step 3: Create the dataset
    logger.info("Creating the dataset...")
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    create_dataset([item['file_path'] for item in all_downloaded_files],
                   [item['label'] for item in all_downloaded_files],
                   output_dir)

    # Step 4: Load dataset, build and train the model
    logger.info("Loading dataset and training the model...")
    X_path = os.path.join(output_dir, 'X_data.npy')
    y_path = os.path.join(output_dir, 'y_labels.npy')

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        logger.error("Dataset files (X_data.npy, y_labels.npy) not found. Aborting model training.")
        return

    X = np.load(X_path)
    y = np.load(y_path)

    # Split the data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if len(X_train) == 0:
        logger.error("No training data available after splitting. Aborting.")
        return
        
    # Build the CNN model
    input_shape = X_train.shape[1:]
    model = build_transit_detection_model(input_shape)

    # Train the model
    # Note: `train_enhanced_model` is used here as a replacement for the training part of the old function.
    history = train_enhanced_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_name='transit_detector_cnn',
        output_dir='models_trained'  # Specify where to save the trained model and history
    )
    
    logger.info("Model training complete.")
    logger.info("Pipeline finished successfully.")