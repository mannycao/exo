# pipeline/enhanced_pipeline_runner.py

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
import config
from pathlib import Path

from data.data_fetcher import get_mock_target_lists_and_data
from data.dataset_generator import create_dataset
from models.cnn_model import build_transit_detection_model
from models.model_trainer import train_enhanced_model

def run_enhanced_pipeline(config):
    """
    Main function to run the enhanced exoplanet detection pipeline.
    Uses mock data and correctly calls all functions with all required arguments.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting the ENHANCED exoplanet detection pipeline.")

    downloaded_files = get_mock_target_lists_and_data(config)
    if not downloaded_files:
        logger.error("Mock data generation failed. Aborting.")
        return

    logger.info("Creating the dataset from MOCK FITS files...")
    
    output_dir = Path(config.DATA_DIR) / "processed"
    os.makedirs(output_dir, exist_ok=True)

    create_dataset(
        file_paths=[item['file_path'] for item in downloaded_files],
        labels=[item['label'] for item in downloaded_files],
        output_dir=output_dir
    )
    
    logger.info("Loading dataset for training...")
    X_path = output_dir / 'X_data.npy'
    y_path = output_dir / 'y_labels.npy'
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        logger.error(f"Dataset files not found in {output_dir}. Aborting.")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if len(X_train) == 0:
        logger.error("No training data available after splitting. Aborting.")
        return

    logger.info("Building the model...")
    input_shape = X_train.shape[1:]
    model = build_transit_detection_model(input_shape)

    logger.info("Training the model...")
    # --- THIS IS THE FIX ---
    # Call the training function with the required 'model_name' argument
    train_enhanced_model(
        model=model,
        model_name="exo_cnn_mock_model",  # Provide a name for the model
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    logger.info("Enhanced pipeline finished successfully using MOCK DATA.")