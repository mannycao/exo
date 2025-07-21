# pipeline/enhanced_pipeline_runner.py

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
import config
from pathlib import Path
from datetime import datetime

from data.data_fetcher import get_mock_target_lists_and_data
from data.dataset_generator import create_dataset
from models.cnn_model import build_transit_detection_model
from models.model_trainer import train_enhanced_model
from pipeline.report_generator import generate_report

# --- THIS IS THE FIX ---
# Update the function to accept the 'args' from the command line
def run_enhanced_pipeline(config, args):
    """
    Main function to run the pipeline, handling config and command-line arguments.
    """
    logger = logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(config.RESULTS_DIR) / f"pipeline_run_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Pipeline run started. All results will be saved in: {result_dir}")

    # The pipeline now correctly receives 'args', though the mock data function
    # doesn't use it yet. This structure is ready for when you switch to real data.
    logger.info(f"Command-line arguments received: {args}")

    downloaded_files = get_mock_target_lists_and_data(config)
    if not downloaded_files:
        logger.error("Mock data generation failed. Aborting.")
        return

    processed_data_dir = result_dir / "processed_data"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    logger.info("Creating the dataset from MOCK FITS files...")
    create_dataset(
        file_paths=[item['file_path'] for item in downloaded_files],
        labels=[item['label'] for item in downloaded_files],
        output_dir=processed_data_dir
    )
    
    logger.info("Loading dataset for training...")
    X_path = processed_data_dir / 'X_data.npy'
    y_path = processed_data_dir / 'y_labels.npy'
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info("Building the model...")
    input_shape = X_train.shape[1:]
    model = build_transit_detection_model(input_shape)

    logger.info("Training the model...")
    
    model, history = train_enhanced_model(
        model=model,
        model_name="exo_cnn_mock_model",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=result_dir
    )
    
    if history:
        model_results_for_report = {
            'cnn_model': model,
            'cnn_history': history.history,
            'cnn_metrics': {k: v[-1] for k, v in history.history.items()}
        }
        processed_results_for_report = [{'file_path': item['file_path'], 'success': True, 'transit_count': 1} for item in downloaded_files]

        generate_report(
            results=processed_results_for_report,
            model_results=model_results_for_report,
            timestamp=timestamp,
            output_dir=str(result_dir)
        )

    logger.info(f"Enhanced pipeline finished successfully. Report generated in {result_dir}")