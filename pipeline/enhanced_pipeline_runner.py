# pipeline/enhanced_pipeline_runner.py

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
import config
from pathlib import Path
from datetime import datetime

# Your existing, correct imports
from data.data_fetcher import get_mock_target_lists_and_data
from data.dataset_generator import create_dataset
from models.cnn_model import build_transit_detection_model
from models.model_trainer import train_enhanced_model
# Import your existing report generator
from pipeline.report_generator import generate_report

def run_enhanced_pipeline(config):
    """
    Main function to run the pipeline with consolidated results and reporting.
    """
    logger = logging.getLogger(__name__)

    # --- Create a single, timestamped directory for all results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(config.RESULTS_DIR) / f"pipeline_run_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Pipeline run started. All results will be saved in: {result_dir}")

    # Generate mock data to ensure the pipeline runs
    downloaded_files = get_mock_target_lists_and_data(config)
    if not downloaded_files:
        logger.error("Mock data generation failed. Aborting.")
        return

    # --- Use the new result_dir for processed data ---
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
    
    # --- Pass the consolidated result_dir to the trainer ---
    # The 'model' and 'history' objects are now returned from the function
    model, history = train_enhanced_model(
        model=model,
        model_name="exo_cnn_mock_model",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=result_dir  # Instruct the trainer to save the model and plots here
    )
    
    # --- Final Step: Generate the HTML report using your existing function ---
    if history:
        # Create a dictionary of results to pass to your report generator
        model_results_for_report = {
            'cnn_model': model,
            'cnn_history': history.history,
            'cnn_metrics': {k: v[-1] for k, v in history.history.items()} # Get final metrics
        }
        # Assuming `results` should be a list of processed file info
        processed_results_for_report = [{'file_path': item['file_path'], 'success': True, 'transit_count': 1} for item in downloaded_files]

        generate_report(
            results=processed_results_for_report,
            model_results=model_results_for_report,
            timestamp=timestamp,
            output_dir=str(result_dir)
        )

    logger.info(f"Enhanced pipeline finished successfully. Report generated in {result_dir}")