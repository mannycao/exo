# pipeline/enhanced_pipeline_runner.py

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
import config
from pathlib import Path

# These imports are correct and will now be used properly
from data.dataset_generator import create_dataset
from models.cnn_model import build_transit_detection_model
from models.model_trainer import train_enhanced_model
from pipeline.report_generator import generate_report

# --- THIS IS THE FIX ---
# Update the function to accept 'light_curve_files' from main.py
def run_enhanced_pipeline(light_curve_files, output_dir_str):
    """
    This is the core processing pipeline. It takes a list of light curve files,
    processes them, trains a model, and generates a report.
    """
    logger = logging.getLogger(__name__)

    # The main script now provides the output directory
    result_dir = Path(output_dir_str)
    logger.info(f"Enhanced pipeline runner received {len(light_curve_files)} files. Saving results to {result_dir}")

    # --- Use the new result_dir for processed data ---
    processed_data_dir = result_dir / "processed_data"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    logger.info("Creating the dataset from provided light curve files...")
    # The 'labels' are now passed correctly as 'type' from the main script
    create_dataset(
        file_paths=[item['file_path'] for item in light_curve_files],
        labels=[item['type'] for item in light_curve_files],
        output_dir=processed_data_dir
    )
    
    logger.info("Loading dataset for training...")
    X_path = processed_data_dir / 'X_data.npy'
    y_path = processed_data_dir / 'y_labels.npy'
    
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        logger.error("Dataset files (X_data.npy, y_labels.npy) were not created. Aborting.")
        return None

    X = np.load(X_path)
    y = np.load(y_path)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info("Building the model...")
    input_shape = X_train.shape[1:]
    model = build_transit_detection_model(input_shape)

    logger.info("Training the model...")
    
    model, history = train_enhanced_model(
        model=model,
        model_name="exo_cnn_model",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=result_dir
    )
    
    pipeline_results = {
        'result_dir_actual': str(result_dir),
        'successfully_processed_count': len(light_curve_files),
        'transit_count': -1 # Placeholder, as this isn't calculated yet
    }

    if history:
        model_results = {
            'cnn_model': model,
            'cnn_history': history.history,
            'cnn_metrics': {k: v[-1] for k, v in history.history.items()}
        }
        # The 'results' for the report can be a simple representation of what was processed
        report_results = [{'file_path': item['file_path'], 'success': True} for item in light_curve_files]

        report_path = generate_report(
            results=report_results,
            model_results=model_results,
            timestamp=result_dir.name.replace("run_", ""),
            output_dir=str(result_dir)
        )
        pipeline_results['report_path'] = str(report_path)


    logger.info(f"Enhanced pipeline finished successfully.")
    return pipeline_results