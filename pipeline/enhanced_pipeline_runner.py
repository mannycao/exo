# pipeline/enhanced_pipeline_runner.py

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

import config
from data.dataset_generator import create_dataset, balance_dataset
from models.multimodal_model import build_multimodal_fusion_model
from models.model_trainer import train_enhanced_model
from pipeline.report_generator import generate_report

logger = logging.getLogger(__name__)

def run_enhanced_pipeline(light_curve_files, output_dir_str):
    """
    The core pipeline, now upgraded for multimodal data processing and training.
    """
    result_dir = Path(output_dir_str)
    logger.info(f"Enhanced multimodal pipeline runner started. Saving results to {result_dir}")

    processed_data_dir = result_dir / "processed_data"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    logger.info("Creating the multimodal dataset (time-series and images)...")
    create_dataset(
        file_paths=[item['file_path'] for item in light_curve_files],
        labels=[item['type'] for item in light_curve_files],
        output_dir=processed_data_dir,
        image_size=config.IMAGE_SIZE
    )
    
    logger.info("Loading multimodal dataset for training...")
    try:
        X_ts = np.load(processed_data_dir / 'X_timeseries.npy')
        X_img = np.load(processed_data_dir / 'X_images.npy')
        y = np.load(processed_data_dir / 'y_labels.npy')
    except FileNotFoundError:
        logger.error("Could not find multimodal dataset files. Aborting.")
        return None

    if len(np.unique(y)) < 2:
        logger.error(f"Dataset contains only one class. Cannot train model.")
        return {'error': 'Single class dataset'}

    logger.info("Balancing the multimodal dataset...")
    [X_ts, X_img], y = balance_dataset([X_ts, X_img], y)
    
    X_ts_train, X_ts_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
        X_ts, X_img, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Building the multimodal fusion model...")
    model = build_multimodal_fusion_model(
        image_shape=X_img_train.shape[1:],
        timeseries_shape=X_ts_train.shape[1:]
    )

    logger.info("Training the multimodal model...")
    
    # --- THIS IS THE FIX ---
    # The order of inputs now matches the model definition: image first, then time-series.
    model, history = train_enhanced_model(
        model=model,
        model_name="exo_multimodal_model",
        X_train=[X_img_train, X_ts_train], # Correct order
        y_train=y_train,
        X_val=[X_img_val, X_ts_val],     # Correct order
        y_val=y_val,
        output_dir=result_dir
    )
    # ^^^^^^^^^^^^^^^^^^^^^^^^^
    
    pipeline_results = {
        'result_dir_actual': str(result_dir),
        'successfully_processed_count': len(light_curve_files),
        'transit_count': -1
    }

    if history:
        model_results = {
            'cnn_model': model,
            'cnn_history': history.history,
            'cnn_metrics': {k: v[-1] for k, v in history.history.items()}
        }
        report_results = [{'file_path': item['file_path'], 'success': True} for item in light_curve_files]
        report_path = generate_report(
            results=report_results,
            model_results=model_results,
            timestamp=result_dir.name.replace("run_", ""),
            output_dir=str(result_dir)
        )
        pipeline_results['report_path'] = str(report_path)

    logger.info(f"Enhanced multimodal pipeline finished successfully.")
    return pipeline_results