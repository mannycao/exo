# pipeline/enhanced_pipeline_runner.py

import logging
import os
import numpy as np
import pandas as pd # Added import
from sklearn.model_selection import train_test_split
from pathlib import Path

import config
from data.dataset_generator import create_dataset
from data.augmentation_utils import augment_data
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
    
    # Load exoplanet metadata for period information
    # Assuming exoplanet_labels.csv is in data/metadata relative to project root
    project_root = Path(__file__).resolve().parents[1] # Go up two levels from pipeline/ to project root
    metadata_path = project_root / "data" / "metadata" / "exoplanet_labels.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}. Aborting.")
        return None
    exoplanet_metadata_df = pd.read_csv(metadata_path)

    create_dataset(
        file_paths=[item['file_path'] for item in light_curve_files],
        labels=[item['type'] for item in light_curve_files],
        output_dir=processed_data_dir,
        metadata_df=exoplanet_metadata_df,
        image_size=config.IMAGE_SIZE
    )
    
    logger.info("Loading multimodal dataset for training...")
    try:
        X_ts = np.load(processed_data_dir / 'X_timeseries.npy')
        X_img = np.load(processed_data_dir / 'X_images.npy')
        X_features = np.load(processed_data_dir / 'X_features.npy')
        y = np.load(processed_data_dir / 'y_labels.npy')
    except FileNotFoundError:
        logger.error("Could not find multimodal dataset files. Aborting.")
        return None

    if len(np.unique(y)) < 2:
        logger.error(f"Dataset contains only one class. Cannot train model.")
        return {'error': 'Single class dataset'}

    
    
    X_ts_train, X_ts_val, X_img_train, X_img_val, X_features_train, X_features_val, y_train, y_val = train_test_split(
        X_ts, X_img, X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Building the multimodal fusion model...")
    model = build_multimodal_fusion_model(
        image_shape=X_img_train.shape[1:],
        timeseries_shape=X_ts_train.shape[1:],
        feature_shape=X_features_train.shape[1:]
    )

    logger.info("Training the multimodal model...")
    
    # --- THIS IS THE FIX ---
    # The order of inputs now matches the model definition: image, time-series, and features.
    model, history = train_enhanced_model(
        model=model,
        model_name="exo_multimodal_model",
        X_train=[X_img_train, X_ts_train, X_features_train], # Correct order
        y_train=y_train,
        X_val=[X_img_val, X_ts_val, X_features_val],     # Correct order
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
            'cnn_model': model, # This is actually the multimodal model
            'cnn_history': history.history,
            'cnn_metrics': {k: v[-1] for k, v in history.history.items()}
        }
        # Add custom metrics to the model_results for reporting
        model_results['cnn_metrics']['val_precision_custom'] = history.history['val_precision_custom'][-1]
        model_results['cnn_metrics']['val_recall_custom'] = history.history['val_recall_custom'][-1]
        model_results['cnn_metrics']['val_f1_custom'] = history.history['val_f1_custom'][-1]

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
