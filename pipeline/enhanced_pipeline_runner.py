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
    
import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import time # Added import
import json # Added import

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
    
    project_root = Path(__file__).resolve().parents[1]
    metadata_path = os.path.join(str(project_root), "data", "metadata", "exoplanet_labels.csv")
    
    if not Path(metadata_path).exists():
        logger.error(f"Metadata file not found: {metadata_path}. Aborting.")
        return None
    exoplanet_metadata_df = pd.read_csv(metadata_path)

    X_ts_raw, X_img_raw, X_features_raw, y_raw, all_pipeline_results_raw = create_dataset(
        file_paths=[item['file_path'] for item in light_curve_files],
        labels=[item['type'] for item in light_curve_files],
        output_dir=processed_data_dir,
        metadata_df=exoplanet_metadata_df,
        image_size=config.IMAGE_SIZE
    )
    
    logger.info("Loading multimodal dataset for training...")
    if X_ts_raw is None or X_img_raw is None or X_features_raw is None or y_raw is None:
        logger.error("Could not create multimodal dataset files. Aborting.")
        return None

    if len(np.unique(y_raw)) < 2:
        logger.error(f"Dataset contains only one class. Cannot train model.")
        return {'error': 'Single class dataset'}

    # Augment data
    augmentation_start_time = time.time()
    X_img_aug, X_ts_aug, X_features_aug, y_aug, all_pipeline_results_aug = augment_data(
        [X_img_raw, X_ts_raw, X_features_raw], y_raw, all_pipeline_results_raw,
        augmentation_factor=config.AUGMENTATION_FACTOR
    )
    augmentation_time = time.time() - augmentation_start_time
    logger.info(f"Data augmentation completed in {augmentation_time:.2f} seconds. Total samples: {len(y_aug)}")

    # Save augmented data
    save_start_time = time.time()
    np.save(os.path.join(processed_data_dir, 'X_timeseries.npy'), X_ts_aug)
    np.save(os.path.join(processed_data_dir, 'X_images.npy'), X_img_aug)
    np.save(os.path.join(processed_data_dir, 'X_features.npy'), X_features_aug)
    np.save(os.path.join(processed_data_dir, 'y_labels.npy'), y_aug)
    # Save augmented pipeline results
    with open(os.path.join(processed_data_dir, 'all_pipeline_results.json'), 'w') as f:
        json.dump(all_pipeline_results_aug, f, indent=2)
    save_time = time.time() - save_start_time
    logger.info(f"Augmented data saving completed in {save_time:.2f} seconds.")


    # Split data into training and validation sets, including all_pipeline_results
    X_ts_train, X_ts_val, \
    X_img_train, X_img_val, \
    X_features_train, X_features_val, \
    y_train, y_val, \
    all_pipeline_results_train, all_pipeline_results_val = train_test_split(
        X_ts_aug, X_img_aug, X_features_aug, y_aug, all_pipeline_results_aug, # Use augmented data
        test_size=0.2, random_state=42, stratify=y_aug
    )

    logger.info("Building the multimodal fusion model...")
    model = build_multimodal_fusion_model(
        image_shape=X_img_train.shape[1:],
        timeseries_shape=X_ts_train.shape[1:],
        feature_shape=X_features_train.shape[1:]
    )

    logger.info("Training the multimodal model...")
    
    model, history = train_enhanced_model(
        model=model,
        model_name="exo_multimodal_model",
        X_train=[X_img_train, X_ts_train, X_features_train],
        y_train=y_train,
        X_val=[X_img_val, X_ts_val, X_features_val],
        y_val=y_val,
        output_dir=result_dir
    )
    
    # Make predictions on the validation set
    y_pred_val = model.predict([X_img_val, X_ts_val, X_features_val])
    y_pred_val_classes = (y_pred_val > 0.5).astype(int) # Assuming binary classification

    pipeline_results = {
        'result_dir_actual': str(result_dir),
        'successfully_processed_count': len(light_curve_files),
        'transit_count': -1 # This will be updated in report_generator
    }

    if history:
        model_results = {
            'cnn_model': model,
            'cnn_history': history.history,
            'cnn_metrics': {k: v[-1] for k, v in history.history.items()}
        }
        # Add custom metrics to the model_results for reporting
        model_results['cnn_metrics']['val_precision_custom'] = history.history['val_precision_custom'][-1]
        model_results['cnn_metrics']['val_recall_custom'] = history.history['val_recall_custom'][-1]
        model_results['cnn_metrics']['val_f1_custom'] = history.history['val_f1_custom'][-1]

        report_path = generate_report(
            results=all_pipeline_results_val, # Pass validation results
            model_results=model_results,
            y_true=y_val,
            y_pred=y_pred_val_classes,
            timestamp=result_dir.name.replace("run_", ""),
            output_dir=str(result_dir)
        )
        pipeline_results['report_path'] = str(report_path)

    logger.info(f"Enhanced multimodal pipeline finished successfully.")
    return pipeline_results

    X_ts, X_img, X_features, y, all_pipeline_results = create_dataset(
        file_paths=[item['file_path'] for item in light_curve_files],
        labels=[item['type'] for item in light_curve_files],
        output_dir=processed_data_dir,
        metadata_df=exoplanet_metadata_df,
        image_size=config.IMAGE_SIZE
    )
    
    logger.info("Loading multimodal dataset for training...")
    if X_ts is None or X_img is None or X_features is None or y is None:
        logger.error("Could not create multimodal dataset files. Aborting.")
        return None

    if len(np.unique(y)) < 2:
        logger.error(f"Dataset contains only one class. Cannot train model.")
        return {'error': 'Single class dataset'}

    
    
    X_ts_train, X_ts_val, \
    X_img_train, X_img_val, \
    X_features_train, X_features_val, \
    y_train, y_val, \
    all_pipeline_results_train, all_pipeline_results_val = train_test_split(
        X_ts, X_img, X_features, y, all_pipeline_results,
        test_size=0.2, random_state=42, stratify=y
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
    
    # Make predictions on the validation set
    y_pred_val = model.predict([X_img_val, X_ts_val, X_features_val])
    y_pred_val_classes = (y_pred_val > 0.5).astype(int) # Assuming binary classification

    pipeline_results = {
        'result_dir_actual': str(result_dir),
        'successfully_processed_count': len(light_curve_files),
        'transit_count': -1 # This will be updated in report_generator
    }

    if history:
        model_results = {
            'cnn_model': model,
            'cnn_history': history.history,
            'cnn_metrics': {k: v[-1] for k, v in history.history.items()}
        }
        # Add custom metrics to the model_results for reporting
        model_results['cnn_metrics']['val_precision_custom'] = history.history['val_precision_custom'][-1]
        model_results['cnn_metrics']['val_recall_custom'] = history.history['val_recall_custom'][-1]
        model_results['cnn_metrics']['val_f1_custom'] = history.history['val_f1_custom'][-1]

        report_path = generate_report(
            results=all_pipeline_results_val, # Pass validation results
            model_results=model_results,
            y_true=y_val,
            y_pred=y_pred_val_classes,
            timestamp=result_dir.name.replace("run_", ""),
            output_dir=str(result_dir)
        )
        pipeline_results['report_path'] = str(report_path)

    logger.info(f"Enhanced multimodal pipeline finished successfully.")
    return pipeline_results
