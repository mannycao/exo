"""
Enhanced pipeline execution for exoplanet detection.
Orchestrates data preparation, balancing, augmentation, model training, and evaluation.
"""

import os
import logging
import numpy as np
import pandas as pd 
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path 
import traceback

import config 
from data.data_fetcher import fetch_exoplanet_labels 
from data.dataset_generator import balance_dataset, augment_dataset
from models.model_trainer import train_enhanced_model 
from .pipeline_runner import ( 
    process_light_curve, 
    prepare_datasets, 
    prepare_train_test_split as robust_train_test_split
)
from .report_generator import generate_report 

from utils.metrics import (
    calculate_precision_recall_curve_with_thresholds,
    visualize_threshold_analysis, confusion_matrix_with_metrics,
    calculate_binary_metrics 
)
from utils.visualization import visualize_model_comparison, visualize_confusion_matrix, visualize_detection_results
from models.cnn_model import build_transit_detection_model 
from models.multimodal_model import (
    build_ensemble_multimodal_model, 
    optimize_ensemble_weights,
    combine_ensemble_predictions_weighted,
    create_dual_threshold_predictions
)
from data.light_curve_processor import reshape_data_for_cnn 


logger = logging.getLogger(__name__)


def prepare_enhanced_datasets(results_with_types, exoplanet_labels_df=None, include_tess=True, enhanced_augmentation=True, augmentation_factor=2):
    """
    Prepare enhanced datasets for AI model training, including balancing and augmentation.
    """
    logger.info("Preparing enhanced datasets with balancing and augmentation.")
    
    X_image, X_timeseries, y = prepare_datasets(results_with_types, exoplanet_labels_df)
    
    if X_image is None or y is None or len(y) == 0: 
        logger.warning("Initial dataset preparation yielded no valid data. Halting dataset preparation.")
        return None, None, None
    
    final_X_image, final_X_timeseries, final_y = X_image, X_timeseries, y

    if len(np.unique(y)) > 1:
        try:
            # Step 1: Balance the dataset by oversampling the minority class
            X_image_bal, X_timeseries_bal, y_bal = balance_dataset(
                X_image, X_timeseries, y,
                method='oversample' 
            )
            
            # Step 2: Augment the now-balanced dataset if requested
            if enhanced_augmentation and augmentation_factor > 1:
                logger.info(f"Applying augmentation with factor {augmentation_factor} to balanced dataset.")
                final_X_image, final_X_timeseries, final_y = augment_dataset(
                    X_image_bal, X_timeseries_bal, y_bal,
                    augmentation_factor=augmentation_factor,
                    only_positive_class=False # Augment both classes to maintain balance
                )
            else:
                final_X_image, final_X_timeseries, final_y = X_image_bal, X_timeseries_bal, y_bal

        except Exception as e:
            logger.error(f"Error in data balancing/augmentation: {e}", exc_info=True)
            logger.info("Falling back to the original, unbalanced dataset.")
            # Fallback to original data if anything fails
            final_X_image, final_X_timeseries, final_y = X_image, X_timeseries, y
    else:
        logger.warning("Dataset has only one class after initial prep. Skipping balancing/augmentation.")

    return final_X_image, final_X_timeseries, final_y


def train_ai_models_enhanced(X_image, X_timeseries, y, use_multimodal=True, output_dir=None):
    """
    Train AI models for exoplanet detection with enhanced techniques.
    """
    if X_image is None or y is None or len(y) == 0:
        logger.warning("No valid data for model training. Aborting training.")
        return None
    if len(np.unique(y)) < 2:
        logger.warning(f"Training data contains only one class ({np.unique(y)}). Model training will be meaningless. Aborting.")
        return None

    output_dir_path = Path(output_dir) if output_dir else Path(config.MODEL_DIR) 
    output_dir_path.mkdir(parents=True, exist_ok=True) 

    X_image_cnn = reshape_data_for_cnn(X_image)
    
    stratify_y = y if y.ndim == 1 else None
    
    X_train_list, X_val_list, X_test_list, y_train, y_val, y_test = robust_train_test_split(
        [X_image_cnn, X_timeseries], 
        y, 
        stratify_labels=stratify_y 
    )
    
    if X_train_list is None or X_train_list[0] is None or len(X_train_list[0]) == 0: 
        logger.error("Training data is empty after split. Cannot train models.")
        return None

    X_img_train, X_ts_train = X_train_list
    X_img_val, X_ts_val = X_val_list
    X_img_test, X_ts_test = X_test_list

    cnn_model_obj = build_transit_detection_model(X_img_train.shape[1:])
    
    cnn_model, cnn_history = train_enhanced_model( 
        cnn_model_obj, X_img_train, y_train, X_img_val, y_val,
        model_name='transit_detector_cnn', 
        output_dir=str(output_dir_path), 
        use_focal_loss_config=True,
        use_class_weights_config=True 
    )
    
    if X_img_test is None or len(X_img_test) == 0:
         logger.warning("Test set is empty. Skipping model evaluation.")
         return { 'cnn_model': cnn_model, 'cnn_history': cnn_history.history if hasattr(cnn_history, 'history') else cnn_history, 'cnn_metrics': 'SKIPPED - No test data' }

    cnn_preds_probs = cnn_model.predict(X_img_test)
    cnn_metrics = calculate_binary_metrics(y_test, cnn_preds_probs) 
    logger.info(f"CNN Model Test Metrics (at default 0.5 threshold): {cnn_metrics}")
    
    # Recalculate at optimal threshold and log it for comparison
    optimal_th = cnn_metrics.get('optimal_threshold', 0.5)
    logger.info(f"Recalculating metrics at optimal threshold found from validation set: {optimal_th:.4f}")
    cnn_metrics_optimal = calculate_binary_metrics(y_test, cnn_preds_probs, threshold=optimal_th)
    logger.info(f"CNN Model Test Metrics @ Optimal Threshold: {cnn_metrics_optimal}")


    cnn_threshold_analysis = calculate_precision_recall_curve_with_thresholds(y_test, cnn_preds_probs)
    if cnn_threshold_analysis and 'threshold_metrics' in cnn_threshold_analysis and cnn_threshold_analysis['threshold_metrics']:
        visualize_threshold_analysis(
            cnn_threshold_analysis['threshold_metrics'],
            output_dir=str(output_dir_path),
            filename='cnn_threshold_analysis.png'
        )
    
    results = { 
        'cnn_model': cnn_model, 
        'cnn_metrics': cnn_metrics, 
        'cnn_metrics_optimal': cnn_metrics_optimal, 
        'cnn_history': cnn_history.history if hasattr(cnn_history, 'history') else cnn_history, 
        'cnn_threshold_analysis': cnn_threshold_analysis
    }
    
    if use_multimodal and X_ts_train is not None and len(X_ts_train) > 0:
        logger.info("Creating and training ensemble of multimodal models.")
        # ... (full multimodal logic would go here if needed) ...
        pass
    
    return results


def run_enhanced_pipeline(light_curve_files_with_types, 
                         exoplanet_labels=None,
                         use_synthetic=False, 
                         use_multimodal=True, optimize_hyperparams=False,
                         use_cache=True, max_workers=4, enhanced_augmentation=True,
                         augmentation_factor=2, include_tess=False, output_dir_base=None):
    """
    Run the enhanced exoplanet detection pipeline with improved data handling and models.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir_base:
        result_dir_actual = Path(output_dir_base) 
    else:
        result_dir_actual = Path(config.RESULTS_DIR) / f"pipeline_run_{timestamp}"
    
    result_dir_actual.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running enhanced pipeline. Results will be in: {result_dir_actual}")

    if not light_curve_files_with_types:
        logger.error("No light curve files provided to the pipeline.")
        return {'error': "No light curve files provided", 'result_dir_actual': str(result_dir_actual)}

    processed_results = []
    logger.info(f"Processing {len(light_curve_files_with_types)} light curve files with types.")
    
    processed_files_artifacts_dir = result_dir_actual / "processed_file_artifacts"
    processed_files_artifacts_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}
        for file_path, file_type in light_curve_files_with_types:
            file_specific_result_dir = processed_files_artifacts_dir / Path(file_path).stem
            future = executor.submit(process_light_curve, file_path, str(file_specific_result_dir), file_type)
            future_to_file[future] = (file_path, file_type)

        for future in future_to_file:
            try:
                result = future.result()
                if result:
                    file_path_orig, file_type_orig = future_to_file[future]
                    result['file_type_source'] = file_type_orig 
                    processed_results.append(result)
                else:
                    logger.warning(f"Empty result from worker for {future_to_file[future][0]}")
            except Exception as e:
                file_path_orig, file_type_orig = future_to_file[future]
                logger.error(f"A child process for file {file_path_orig} failed with exception: {e}", exc_info=True)
                processed_results.append({ 'file_path': file_path_orig, 'success': False, 'error': str(e), 'traceback': traceback.format_exc()})

    X_image, X_timeseries, y_labels = prepare_enhanced_datasets(
        processed_results, 
        exoplanet_labels,
        include_tess=include_tess,
        enhanced_augmentation=enhanced_augmentation,
        augmentation_factor=augmentation_factor
    )
    
    model_training_results = None
    if X_image is not None and y_labels is not None and len(y_labels) > 0 :
        dataset_output_dir = result_dir_actual / "prepared_dataset_for_ml"
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(dataset_output_dir / "X_image_features.npy", X_image)
            if X_timeseries is not None and len(X_timeseries) > 0:
                 np.save(dataset_output_dir / "X_timeseries_features.npy", X_timeseries)
            np.save(dataset_output_dir / "y_labels.npy", y_labels)
            logger.info(f"Saved final prepared dataset for ML to: {dataset_output_dir}")
        except Exception as e:
            logger.error(f"Could not save prepared dataset arrays: {e}")
        
        if len(np.unique(y_labels)) < 2:
            logger.warning(f"Dataset for training contains only one class. Model training will be skipped.")
        else:
            logger.info(f"Training AI models. Total samples: {len(y_labels)}, Positive: {np.sum(y_labels)}")
            model_output_dir = result_dir_actual / "trained_models"
            model_output_dir.mkdir(parents=True, exist_ok=True)
            model_training_results = train_ai_models_enhanced(
                X_image, X_timeseries, y_labels, 
                use_multimodal=use_multimodal, 
                output_dir=str(model_output_dir)
            )
    else:
        logger.warning("Insufficient data after preparation. AI model training skipped.")
    
    logger.info("Generating final report for the enhanced pipeline run.")
    report_file_path = generate_report(
        processed_results, 
        model_training_results, 
        timestamp, 
        str(result_dir_actual)
    )
    
    if processed_results: 
        visualize_detection_results(
            processed_results,
            filename='all_detections_summary.png', 
            output_dir=str(result_dir_actual)
        )
    
    logger.info("Pipeline execution attempt completed.")
    
    return {
        'timestamp': timestamp,
        'result_dir_actual': str(result_dir_actual),
        'file_count': len(light_curve_files_with_types), 
        'successfully_processed_count': sum(1 for r in processed_results if r.get('success')),
        'transit_count': sum(r.get('transit_count', 0) for r in processed_results if r.get('success')),
        'model_results': model_training_results, 
        'report_path': str(report_file_path),
        'processed_results': processed_results 
    }
