"""
Main pipeline execution for exoplanet detection.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import config
from data.data_fetcher import download_light_curves, fetch_exoplanet_labels
from data.light_curve_processor import (
    preprocess_light_curve, detect_transits, extract_transit_features, 
    create_image_representations, reshape_data_for_cnn
)
from detection.periodicity_analyzer import analyze_periodicity
from detection.transit_detector import (
    apply_transit_modeling, estimate_planet_properties
)
from models.cnn_model import (
    build_transit_detection_model, build_multimodal_model
)
from models.model_trainer import (
    train_model, evaluate_model, prepare_train_test_split
)
from utils.visualization import (
    visualize_transit, visualize_folded_transit, visualize_periodogram,
    visualize_transit_model, visualize_detection_results
)
from pipeline.report_generator import generate_report

logger = logging.getLogger(__name__)

def setup_logging(log_level=None):
    """
    Set up logging for the exoplanet detection pipeline.
    
    Args:
        log_level: Logging level (default: from config)
    """
    import logging
    import os
    import config
    
    # Use specified log level or default from config
    if log_level is None:
        log_level = config.LOG_LEVEL
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Create logger specifically for the pipeline
    logger = logging.getLogger("exoplanet_detection")
    logger.setLevel(log_level)
    
    logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    logger.info(f"Log file: {config.LOG_FILE}")
    
def process_light_curve(file_path, result_dir=None):
    """
    Process a single light curve file.
    
    Args:
        file_path: Path to the light curve file
        result_dir: Directory to save results (optional)
    
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Processing {file_path}")
        
        # Create result directory if needed
        if result_dir is None:
            result_dir = os.path.join(
                config.RESULTS_DIR, 
                os.path.basename(file_path).split('.')[0]
            )
        os.makedirs(result_dir, exist_ok=True)
        
        # Step 1: Preprocess the light curve
        time, flux, metadata = preprocess_light_curve(file_path)
        if time is None or flux is None:
            logger.warning(f"Failed to preprocess {file_path}")
            return {
                'file_path': file_path,
                'success': False,
                'error': "Failed to preprocess light curve"
            }
        
        # Step 2: Transit detection
        transit_info = detect_transits(time, flux)
        if transit_info is None:
            logger.info(f"No transits detected in {file_path}")
            # Save basic visualization
            visualize_transit(
                time, flux, 
                filename='light_curve.png',
                output_dir=result_dir
            )
            return {
                'file_path': file_path,
                'metadata': metadata,
                'transit_count': 0,
                'success': True,
                'result_dir': result_dir
            }
        
        # Step 3: Periodicity analysis
        periodicity_info = analyze_periodicity(time, flux, transit_info)
        
        # Step 4: Transit modeling
        transit_model = None
        if periodicity_info is not None:
            transit_model = apply_transit_modeling(time, flux, transit_info, periodicity_info)
        
        # Step 5: Estimate planet properties
        planet_properties = None
        if periodicity_info is not None:
            planet_properties = estimate_planet_properties(transit_info, periodicity_info)
        
        # Step 6: Feature extraction for AI models
        transit_segments = extract_transit_features(time, flux, transit_info)
        transit_images = None
        if transit_segments is not None:
            transit_images = create_image_representations(transit_segments)
        
        # Step 7: Visualizations
        visualize_transit(
            time, flux, transit_info, 
            filename='light_curve.png',
            output_dir=result_dir
        )
        
        if periodicity_info is not None:
            periodogram = periodicity_info.get('periodogram')
            if periodogram:
                visualize_periodogram(
                    periodogram['period'], periodogram['power'],
                    periodogram['peak_periods'],
                    filename='periodogram.png',
                    output_dir=result_dir
                )
        
        if transit_model is not None:
            visualize_transit_model(
                time, flux, transit_model['model_flux'],
                filename='transit_model.png',
                output_dir=result_dir
            )
        
        # Step 8: Save folded light curve if periodicity detected
        if periodicity_info is not None and periodicity_info.get('median_period'):
            from detection.periodicity_analyzer import (
                calculate_folded_lightcurve, bin_folded_lightcurve
            )
            phase, folded_flux = calculate_folded_lightcurve(
                time, flux, periodicity_info['median_period']
            )
            if phase is not None and folded_flux is not None:
                bin_centers, binned_flux, binned_error = bin_folded_lightcurve(phase, folded_flux)
                visualize_folded_transit(
                    phase, folded_flux, bin_centers, binned_flux, binned_error,
                    periodicity_info['median_period'],
                    filename='folded_light_curve.png',
                    output_dir=result_dir
                )
        
        # Step 9: Return the results
        expected_files = ['light_curve.png']
        if periodicity_info is not None:
            expected_files.append('periodogram.png')
        if transit_model is not None:
            expected_files.append('transit_model.png')
        
        for expected_file in expected_files:
            file_path_check = os.path.join(result_dir, expected_file)
            if not os.path.exists(file_path_check):
                logger.warning(f"Expected file not created: {file_path_check}")
        return {
            'file_path': file_path,
            'metadata': metadata,
            'transit_count': len(transit_info['peak_indices']),
            'periodicity': periodicity_info['median_period'] if periodicity_info else None,
            'planet_properties': planet_properties,
            'transit_model': transit_model is not None,
            'has_transit_images': transit_images is not None,
            'transit_images': transit_images,
            'transit_segments': transit_segments,
            'result_dir': result_dir,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        return {
            'file_path': file_path,
            'success': False,
            'error': str(e)
        }

def validate_transit_detections(results):
    """
    Performs additional validation on transit detections to filter false positives.
    
    Args:
        results: List of transit detection results
    
    Returns:
        list: Filtered transit detection results
    """
    validated_results = []
    
    for result in results:
        # Skip non-transit or failed results
        if not result.get('success', False) or result.get('transit_count', 0) == 0:
            result['validation_status'] = 'no_transit'
            validated_results.append(result)
            continue
        
        # Extract key properties
        transit_info = result.get('transit_info', {})
        periodicity_info = result.get('periodicity_info', {})
        
        # Initialize validation flags
        valid_transit_shape = False
        valid_period = False
        valid_depth = False
        valid_duration = False
        
        # 1. Check for realistic transit depth
        depths = transit_info.get('depths', [])
        if depths:
            median_depth = np.median([abs(d) for d in depths])
            # Typical transit depths range from 0.0001 to 0.02
            valid_depth = 0.0001 <= median_depth <= 0.02
        
        # 2. Check for periodic signal
        period = result.get('periodicity')
        if period:
            # Most planets have periods between 0.5 and 365 days
            valid_period = 0.5 <= period <= 365
        
        # 3. Check for consistent transit durations
        widths = transit_info.get('widths', [])
        if widths and len(widths) > 1:
            # Transit durations should be consistent
            width_std = np.std(widths)
            width_mean = np.mean(widths)
            valid_duration = width_std / width_mean < 0.3  # Less than 30% variation
        
        # 4. Check for appropriate transit duration vs. period relationship
        # Transit duration should roughly follow Kepler's laws
        if valid_period and widths:
            avg_duration_days = np.mean(widths) * np.median(np.diff(result.get('time', [])))
            # Transit duration is approximately (R_star/a) * P
            # For typical systems, this ratio is ~0.05-0.2 of the period
            valid_duration_period_ratio = 0.002 <= (avg_duration_days / period) <= 0.2
            valid_duration = valid_duration and valid_duration_period_ratio
        
        # Calculate overall validation score
        validation_score = sum([
            valid_transit_shape * 1.0,
            valid_period * 1.0,
            valid_depth * 1.0,
            valid_duration * 1.0
        ])
        
        # Decision: at least 2 validation criteria must be met
        is_validated = validation_score >= 2.0
        
        # Add validation info to result
        result['validation_status'] = 'validated' if is_validated else 'rejected'
        result['validation_score'] = validation_score
        result['validation_details'] = {
            'valid_transit_shape': valid_transit_shape,
            'valid_period': valid_period,
            'valid_depth': valid_depth,
            'valid_duration': valid_duration
        }
        
        validated_results.append(result)
    
    # Log validation results
    validated_count = sum(1 for r in validated_results if r.get('validation_status') == 'validated')
    rejected_count = sum(1 for r in validated_results if r.get('validation_status') == 'rejected')
    logger.info(f"Validation results: {validated_count} validated, {rejected_count} rejected")
    
    return validated_results

def prepare_datasets(results, exoplanet_labels=None):
    """
    Improved dataset preparation with better labeling criteria.
    """
    from data.dataset_generator import balance_dataset, augment_dataset
    
    # Skip entries with no transit images
    valid_results = [r for r in results if r.get('success', False) and r.get('has_transit_images', False)]
    
    if not valid_results:
        logger.warning("No valid transit images found for model training")
        return None, None, None
    
    # Create lookup table for exoplanet hosts
    host_stars = set()
    if exoplanet_labels is not None:
        host_stars = set(exoplanet_labels['hostname'].unique())
    
    # Collect data
    transit_images = []
    transit_segments = []
    labels = []
    
    for result in valid_results:
        # Extract stellar name
        file_name = os.path.basename(result['file_path'])
        stellar_name = file_name.split('_')[0]
        
        # Determine if this is a known exoplanet host
        is_known_host = stellar_name in host_stars
        
        # Add images and time series with improved labeling
        if result.get('transit_images') is not None and result.get('transit_segments') is not None:
            images = result['transit_images']
            segments = result['transit_segments']
            
            for i, (image, segment) in enumerate(zip(images, segments)):
                # Better labeling criteria combining host status AND transit characteristics
                transit_confidence = 0
                
                # If known host, start with higher confidence
                if is_known_host:
                    transit_confidence += 0.5
                
                # Extract segment characteristics for validation
                # 1. Check transit depth (too deep might be binary star)
                min_value = np.min(segment)
                baseline = np.median(segment)
                depth = baseline - min_value
                normalized_depth = depth / baseline
                
                # Realistic transit depths are typically 0.001 to 0.03
                if 0.001 < normalized_depth < 0.03:
                    transit_confidence += 0.25
                elif normalized_depth > 0.1:  # Too deep, likely a binary
                    transit_confidence -= 0.5
                
                # 2. Check transit shape (should be U or V shaped, not W or irregular)
                # Simple test: middle third should be consistently deeper than outer thirds
                segment_len = len(segment)
                third = segment_len // 3
                left = segment[:third]
                middle = segment[third:2*third]
                right = segment[2*third:]
                
                if np.median(middle) < np.median(left) and np.median(middle) < np.median(right):
                    transit_confidence += 0.25
                
                # Decision: label as positive if confidence is high enough
                is_transit = transit_confidence > 0.5
                
                # Add to dataset
                transit_images.append(image)
                transit_segments.append(segment)
                labels.append(1 if is_transit else 0)
    
    # Convert to numpy arrays
    if not transit_images:
        return None, None, None
    
    X_image = np.array(transit_images)
    X_timeseries = np.array(transit_segments)
    y = np.array(labels)
    
    # Balance and augment (keep these calls)
    X_image, X_timeseries, y = balance_dataset(X_image, X_timeseries, y, method='both')
    X_image, X_timeseries, y = augment_dataset(X_image, X_timeseries, y, augmentation_factor=3)
    
    return X_image, X_timeseries, y


def train_ai_models(X_image, X_timeseries, y, use_multimodal=True, output_dir=None):
    """
    Train AI models for exoplanet detection.
    
    Args:
        X_image: Image-based features
        X_timeseries: Time series features
        y: Labels
        use_multimodal: Whether to use multimodal fusion model
        output_dir: Directory to save model outputs (optional)
    
    Returns:
        dict: Trained models and evaluation metrics
    """
    from models.multimodal_model import build_ensemble_multimodal_model, combine_ensemble_predictions_weighted
    from models.multimodal_model import build_attention_fusion_model     
    
    if X_image is None or y is None:
        logger.warning("No valid data for model training")
        return None
    
    if output_dir is None:
        output_dir = config.MODEL_DIR
    
    # Reshape data for CNN
    X_image_cnn = reshape_data_for_cnn(X_image)
    
    # Create train/val/test splits
    X_img_train, X_img_val, X_img_test, y_train, y_val, y_test = prepare_train_test_split(X_image_cnn, y)
    
    # Train CNN model
    cnn_model = build_transit_detection_model((X_image.shape[1], X_image.shape[2]))
    cnn_model, cnn_history = train_model(
        cnn_model, X_img_train, y_train, X_img_val, y_val, 
        model_name='transit_detector',
        output_dir=output_dir
    )
    
    # Evaluate CNN model
    cnn_metrics = evaluate_model(
        cnn_model, X_img_test, y_test, 
        model_name='transit_detector',
        output_dir=output_dir
    )
    
    # Perform threshold analysis on CNN predictions
    cnn_preds = cnn_model.predict(X_img_test)
    cnn_threshold_analysis = calculate_precision_recall_curve_with_thresholds(y_test, cnn_preds)
    
    # Visualize threshold analysis
    visualize_threshold_analysis(
        cnn_threshold_analysis['threshold_metrics'],
        output_dir=output_dir,
        filename='cnn_threshold_analysis.png'
    )
    
    results = {
        'cnn_model': cnn_model,
        'cnn_metrics': cnn_metrics,
        'cnn_history': cnn_history,
        'cnn_threshold_analysis': cnn_threshold_analysis
    }
    
    # Train multimodal model if requested and time series data is available
    if use_multimodal and X_timeseries is not None:
        # Create train/val/test splits for time series
        X_ts_train, X_ts_val, X_ts_test, _, _, _ = prepare_train_test_split(X_timeseries, y)
        
        # Create ensemble of models (3 different architectures)
        logger.info("Creating ensemble of multimodal models")
        models = build_ensemble_multimodal_model(
            image_shape=(X_image.shape[1], X_image.shape[2]),
            timeseries_shape=X_timeseries.shape[1],
            num_classes=1
        )
        
        # Train each model in the ensemble with class weights
        trained_models = []
        model_histories = []
        
        for i, model in enumerate(models):
            logger.info(f"Training ensemble model {i+1}/{len(models)}")
            trained_model, history = train_model(
                model, 
                [X_img_train, X_ts_train], y_train, 
                [X_img_val, X_ts_val], y_val,
                model_name=f'ensemble_model_{i}',
                output_dir=output_dir,
                class_weight=class_weight_dict
            )
            trained_models.append(trained_model)
            model_histories.append(history)
        
        # Generate predictions from each model
        ensemble_predictions = []
        for i, model in enumerate(trained_models):
            logger.info(f"Generating predictions from ensemble model {i+1}/{len(trained_models)}")
            preds = model.predict([X_img_test, X_ts_test])
            ensemble_predictions.append(preds)
        
        # Optimize ensemble weights
        logger.info("Optimizing ensemble weights")
        optimized_weights = optimize_ensemble_weights(
            trained_models, 
            [X_img_val, X_ts_val], 
            y_val
        )
        logger.info(f"Optimized weights: {optimized_weights}")
        
        # Combine predictions using weighted averaging
        combined_preds = combine_ensemble_predictions_weighted(
            ensemble_predictions, 
            weights=optimized_weights,
            method='weighted'
        )
        
        # Perform threshold analysis on ensemble predictions
        ensemble_threshold_analysis = calculate_precision_recall_curve_with_thresholds(y_test, combined_preds)
        
        # Visualize threshold analysis
        visualize_threshold_analysis(
            ensemble_threshold_analysis['threshold_metrics'],
            output_dir=output_dir,
            filename='ensemble_threshold_analysis.png'
        )
        
        # Create dual threshold predictions
        high_precision_threshold = ensemble_threshold_analysis['threshold_metrics'][6]['threshold']  # ~0.7
        high_recall_threshold = ensemble_threshold_analysis['threshold_metrics'][2]['threshold']     # ~0.3
        
        high_precision_preds, high_recall_preds, confidence_levels = create_dual_threshold_predictions(
            combined_preds,
            high_precision_threshold=high_precision_threshold,
            high_recall_threshold=high_recall_threshold
        )
        
        # Evaluate dual threshold predictions
        high_precision_metrics = confusion_matrix_with_metrics(y_test, high_precision_preds)
        high_recall_metrics = confusion_matrix_with_metrics(y_test, high_recall_preds)
        
        # Visualize confusion matrices
        visualize_confusion_matrix(
            y_test, high_precision_preds,
            classes=['Non-Transit', 'Transit'],
            normalize=False,
            title='High Precision Mode Confusion Matrix',
            filename='high_precision_confusion_matrix.png',
            output_dir=output_dir
        )
        
        visualize_confusion_matrix(
            y_test, high_recall_preds,
            classes=['Non-Transit', 'Transit'],
            normalize=False,
            title='High Recall Mode Confusion Matrix',
            filename='high_recall_confusion_matrix.png',
            output_dir=output_dir
        )
        
        # Calculate ensemble metrics using traditional methods for comparison
        from utils.metrics import calculate_binary_metrics
        ensemble_metrics = calculate_binary_metrics(y_test, combined_preds)
        
        # Add ensemble results to overall results
        results.update({
            'ensemble_models': trained_models,
            'ensemble_histories': model_histories,
            'ensemble_weights': optimized_weights,
            'ensemble_metrics': ensemble_metrics,
            'ensemble_threshold_analysis': ensemble_threshold_analysis,
            'high_precision_metrics': high_precision_metrics,
            'high_recall_metrics': high_recall_metrics,
            'high_precision_threshold': high_precision_threshold,
            'high_recall_threshold': high_recall_threshold
        })
        
        # Compare model performance
        from utils.visualization import visualize_model_comparison
        visualize_model_comparison(
            {
                'CNN': cnn_metrics,
                'Ensemble': ensemble_metrics,
                'High Precision': high_precision_metrics,
                'High Recall': high_recall_metrics
            },
            filename='model_comparison.png',
            output_dir=output_dir
        )
        
        # Create comparison table of precision-recall tradeoffs
        tradeoff_table = pd.DataFrame([
            {
                'Model': 'CNN',
                'Precision': cnn_metrics['precision'],
                'Recall': cnn_metrics['recall'],
                'F1 Score': cnn_metrics['f1_score'],
                'False Positives': cnn_metrics['false_positives'],
                'False Negatives': cnn_metrics['false_negatives']
            },
            {
                'Model': 'Ensemble',
                'Precision': ensemble_metrics['precision'],
                'Recall': ensemble_metrics['recall'],
                'F1 Score': ensemble_metrics['f1_score'],
                'False Positives': ensemble_metrics['false_positives'],
                'False Negatives': ensemble_metrics['false_negatives']
            },
            {
                'Model': 'High Precision Mode',
                'Precision': high_precision_metrics['precision'],
                'Recall': high_precision_metrics['recall'],
                'F1 Score': high_precision_metrics['f1_score'],
                'False Positives': high_precision_metrics['false_positives'],
                'False Negatives': high_precision_metrics['false_negatives']
            },
            {
                'Model': 'High Recall Mode',
                'Precision': high_recall_metrics['precision'],
                'Recall': high_recall_metrics['recall'],
                'F1 Score': high_recall_metrics['f1_score'],
                'False Positives': high_recall_metrics['false_positives'],
                'False Negatives': high_recall_metrics['false_negatives']
            }
        ])
        
        # Save comparison table
        tradeoff_table.to_csv(os.path.join(output_dir, 'precision_recall_tradeoffs.csv'), index=False)
    
    return results


def run_enhanced_pipeline(obs_table=None, exoplanet_labels=None, use_synthetic=True, 
                         use_multimodal=True, optimize_hyperparams=False, use_cache=True,
                         max_workers=4):
    """
    Run the enhanced exoplanet detection pipeline with improved data and models.
    
    Args:
        obs_table: Table of observations (optional)
        exoplanet_labels: DataFrame with exoplanet labels (optional)
        use_synthetic: Whether to use synthetic data
        use_multimodal: Whether to use multimodal fusion model
        optimize_hyperparams: Whether to optimize hyperparameters
        use_cache: Whether to use cached data
        max_workers: Maximum number of parallel workers
    
    Returns:
        dict: Pipeline results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(config.RESULTS_DIR, f"pipeline_run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    test_file_path = os.path.join(result_dir, "pipeline_started.txt")
    with open(test_file_path, 'w') as f:
        f.write(f"Pipeline started at {datetime.now()}\n")
    
    logger.info(f"Starting enhanced pipeline run, results in: {result_dir}")
    
    if use_synthetic:
        logger.info("Using enhanced synthetic data")
        # Use enhanced synthetic data generation
        X_ts, X_image, y = generate_enhanced_synthetic_data(n_samples=1500)
        
        # Train models with enhanced approach
        model_results = train_ai_models_enhanced(X_image, X_ts, y, use_multimodal, output_dir=result_dir)
        
        return {
            'timestamp': timestamp,
            'result_dir': result_dir,
            'synthetic_data': True,
            'model_results': model_results
        }
    
    # Check inputs
    if obs_table is None:
        raise ValueError("Observation table is required for real data pipeline")
    
    # Download light curve files
    logger.info("Downloading light curve files")
    light_curve_files = download_light_curves(obs_table, use_cache, max_workers)
    
    if not light_curve_files:
        logger.error("No light curve files downloaded")
        return {
            'timestamp': timestamp,
            'result_dir': result_dir,
            'error': "No light curve files downloaded"
        }
    
    # Process light curves in parallel
    logger.info(f"Processing {len(light_curve_files)} light curve files")
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a subdirectory for each file's results
        file_results = []
        for file_path in light_curve_files:
            file_result_dir = os.path.join(
                result_dir, 
                os.path.basename(file_path).split('.')[0]
            )
            file_results.append(
                executor.submit(process_light_curve, file_path, file_result_dir)
            )
        
        # Collect results
        for future in file_results:
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"Added result for {result.get('file_path', 'unknown')}")
                else:
                    logger.warning("Got empty result from worker")
            except Exception as e:
                logger.error(f"Error processing light curve: {e}", exc_info=True)
    
    # Prepare data for AI training
    logger.info("Preparing datasets for AI model training")
    X_image, X_timeseries, y = prepare_datasets(results, exoplanet_labels)
    
    # Train AI models
    model_results = None
    if X_image is not None and y is not None:
        logger.info("Training AI models")
        model_results = train_ai_models_enhanced(X_image, X_timeseries, y, use_multimodal, output_dir=result_dir)
    else:
        logger.warning("Insufficient data for AI model training")
    
    # Generate report
    logger.info("Generating report")
    report_path = generate_report(results, model_results, timestamp, result_dir)
    
    # Create summary visualization
    visualize_detection_results(
        results, 
        filename='detection_summary.png',
        output_dir=result_dir
    )
    
    logger.info(f"Pipeline completed successfully. Report: {report_path}")
    
    # Return the summarized results
    pipeline_results = {
        'timestamp': timestamp,
        'result_dir': result_dir,
        'file_count': len(light_curve_files),
        'transit_count': sum(1 for r in results if r.get('transit_count', 0) > 0),
        'model_results': model_results,
        'report_path': report_path
    }
    
    return pipeline_results
    
    results = {
        'cnn_model': cnn_model,
        'cnn_metrics': cnn_metrics,
        'cnn_history': cnn_history
    }
    
    # Train multimodal model if requested and time series data is available
    if use_multimodal and X_timeseries is not None:
        # Create train/val/test splits for time series
        X_ts_train, X_ts_val, X_ts_test, _, _, _ = prepare_train_test_split(X_timeseries, y)
        
        # === ENSEMBLE APPROACH ===
        # Create an ensemble of models (3 different architectures)
        logger.info("Creating ensemble of multimodal models")
        models = build_ensemble_multimodal_model(
            image_shape=(X_image.shape[1], X_image.shape[2]),
            timeseries_shape=X_timeseries.shape[1],
            num_classes=1
        )
        
        # Train each model in the ensemble
        trained_models = []
        model_histories = []
        
        for i, model in enumerate(models):
            logger.info(f"Training ensemble model {i+1}/{len(models)}")
            trained_model, history = train_model(
                model, 
                [X_img_train, X_ts_train], y_train, 
                [X_img_val, X_ts_val], y_val,
                model_name=f'ensemble_model_{i}',
                output_dir=output_dir
            )
            trained_models.append(trained_model)
            model_histories.append(history)
        
        # Generate predictions from each model
        ensemble_predictions = []
        for i, model in enumerate(trained_models):
            logger.info(f"Generating predictions from ensemble model {i+1}/{len(trained_models)}")
            preds = model.predict([X_img_test, X_ts_test])
            ensemble_predictions.append(preds)
        
        # Combine predictions using averaging
        logger.info("Combining ensemble predictions")
        combined_preds = combine_ensemble_predictions_weighted(ensemble_predictions, method='average')
        
        # Evaluate ensemble performance
        from utils.metrics import calculate_binary_metrics
        ensemble_metrics = calculate_binary_metrics(y_test, combined_preds)
        
        # Add ensemble results
        results.update({
            'ensemble_models': trained_models,
            'ensemble_histories': model_histories,
            'ensemble_metrics': ensemble_metrics,
            # Include the first model as the multimodal model for compatibility
            'multimodal_model': trained_models[0],
            'multimodal_metrics': ensemble_metrics
        })
        
        # Compare model performance
        from utils.visualization import visualize_model_comparison
        visualize_model_comparison(
            {
                'CNN': cnn_metrics,
                'Ensemble': ensemble_metrics
            },
            filename='model_comparison.png',
            output_dir=output_dir
        )
    
    return results


def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """
    Generate synthetic data for testing the pipeline.
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Level of noise to add
    
    Returns:
        tuple: (X_timeseries, X_image, y)
    """
    logger.info(f"Generating {n_samples} synthetic samples")
    
    # Generate synthetic time series data
    synthetic_time_series = []
    synthetic_images = []
    synthetic_labels = []
    
    # Create transit and non-transit examples
    for i in range(n_samples):
        # Generate time parameter
        t = np.linspace(0, 10, 100)
        
        # Decide if this is a transit (1) or not (0)
        is_transit = np.random.choice([0, 1])
        
        if is_transit:
            # Generate a transit-like signal
            # Transit occurs at random time between 2 and 8
            transit_time = np.random.uniform(2, 8)
            transit_width = np.random.uniform(0.2, 0.5)
            transit_depth = np.random.uniform(0.01, 0.03)
            
            # Generate the light curve
            flux = np.ones_like(t)
            transit_mask = (t > transit_time - transit_width/2) & (t < transit_time + transit_width/2)
            flux[transit_mask] = 1 - transit_depth
            
            # Add noise
            flux += np.random.normal(0, noise_level, len(t))
        else:
            # Generate a non-transit light curve
            flux = np.ones_like(t)
            
            # Add noise
            flux += np.random.normal(0, noise_level, len(t))
            
            # Add a small sinusoidal variation
            flux += 0.01 * np.sin(2 * np.pi * t / 2)
        
        # Create a 2D image representation
        image = np.zeros((64, 64))
        for j in range(64):
            # Resample the flux to match the image width
            resampled_flux = np.interp(
                np.linspace(0, len(flux)-1, 64), 
                np.arange(len(flux)), 
                flux
            )
            image[j, :] = resampled_flux
        
        synthetic_time_series.append(flux)
        synthetic_images.append(image)
        synthetic_labels.append(is_transit)
    
    # Convert to numpy arrays
    X_ts = np.array(synthetic_time_series)
    X_image = np.array(synthetic_images)
    y = np.array(synthetic_labels)
    
    return X_ts, X_image, y


def run_pipeline(obs_table=None, exoplanet_labels=None, use_synthetic=False, 
                 use_multimodal=True, optimize_hyperparams=False, use_cache=True,
                 max_workers=4):
    """
    Run the full exoplanet detection pipeline.
    
    Args:
        obs_table: Table of observations (optional)
        exoplanet_labels: DataFrame with exoplanet labels (optional)
        use_synthetic: Whether to use synthetic data
        use_multimodal: Whether to use multimodal fusion model
        optimize_hyperparams: Whether to optimize hyperparameters
        use_cache: Whether to use cached data
        max_workers: Maximum number of parallel workers
    
    Returns:
        dict: Pipeline results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(config.RESULTS_DIR, f"pipeline_run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    test_file_path = os.path.join(result_dir, "pipeline_started.txt")
    with open(test_file_path, 'w') as f:
        f.write(f"Pipeline started at {datetime.now()}\n")
    
    logger.info(f"Starting pipeline run, results in: {result_dir}")
    
    if use_synthetic:
        logger.info("Using synthetic data")
        X_ts, X_image, y = generate_synthetic_data()
        
        # Train models
        model_results = train_ai_models(X_image, X_ts, y, use_multimodal, output_dir=result_dir)
        
        return {
            'timestamp': timestamp,
            'result_dir': result_dir,
            'synthetic_data': True,
            'model_results': model_results
        }
    
    # Check inputs
    if obs_table is None:
        raise ValueError("Observation table is required for real data pipeline")
    
    # Download light curve files
    logger.info("Downloading light curve files")
    light_curve_files = download_light_curves(obs_table, use_cache, max_workers)
    
    if not light_curve_files:
        logger.error("No light curve files downloaded")
        return {
            'timestamp': timestamp,
            'result_dir': result_dir,
            'error': "No light curve files downloaded"
        }
    
    # Process light curves in parallel
    logger.info(f"Processing {len(light_curve_files)} light curve files")
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a subdirectory for each file's results
        file_results = []
        for file_path in light_curve_files:
            file_result_dir = os.path.join(
                result_dir, 
                os.path.basename(file_path).split('.')[0]
            )
            file_results.append(
                executor.submit(process_light_curve, file_path, file_result_dir)
            )
        
        # Collect results
        for future in file_results:
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logger.info(f"Added result for {result.get('file_path', 'unknown')}")
                else:
                    logger.warning("Got empty result from worker")
            except Exception as e:
                logger.error(f"Error processing light curve: {e}", exc_info=True)
    
    # Prepare data for AI training
    logger.info("Preparing datasets for AI model training")
    X_image, X_timeseries, y = prepare_datasets(results, exoplanet_labels)
    
    # Train AI models
    model_results = None
    if X_image is not None and y is not None:
        logger.info("Training AI models")
        model_results = train_ai_models(X_image, X_timeseries, y, use_multimodal, output_dir=result_dir)
    else:
        logger.warning("Insufficient data for AI model training")
    
    # Generate report
    logger.info("Generating report")
    report_path = generate_report(results, model_results, timestamp, result_dir)
    
    # Create summary visualization
    visualize_detection_results(
        results, 
        filename='detection_summary.png',
        output_dir=result_dir
    )
    
    logger.info(f"Pipeline completed successfully. Report: {report_path}")
    
    # Return the summarized results
    pipeline_results = {
        'timestamp': timestamp,
        'result_dir': result_dir,
        'file_count': len(light_curve_files),
        'transit_count': sum(1 for r in results if r.get('transit_count', 0) > 0),
        'model_results': model_results,
        'report_path': report_path
    }
    
    return pipeline_results


# ENHANCED PIPELINE FUNCTIONS

def generate_enhanced_synthetic_data(n_samples=1000, noise_levels=[0.05, 0.1, 0.2], transit_depth_range=(0.005, 0.05)):
    """
    Generate more diverse synthetic data for testing the pipeline.
    
    Args:
        n_samples: Number of samples to generate
        noise_levels: List of noise levels to apply
        transit_depth_range: Range of transit depths (min, max)
    
    Returns:
        tuple: (X_timeseries, X_image, y)
    """
    logger.info(f"Generating {n_samples} enhanced synthetic samples")
    
    # Generate synthetic time series data
    synthetic_time_series = []
    synthetic_images = []
    synthetic_labels = []
    transit_properties = []  # Store properties for later analysis
    
    # Create transit and non-transit examples
    for i in range(n_samples):
        # Generate time parameter (longer time series for more complex signals)
        t = np.linspace(0, 15, 150)
        
        # Decide if this is a transit (1) or not (0)
        is_transit = np.random.choice([0, 1])
        
        # Select random noise level from options
        noise_level = np.random.choice(noise_levels)
        
        if is_transit:
            # Generate a transit-like signal with more variability
            # Transit occurs at random time between 3 and 12
            transit_time = np.random.uniform(3, 12)
            transit_width = np.random.uniform(0.1, 0.8)  # More varied durations
            transit_depth = np.random.uniform(transit_depth_range[0], transit_depth_range[1])
            
            # Add possibility of multiple transits for some examples
            num_transits = np.random.choice([1, 1, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.05, 0.05])
            
            # Generate the light curve
            flux = np.ones_like(t)
            
            # Add stellar variability (sine wave with random period and amplitude)
            if np.random.random() < 0.5:  # 50% chance of stellar variability
                var_period = np.random.uniform(3, 10)
                var_amplitude = np.random.uniform(0.001, 0.01)
                flux += var_amplitude * np.sin(2 * np.pi * t / var_period)
            
            # Add transits
            for j in range(num_transits):
                if j > 0:
                    # For multiple transits, space them out based on a "period"
                    period = np.random.uniform(2, 5)
                    current_transit_time = transit_time + j * period
                    if current_transit_time > max(t):
                        continue
                else:
                    current_transit_time = transit_time
                
                # Create transit shape (more realistic transition)
                # Use a smoother transit profile - approximating limb darkening
                transit_mask = np.abs(t - current_transit_time) < transit_width/2
                distance = np.abs(t - current_transit_time) / (transit_width/2)
                distance = distance[transit_mask]
                
                # Smooth transit shape (quadratic limb darkening approximation)
                transit_shape = 1 - transit_depth * (1 - 0.6*distance**2)
                flux[transit_mask] = flux[transit_mask] * transit_shape
            
            # Track transit properties for this example
            transit_properties.append({
                'depth': transit_depth,
                'width': transit_width,
                'time': transit_time,
                'num_transits': num_transits,
                'noise_level': noise_level
            })
            
        else:
            # Generate a non-transit light curve with realistic stellar variations
            flux = np.ones_like(t)
            
            # Add stellar variability (sine waves with random period and amplitude)
            if np.random.random() < 0.7:  # 70% chance of variability in non-transit
                var_period1 = np.random.uniform(2, 7)
                var_amplitude1 = np.random.uniform(0.001, 0.015)
                flux += var_amplitude1 * np.sin(2 * np.pi * t / var_period1)
                
                # Sometimes add a second variability component
                if np.random.random() < 0.4:
                    var_period2 = np.random.uniform(7, 15)
                    var_amplitude2 = np.random.uniform(0.001, 0.01)
                    flux += var_amplitude2 * np.sin(2 * np.pi * t / var_period2)
            
            # Occasionally add a non-transit dip (e.g., star spot)
            if np.random.random() < 0.15:  # 15% chance of star spot
                spot_time = np.random.uniform(3, 12)
                spot_width = np.random.uniform(0.3, 1.5)  # Wider than transits
                spot_depth = np.random.uniform(0.005, 0.02)
                
                spot_mask = np.abs(t - spot_time) < spot_width/2
                distance = np.abs(t - spot_time) / (spot_width/2)
                distance = distance[spot_mask]
                
                # Smoother and wider profile than transit
                spot_shape = 1 - spot_depth * np.cos(distance * np.pi/2)**2
                flux[spot_mask] = flux[spot_mask] * spot_shape
        
        # Add noise after all other signal components
        flux += np.random.normal(0, noise_level, len(t))
        
        # Create a 2D image representation with more detail (higher resolution)
        image = np.zeros((64, 64))
        for j in range(64):
            # Resample the flux to match the image width
            resampled_flux = np.interp(
                np.linspace(0, len(flux)-1, 64), 
                np.arange(len(flux)), 
                flux
            )
            image[j, :] = resampled_flux
        
        synthetic_time_series.append(flux)
        synthetic_images.append(image)
        synthetic_labels.append(is_transit)
    
    # Convert to numpy arrays
    X_ts = np.array(synthetic_time_series)
    X_image = np.array(synthetic_images)
    y = np.array(synthetic_labels)
    
    # Print dataset statistics
    transit_count = np.sum(y)
    logger.info(f"Generated dataset with {transit_count} transit examples ({transit_count/len(y)*100:.1f}%)")
    
    # Store transit properties for analysis
    if len(transit_properties) > 0:
        props_df = pd.DataFrame(transit_properties)
        os.makedirs(config.METADATA_DIR, exist_ok=True)
        props_df.to_csv(os.path.join(config.METADATA_DIR, 'synthetic_transit_properties.csv'), index=False)
    
    return X_ts, X_image, y


def train_ai_models_enhanced(X_image, X_timeseries, y, use_multimodal=True, output_dir=None):
    """
    Train AI models for exoplanet detection with enhanced ensemble and dual thresholds.
    
    Args:
        X_image: Image-based features
        X_timeseries: Time series features
        y: Labels
        use_multimodal: Whether to use multimodal fusion model
        output_dir: Directory to save model outputs (optional)
    
    Returns:
        dict: Trained models and evaluation metrics
    """
    from models.multimodal_model import build_ensemble_multimodal_model, combine_ensemble_predictions_weighted, optimize_ensemble_weights
    from models.multimodal_model import create_dual_threshold_predictions
    from utils.metrics import calculate_precision_recall_curve_with_thresholds, visualize_threshold_analysis, confusion_matrix_with_metrics
    from utils.visualization import visualize_confusion_matrix
    
    if X_image is None or y is None:
        logger.warning("No valid data for model training")
        return None
    
    if output_dir is None:
        output_dir = config.MODEL_DIR
    
    # Create train/val/test splits with stratification
    X_img_train, X_img_val, X_img_test, y_train, y_val, y_test = prepare_train_test_split(X_image, y)
    
    # Train CNN model with class balancing
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    
    # Create a balanced class weight
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Build CNN model
    cnn_model = build_transit_detection_model((X_image.shape[1], X_image.shape[2]))
    
    # Train with class weights
    cnn_model, cnn_history = train_model(
        cnn_model, X_img_train, y_train, X_img_val, y_val, 
        model_name='transit_detector',
        output_dir=output_dir,
        class_weight=class_weight_dict
    )
    
    # Evaluate CNN model
    cnn_metrics = evaluate_model(
        cnn_model, X_img_test, y_test, 
        model_name='transit_detector',
        output_dir=output_dir
    )