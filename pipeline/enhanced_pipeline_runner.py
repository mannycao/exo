"""
Enhanced pipeline execution for exoplanet detection.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import config
from data.data_fetcher import download_light_curves, fetch_exoplanet_labels
from data.enhanced_augmentation import (
    augment_and_balance_dataset, combine_kepler_tess_data, analyze_data_characteristics
)
from models.enhanced_trainer import train_enhanced_model
from pipeline.pipeline_runner import (
    process_light_curve, prepare_datasets, prepare_train_test_split
)
from utils.metrics import (
    calculate_precision_recall_curve_with_thresholds,
    visualize_threshold_analysis, confusion_matrix_with_metrics
)
from utils.visualization import visualize_model_comparison, visualize_confusion_matrix
from models.multimodal_model import (
    optimize_ensemble_weights,
    combine_ensemble_predictions_weighted,
    create_dual_threshold_predictions
)

logger = logging.getLogger(__name__)


def prepare_enhanced_datasets(results, exoplanet_labels=None, include_tess=True, augmentation_factor=3):
    """
    Prepare enhanced datasets for AI model training based on processing results.
    
    Args:
        results: List of processing results
        exoplanet_labels: DataFrame with exoplanet labels (optional)
        include_tess: Whether to include TESS data
        augmentation_factor: Factor by which to augment positive examples
    
    Returns:
        tuple: (X_image, X_timeseries, y) for model training
    """
    logger.info("Preparing enhanced datasets with augmentation")
    
    # First, prepare the base dataset
    X_image, X_timeseries, y = prepare_datasets(results, exoplanet_labels)
    
    if X_image is None or X_timeseries is None or y is None:
        logger.warning("No valid transit images found for model training")
        return None, None, None
    
    # Option to combine with TESS data
    if include_tess:
        try:
            logger.info("Adding TESS data to training set")
            X_image_combined, X_ts_combined, y_combined = combine_kepler_tess_data(
                max_kepler=100,  # Adjust as needed
                max_tess=50      # Adjust as needed
            )
            
            # If successful, use the combined dataset
            if X_image_combined is not None and len(X_image_combined) > 0:
                X_image, X_timeseries, y = X_image_combined, X_ts_combined, y_combined
                logger.info(f"Combined dataset size: {len(y)} examples")
        except Exception as e:
            logger.warning(f"Failed to combine with TESS data: {e}")
            # Fall back to Kepler-only data
    
    # Analyze data characteristics before augmentation
    try:
        data_stats = analyze_data_characteristics(X_image, X_timeseries, y)
        logger.info(f"Dataset before augmentation: {data_stats['positive_examples']} positive, "
                   f"{data_stats['negative_examples']} negative examples")
    except Exception as e:
        logger.warning(f"Error analyzing data characteristics: {e}")
    
    # Apply enhanced augmentation
    try:
        X_image_aug, X_timeseries_aug, y_aug = augment_and_balance_dataset(
            X_image, X_timeseries, y,
            augmentation_factor=augmentation_factor,
            balance_ratio=0.3  # Target 50% positive examples
        )
        
        logger.info(f"Augmented dataset size: {len(y_aug)} examples")
        
        # Return the augmented dataset
        return X_image_aug, X_timeseries_aug, y_aug
        
    except Exception as e:
        logger.error(f"Error in data augmentation: {e}", exc_info=True)
        # Fall back to the original dataset
        return X_image, X_timeseries, y


def train_ai_models_enhanced(X_image, X_timeseries, y, use_multimodal=True, output_dir=None):
    """
    Train AI models for exoplanet detection with enhanced techniques.
    
    Args:
        X_image: Image-based features
        X_timeseries: Time series features
        y: Labels
        use_multimodal: Whether to use multimodal fusion model
        output_dir: Directory to save model outputs (optional)
    
    Returns:
        dict: Trained models and evaluation metrics
    """
    from models.cnn_model import build_transit_detection_model
    from models.multimodal_model import build_ensemble_multimodal_model
    from data.light_curve_processor import reshape_data_for_cnn
    
    if X_image is None or y is None:
        logger.warning("No valid data for model training")
        return None
    
    if output_dir is None:
        output_dir = config.MODEL_DIR
    
    # Reshape data for CNN
    X_image_cnn = reshape_data_for_cnn(X_image)
    
    # Create train/val/test splits
    X_img_train, X_img_val, X_img_test, y_train, y_val, y_test = prepare_train_test_split(X_image_cnn, y)
    
    # Train CNN model with enhanced techniques
    cnn_model = build_transit_detection_model((X_image.shape[1], X_image.shape[2]))
    cnn_model, cnn_history = train_enhanced_model(
        cnn_model, X_img_train, y_train, X_img_val, y_val, 
        model_name='transit_detector',
        output_dir=output_dir,
        use_focal_loss=True,
        use_class_weights=True
    )
    
    # Evaluate CNN model
    from utils.metrics import calculate_binary_metrics
    
    # Get predictions
    cnn_preds = cnn_model.predict(X_img_test)
    
    # Calculate metrics
    cnn_metrics = calculate_binary_metrics(y_test, cnn_preds)
    
    # Perform threshold analysis on CNN predictions
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
        
        # Train each model in the ensemble with enhanced techniques
        trained_models = []
        model_histories = []
        
        for i, model in enumerate(models):
            logger.info(f"Training ensemble model {i+1}/{len(models)}")
            trained_model, history = train_enhanced_model(
                model, 
                [X_img_train, X_ts_train], y_train, 
                [X_img_val, X_ts_val], y_val,
                model_name=f'ensemble_model_{i}',
                output_dir=output_dir,
                use_focal_loss=True,
                use_class_weights=True
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
        
        # Calculate ensemble metrics
        ensemble_metrics = calculate_binary_metrics(y_test, combined_preds)
        
        # Find optimal thresholds for different objectives
        high_precision_threshold = ensemble_threshold_analysis.get('optimal_f1_threshold', 0.7)
        high_recall_threshold = max(0.3, ensemble_threshold_analysis.get('optimal_f2_threshold', 0.3))
        
        # Create dual threshold predictions
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


def run_enhanced_pipeline(obs_table=None, exoplanet_labels=None, light_curve_files=None,
                         use_synthetic=True, use_multimodal=True, optimize_hyperparams=False, 
                         use_cache=True, max_workers=4, enhanced_augmentation=True, 
                         augmentation_factor=3, include_tess=False):
    """
    Run the enhanced exoplanet detection pipeline with improved data and models.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(config.RESULTS_DIR, f"pipeline_run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    test_file_path = os.path.join(result_dir, "pipeline_started.txt")
    with open(test_file_path, 'w') as f:
        f.write(f"Pipeline started at {datetime.now()}\n")
    
    logger.info(f"Starting enhanced pipeline run, results in: {result_dir}")
    
    # Generate synthetic data if requested
    if use_synthetic:
        logger.info("Generating synthetic light curves for testing")
        # Import the function to generate samples
        from data.data_fetcher import generate_sample_light_curves
        light_curve_files = generate_sample_light_curves(num_samples=20)
        
        # Create some basic exoplanet labels if needed
        if exoplanet_labels is None:
            import pandas as pd
            exoplanet_labels = pd.DataFrame({
                'hostname': [f'KIC 1000000{i}' for i in range(10)],
                'pl_name': [f'KIC 1000000{i} b' for i in range(10)],
                'pl_orbper': [np.random.uniform(1, 30) for _ in range(10)],
                'pl_rade': [np.random.uniform(1, 5) for _ in range(10)],
                'discoverymethod': ['Transit' for _ in range(10)]
            })
    else:
        # Check inputs for real data pipeline
        if obs_table is None and light_curve_files is None:
            raise ValueError("Either observation table or light curve files must be provided for real data pipeline")
    
    # Download light curve files if needed and not provided
    if light_curve_files is None:
        if use_synthetic:
            logger.info("Generating synthetic light curves for testing")
            # Import the function to generate samples
            from data.data_fetcher import generate_sample_light_curves
            light_curve_files = generate_sample_light_curves(num_samples=20)
        else:
            # Use enhanced data fetcher with direct download from Kepler archive
            logger.info(f"Fetching Kepler data")
            light_curve_files = download_light_curves(obs_table, use_cache, max_workers)
    else:
        logger.info(f"Using {len(light_curve_files)} provided light curve files")

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
    
    if enhanced_augmentation:
        X_image, X_timeseries, y = prepare_enhanced_datasets(
            results, 
            exoplanet_labels,
            include_tess=include_tess,
            augmentation_factor=augmentation_factor
        )
    else:
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
    from pipeline.report_generator import generate_report
    report_path = generate_report(results, model_results, timestamp, result_dir)
    
    # Create summary visualization
    from utils.visualization import visualize_detection_results
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
