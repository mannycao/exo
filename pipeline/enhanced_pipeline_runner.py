# pipeline/enhanced_pipeline_runner.py

import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import time
import json

import config
from data.dataset_generator import create_dataset
from data.augmentation_utils import augment_data
from models.multimodal_model import build_multimodal_fusion_model
from models.enhanced_model_trainer import EnhancedModelTrainer
from models.model_trainer_utils import train_enhanced_model
from pipeline.report_generator import generate_report

# CACL XAI Imports
from xai_cacl_explainer import CACLFeatureExtractor, explain_with_cacl, compute_confidence_score, train_confidence_model
from cacl_utils import create_feature_partitions, build_context_groups, compute_dependency_matrix, compute_context_embeddings
import torch # Required for CACLFeatureExtractor


logger = logging.getLogger(__name__)

def run_enhanced_pipeline(light_curve_files, output_dir_str, timestamp):
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


    # Add original index to each result for later reference
    for i, res in enumerate(all_pipeline_results_aug):
        res['original_index'] = i

    # Split data into training and validation sets, including all_pipeline_results
    X_ts_train, X_ts_val, \
    X_img_train, X_img_val, \
    X_features_train, X_features_val, \
    y_train, y_val, \
    all_pipeline_results_train, all_pipeline_results_val = train_test_split(
        X_ts_aug, X_img_aug, X_features_aug, y_aug, all_pipeline_results_aug, # Use augmented data
        test_size=0.2, random_state=42, stratify=y_aug
    )

    if config.USE_MULTIMODAL:
        logger.info("Building the multimodal fusion model...")
        multimodal_model = build_multimodal_fusion_model(
            timeseries_shape=X_ts_train.shape[1:],
            image_shape=X_img_train.shape[1:],
            feature_shape=X_features_train.shape[1:]
        )

        # Initialize the multimodal model trainer
        trainer = EnhancedModelTrainer(
            model=multimodal_model,
            output_dir=output_dir_str,
            use_bayesian=True,
            use_cacl=True,  # Key change: Enable CACL
            cacl_model_path=config.CACL_MODEL_PATH
        )

        # Prepare CACL context if enabled
        if trainer.use_cacl:
            logger.info("Initializing CACL Explainer and preparing context...")
            # Create CACL partitions based on the training data
            # For time-series data, partitions are typically contiguous segments
            sample_light_curve_length = X_ts_train.shape[1]
            partition_size = sample_light_curve_length // config.CACL_K_PARTITIONS
            cacl_partitions = [] # Initialize to empty list
            cacl_partitions = [[j for j in range(i * partition_size, (i + 1) * partition_size)] for i in range(config.CACL_K_PARTITIONS - 1)]
            cacl_partitions.append([j for j in range((config.CACL_K_PARTITIONS - 1) * partition_size, sample_light_curve_length)])
            cacl_partitions = [p for p in cacl_partitions if p] # Filter out empty partitions

            trainer.cacl_explainer.partitions = cacl_partitions # Update explainer with actual partitions

            # Pre-compute context embeddings and dependency matrix
            # This is done once after data loading and before training/explanation generation
            # The trainer will handle the actual computation when get_cacl_explanations is called
            logger.info("CACL Explainer and context prepared.")

        # Prepare datasets for the trainer
        train_dataset = (X_ts_train, X_img_train, X_features_train, y_train)
        validation_dataset = (X_ts_val, X_img_val, X_features_val, y_val)

        logger.info("Training the multimodal model...")
        history, y_val_actual, y_pred_val_raw, cnn_metrics = trainer.train(
            train_dataset,
            validation_dataset,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE
        )

        # After training, get CACL explanations
        cacl_explanations = {}
        cacl_stats = {}

        # Convert probabilities to binary predictions for validation set
        y_pred_binary = np.round(y_pred_val_raw).astype(int)

        # Add classification to each result in all_pipeline_results_val
        for i, result in enumerate(all_pipeline_results_val):
            true_label = y_val_actual[i]
            predicted_label = y_pred_binary[i]

            if true_label == 1 and predicted_label == 1:
                result['classification'] = 'True Positive'
            elif true_label == 0 and predicted_label == 1:
                result['classification'] = 'False Positive'
            elif true_label == 0 and predicted_label == 0:
                result['classification'] = 'True Negative'
            elif true_label == 1 and predicted_label == 0:
                result['classification'] = 'False Negative'
            else:
                result['classification'] = 'Unknown' # Should not happen with binary classification

        if trainer.use_cacl:
            logger.info("Generating CACL explanations...")
            logger.debug(f"Validation Dataset Length: {len(validation_dataset[0])}")
            logger.debug(f"X_ts_val shape: {validation_dataset[0].shape}")
            
            # Get original indices of validation samples
            validation_original_indices = [res['original_index'] for res in all_pipeline_results_val]
            validation_classifications = [res['classification'] for res in all_pipeline_results_val]
            print(f"DEBUG: validation_original_indices: {validation_original_indices}")
            print(f"DEBUG: validation_classifications: {validation_classifications}")

            cacl_explanations = trainer.get_cacl_explanations(validation_dataset, cacl_partitions, validation_original_indices, validation_classifications)
            logger.info("CACL explanations generated.")
            print(f"DEBUG: cacl_explanations keys: {cacl_explanations.keys()}")

            # Calculate mean and range for Max Disagreement and Context Similarity
            all_max_disagreements = [exp['max_disagreement'] for exp in cacl_explanations.values() if 'max_disagreement' in exp]
            all_context_similarities = [exp['context_similarity'] for exp in cacl_explanations.values() if 'context_similarity' in exp]
            all_violated_dependencies = [len(exp['violated_dependencies']) for exp in cacl_explanations.values() if 'violated_dependencies' in exp]

            confidence_model = train_confidence_model(cacl_explanations, y_val_actual, validation_original_indices)

            # Compute and add confidence score to each explanation
            for record_id, explanation in cacl_explanations.items():
                confidence_score = compute_confidence_score(
                    explanation,
                    confidence_model
                )
                explanation['confidence_score'] = confidence_score

            cacl_stats = {
                'max_disagreement_mean': np.mean(all_max_disagreements) if all_max_disagreements else 'N/A',
                'max_disagreement_min': min(all_max_disagreements) if all_max_disagreements else 'N/A',
                'max_disagreement_max': max(all_max_disagreements) if all_max_disagreements else 'N/A',
                'context_similarity_mean': np.mean(all_context_similarities) if all_context_similarities else 'N/A',
                'context_similarity_min': min(all_context_similarities) if all_context_similarities else 'N/A',
                'context_similarity_max': max(all_context_similarities) if all_context_similarities else 'N/A',
            }

        model_report_data = {
            'cnn_metrics': cnn_metrics,
            'cacl_explanations': cacl_explanations,
            'cacl_stats': cacl_stats
        }

        generate_report(
            results=all_pipeline_results_val,
            model_results=model_report_data,
            y_true=y_val_actual,
            y_pred=y_pred_val_raw,
            timestamp=timestamp,
            output_dir=result_dir,
            validation_original_indices=validation_original_indices,
            validation_classifications=validation_classifications
        )

        logger.info("Enhanced multimodal pipeline finished successfully.")
        return 0
    else:
        logger.info("Skipping multimodal model training as USE_MULTIMODAL is False.")
        return 0
