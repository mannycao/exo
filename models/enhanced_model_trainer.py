import os
import logging
import numpy as np
import tensorflow as tf

import config
from models.model_trainer import ModelTrainer # Changed import
from models.model_trainer_utils import train_enhanced_model # Changed import
from xai_cacl_explainer import CACLFeatureExtractor, explain_with_cacl
from cacl_utils import compute_dependency_matrix, compute_context_embeddings

logger = logging.getLogger(__name__)

class EnhancedModelTrainer(ModelTrainer):
    def __init__(self, model, output_dir, use_bayesian=False, use_cacl=False, cacl_model_path=None):
        super().__init__(model, output_dir)
        self.use_bayesian = use_bayesian
        self.use_cacl = use_cacl
        self.cacl_model_path = cacl_model_path
        self.cacl_explainer = None
        self.cacl_dependency_matrix = None
        self.cacl_context_avg_embeddings = None
        self.cacl_ref_ctx_sim_distribution = None

        if self.use_cacl:
            logger.info("Initializing CACL Explainer...")
            # Calculate partitions based on config values
            sample_light_curve_length = config.FIXED_LENGTH
            partition_size = sample_light_curve_length // config.CACL_K_PARTITIONS
            cacl_initial_partitions = []
            cacl_initial_partitions = [[j for j in range(i * partition_size, (i + 1) * partition_size)] for i in range(config.CACL_K_PARTITIONS - 1)]
            cacl_initial_partitions.append([j for j in range((config.CACL_K_PARTITIONS - 1) * partition_size, sample_light_curve_length)])
            cacl_initial_partitions = [p for p in cacl_initial_partitions if p] # Filter out empty partitions

            self.cacl_explainer = CACLFeatureExtractor(
                model_path=self.cacl_model_path,
                partitions=cacl_initial_partitions, # Pass calculated partitions
                input_dim=config.FIXED_LENGTH,
                output_dim=config.CACL_OUTPUT_DIM,
                proj_dim=config.CACL_PROJ_DIM,
                nhead=config.CACL_NHEAD,
                num_layers=config.CACL_NUM_LAYERS
            )
            logger.info("CACL Explainer initialized.")

    def train(self, train_dataset, validation_dataset, epochs=None, batch_size=None):
        # Unpack datasets
        X_ts_train, X_img_train, X_features_train, y_train = train_dataset
        X_ts_val, X_img_val, X_features_val, y_val = validation_dataset

        # Prepare inputs for the multimodal model
        X_train_inputs = [X_img_train, X_ts_train, X_features_train]
        X_val_inputs = [X_img_val, X_ts_val, X_features_val]

        # Call the enhanced training function
        trained_model, history = train_enhanced_model(
            self.model, 
            X_train_inputs, y_train, 
            X_val_inputs, y_val, 
            model_name="exo_multimodal_model", # Hardcoded for now
            epochs=epochs, 
            batch_size=batch_size,
            output_dir=self.output_dir
        )
        self.model = trained_model # Update the model with the trained one
        final_metrics = {key: value[-1] for key, value in history.history.items()}
        return history, y_val, self.model.predict(X_val_inputs), final_metrics # Return final_metrics as cnn_metrics

    def get_cacl_explanations(self, validation_dataset, cacl_partitions=None, validation_original_indices=None, validation_classifications=None):
        if not self.use_cacl or self.cacl_explainer is None:
            logger.warning("CACL is not enabled or explainer not initialized.")
            return {}

        if cacl_partitions is None:
            logger.error("CACL partitions must be provided to get explanations.")
            return {}

        if validation_original_indices is None:
            logger.error("validation_original_indices must be provided to get explanations.")
            return {}

        if validation_classifications is None:
            logger.error("validation_classifications must be provided to get explanations.")
            return {}

        # Update partitions in the explainer
        self.cacl_explainer.partitions = cacl_partitions

        X_ts_val, _, _, y_val = validation_dataset
        explanations = {}

        # Explain all samples in the validation set
        sample_indices_in_val_dataset = range(len(X_ts_val))

        # Pre-compute context embeddings and dependency matrix if not already done
        if self.cacl_dependency_matrix is None or self.cacl_context_avg_embeddings is None:
            logger.info("Pre-computing CACL context embeddings and dependency matrix...")
            # This part needs access to the full training data (X_ts_train) to build a robust context
            # For simplicity, we'll use a subset of validation data for context here, which is not ideal
            # but allows the pipeline to run.
            # A proper implementation would involve passing X_ts_train to this method or storing it.
            
            # Dummy X_inliers for compute_dependency_matrix and compute_context_embeddings
            # In a real scenario, X_inliers would be a representative set of 'normal' data
            X_inliers = X_ts_val # Use the full validation set
            if len(X_inliers) == 0:
                logger.warning("No inlier data available for CACL context computation.")
                return {}

            # Create partitions for the inlier data
            sample_light_curve_length = X_inliers.shape[1]
            partition_size = sample_light_curve_length // config.CACL_K_PARTITIONS
            inlier_partitions = [[j for j in range(i * partition_size, (i + 1) * partition_size)] for i in range(config.CACL_K_PARTITIONS - 1)]
            inlier_partitions.append([j for j in range((config.CACL_K_PARTITIONS - 1) * partition_size, sample_light_curve_length)])
            inlier_partitions = [p for p in inlier_partitions if p] # Filter out empty partitions

            # Update cacl_explainer partitions for context computation
            self.cacl_explainer.partitions = inlier_partitions

            self.cacl_dependency_matrix = compute_dependency_matrix(X_inliers, self.cacl_explainer.model, inlier_partitions, eta=config.CACL_ETA)
            self.cacl_context_avg_embeddings = compute_context_embeddings(self.cacl_explainer.model, X_inliers, inlier_partitions)
            
            # Compute reference context similarity distribution for flagging
            ref_ctx_sim_distribution = []
            for i in range(len(X_inliers)):
                record_embeddings = self.cacl_explainer.get_partition_embeddings(X_inliers[i])
                # Need to build context_indices for each inlier from X_inliers
                # For simplicity, let's assume a dummy context for now or skip context_flag for inliers
                # A proper implementation would involve build_context_groups on X_inliers
                sims = []
                # Dummy context_indices for now
                dummy_context_indices = [j for j in range(len(X_inliers)) if j != i]
                if dummy_context_indices:
                    avg = np.mean(np.stack(record_embeddings, axis=0), axis=0)
                    for j in dummy_context_indices:
                        sims.append(np.dot(avg, self.cacl_context_avg_embeddings[j]) / (np.linalg.norm(avg) * np.linalg.norm(self.cacl_context_avg_embeddings[j]) + 1e-12))
                if sims:
                    ref_ctx_sim_distribution.append(np.mean(sims))
            self.cacl_ref_ctx_sim_distribution = np.array(ref_ctx_sim_distribution) if len(ref_ctx_sim_distribution) > 0 else np.array([0.0])

            logger.info("CACL context embeddings and dependency matrix pre-computed.")

        for i_val in sample_indices_in_val_dataset:
            original_index = validation_original_indices[i_val]
            classification = validation_classifications[i_val]
            logger.info(f"Explaining validation sample {original_index} (index in val dataset: {i_val})...")
            data_sample = X_ts_val[i_val]
            
            # Build context indices for the current sample
            # This requires the full dataset X_ts_val and potentially X_ts_train
            # For simplicity, we'll use a dummy context for now
            context_indices = [j for j in range(len(X_ts_val)) if j != i_val] # Dummy context

            explanation = explain_with_cacl(
                data_sample=data_sample,
                cacl_explainer=self.cacl_explainer,
                dependency_matrix=self.cacl_dependency_matrix,
                context_avg_embeddings=self.cacl_context_avg_embeddings,
                context_indices=context_indices,
                eta=config.CACL_ETA,
                context_threshold_quantile=config.CACL_CONTEXT_THRESHOLD_QUANTILE,
                ref_ctx_sim_distribution=self.cacl_ref_ctx_sim_distribution
            )
            explanation['classification'] = classification
            explanations[original_index] = explanation
            logger.info(f"Explanation for sample {original_index}: {explanation}")

        logger.info("CACL explanations generated.")
        logger.debug(f"CACL Explanations before return: {explanations}")
        return explanations