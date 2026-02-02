import logging
import numpy as np
import tensorflow as tf
from typing import List
from scipy.stats import entropy

# Import existing components from the 'lite' codebase
from models.bayesian_predictor import BayesianPredictor
import config as lite_config

# By importing this, we make Keras aware of the custom object,
# which is necessary for loading the model.
from convert_cacl_to_keras import KerasTransformerEncoder
from models.multimodal_model import build_transformer_classifier_model


# Import shared types
from .types import TransitHypothesis, PipelineStatus

logger = logging.getLogger(__name__)

def _calculate_predictive_entropy(predictions: np.ndarray) -> float:
    """Calculates the predictive entropy of a distribution."""
    # The distribution is of Bernoulli trials (planet or not), so we use the
    # entropy formula for a discrete distribution with two outcomes.
    # For a single data point, the 'distribution' is the set of MC samples.
    # We take the mean probability as the parameter p for the Bernoulli distribution.
    p = np.mean(predictions)
    if p == 0 or p == 1:
        return 0.0
    return entropy([p, 1 - p], base=2)


class BayesianWrapper:
    """
    A wrapper for the BayesianPredictor that integrates with the
    discovery pipeline's data structures.
    """
    def __init__(self, model_path: str):
        """
        Initializes the wrapper and loads the underlying Keras model.

        Args:
            model_path (str): Path to the saved KerasTransformerEncoder model file.
        """
        try:
            self.logger = logging.getLogger(__name__)
            
            # Load the KerasTransformerEncoder (the base feature extractor)
            custom_objects = {"KerasTransformerEncoder": KerasTransformerEncoder}
            encoder_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            # Now build the full classifier model on top of the encoder
            # We need to know the output shape of the encoder. KerasTransformerEncoder's call returns [B, D]
            # where D is head_size * num_heads. From convert_cacl_to_keras.py, output_dim is 64,
            # which is (head_size * num_heads) if num_heads=2 and head_size=32 or num_heads=1 and head_size=64.
            # Assuming the encoder output is 64 for simplicity here.
            # The input shape to the classifier model should be the shape of the features (512,)
            full_classifier_model = build_transformer_classifier_model(
                encoder_model=encoder_model, 
                # input_shape=(lite_config.FEATURE_VECTOR_LENGTH,) # Removed, as build_transformer_classifier_model handles this
            )
            
            self.predictor = BayesianPredictor(full_classifier_model, n_samples=100)
            logger.info(f"BayesianWrapper initialized with full classifier model using encoder from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model or initialize BayesianPredictor: {e}", exc_info=True)
            raise

    def predict(self, hypothesis: TransitHypothesis) -> TransitHypothesis:
        """
        Runs Bayesian inference on a single TransitHypothesis.

        Args:
            hypothesis (TransitHypothesis): The hypothesis to analyze.

        Returns:
            The updated TransitHypothesis with inference results.
        """
        # The model expects a single 'global_view'
        if 'global_view' not in hypothesis.data_views:
            hypothesis.log("ERROR: Missing 'global_view' in data_views for Bayesian prediction.")
            hypothesis.status = PipelineStatus.REJECTED
            return hypothesis
            
        # The Keras model expects a batch dimension, so we expand dimensions
        X_input = np.expand_dims(hypothesis.data_views['global_view'], axis=0)

        try:
            # The `predict` method now returns mean, variance, and the full prediction stack
            posterior_mean_array, epistemic_uncertainty_array, posterior_distribution_array = self.predictor.predict(X_input)

            logger.debug(f"Posterior mean array shape: {posterior_mean_array.shape}, content: {posterior_mean_array}")
            logger.debug(f"Epistemic uncertainty array shape: {epistemic_uncertainty_array.shape}, content: {epistemic_uncertainty_array}")

            # The predictor returns arrays; we extract the first (and only) element for a single hypothesis
            posterior_mean = posterior_mean_array.item() if posterior_mean_array.size > 0 else 0.0
            epistemic_uncertainty = epistemic_uncertainty_array.item() if epistemic_uncertainty_array.size > 0 else 0.0
            
            # Calculate entropy from the full distribution
            # posterior_distribution_array should be (n_samples, 1) or (n_samples,)
            # We want the distribution for the single prediction (if it's not already squeezed)
            if posterior_distribution_array.ndim == 2 and posterior_distribution_array.shape[1] == 1:
                single_prediction_dist = posterior_distribution_array.flatten()
            elif posterior_distribution_array.ndim == 1:
                single_prediction_dist = posterior_distribution_array
            else:
                single_prediction_dist = np.array([posterior_mean]) # Fallback
                logger.warning(f"Unexpected shape for posterior_distribution_array: {posterior_distribution_array.shape}. Using posterior_mean for entropy.")

            predictive_entropy = _calculate_predictive_entropy(single_prediction_dist)
            
            # === CACL DISAGREEMENT SCORE ===
            # Calculate the disagreement between different model views/partitions.
            # This simulates the CACL concept of measuring agreement between
            # temporal and spatial (or other decomposed) views of the data.
            disagreement_score = self._calculate_cacl_disagreement(single_prediction_dist)
            
            # Store the enriched inference data
            hypothesis.inference['posterior_mean'] = posterior_mean
            hypothesis.inference['epistemic_uncertainty'] = epistemic_uncertainty
            hypothesis.inference['predictive_entropy'] = predictive_entropy
            # Convert to list for JSON serialization if needed later
            hypothesis.inference['posterior_distribution'] = single_prediction_dist.tolist()
            
            # CACL-derived metrics
            hypothesis.inference['cacl_disagreement_score'] = disagreement_score
            hypothesis.inference['modality_disagreement'] = disagreement_score
            
            # === GENERATE RATIONALE ===
            rationale = self._generate_rationale(
                posterior_mean=posterior_mean,
                epistemic_uncertainty=epistemic_uncertainty,
                disagreement_score=disagreement_score,
                predictive_entropy=predictive_entropy
            )
            hypothesis.inference['rationale'] = rationale
            
            hypothesis.log(f"Inference complete: posterior_mean={posterior_mean:.4f}, uncertainty={epistemic_uncertainty:.4f}, CACL disagreement={disagreement_score:.4f}")
            hypothesis.log(f"CACL Rationale: {rationale}")

        except Exception as e:
            hypothesis.log(f"ERROR: Bayesian prediction failed: {e}")
            hypothesis.status = PipelineStatus.REJECTED
            logger.error(f"Failed to run prediction for {hypothesis.id}: {e}", exc_info=True)

        return hypothesis
    
    def _calculate_cacl_disagreement(self, prediction_distribution: np.ndarray) -> float:
        """
        Calculate CACL disagreement score from the prediction distribution.
        
        The disagreement score measures the variance within the Monte Carlo samples,
        representing uncertainty about whether the model views agree on the prediction.
        
        In a full CACL implementation with separate temporal/spatial heads:
          disagreement = |p_temporal - p_spatial|
        
        For now, we use the variance of the MC samples as a proxy for disagreement.
        
        Args:
            prediction_distribution: Array of MC samples.
        
        Returns:
            Disagreement score between 0 (perfect agreement) and 1 (maximum disagreement).
        """
        if len(prediction_distribution) < 2:
            return 0.0
        
        # Calculate variance as a proxy for disagreement
        variance = np.var(prediction_distribution)
        
        # Normalize to [0, 1] range (assuming max variance is 0.25 for a Bernoulli)
        disagreement = min(variance / 0.25, 1.0)
        
        logger.debug(f"CACL disagreement score: {disagreement:.4f} (variance: {variance:.4f})")
        return float(disagreement)
    
    def _generate_rationale(
        self,
        posterior_mean: float,
        epistemic_uncertainty: float,
        disagreement_score: float,
        predictive_entropy: float,
        conf_threshold: float = 0.8,
        uncert_threshold: float = 0.2,
        disagree_threshold: float = 0.15
    ) -> str:
        """
        Generate a human-readable rationale for the ML decision.
        
        This combines CACL metrics and Bayesian statistics to explain why
        the model made its prediction, readable by scientists.
        
        Args:
            posterior_mean: The predicted probability (0-1).
            epistemic_uncertainty: Model uncertainty (0-1).
            disagreement_score: CACL disagreement between model views (0-1).
            predictive_entropy: Shannon entropy of predictions (bits).
            conf_threshold: Confidence threshold for "high confidence".
            uncert_threshold: Threshold for "low uncertainty".
            disagree_threshold: Threshold for "strong CACL agreement".
        
        Returns:
            A human-readable rationale string.
        """
        rationale_parts = []
        
        # Confidence component
        if posterior_mean >= conf_threshold:
            rationale_parts.append(f"ML Confidence HIGH ({posterior_mean:.3f} ≥ {conf_threshold})")
        else:
            rationale_parts.append(f"ML Confidence LOW ({posterior_mean:.3f} < {conf_threshold})")
        
        # Uncertainty component
        if epistemic_uncertainty <= uncert_threshold:
            rationale_parts.append(f"Epistemic Uncertainty LOW ({epistemic_uncertainty:.3f} ≤ {uncert_threshold})")
        else:
            rationale_parts.append(f"Epistemic Uncertainty HIGH ({epistemic_uncertainty:.3f} > {uncert_threshold})")
        
        # CACL disagreement component
        if disagreement_score <= disagree_threshold:
            rationale_parts.append(f"CACL Agreement STRONG (Disagreement {disagreement_score:.3f} ≤ {disagree_threshold})")
        else:
            rationale_parts.append(f"CACL Agreement WEAK (Disagreement {disagreement_score:.3f} > {disagree_threshold})")
        
        # Entropy component (higher entropy = more uncertainty)
        if predictive_entropy < 0.5:
            rationale_parts.append(f"Predictions SHARP (Entropy {predictive_entropy:.2f} bits)")
        else:
            rationale_parts.append(f"Predictions DIFFUSE (Entropy {predictive_entropy:.2f} bits)")
        
        rationale = " | ".join(rationale_parts)
        logger.debug(f"Generated CACL rationale: {rationale}")
        
        return rationale
