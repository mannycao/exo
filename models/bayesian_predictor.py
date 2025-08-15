# models/bayesian_predictor.py

import numpy as np
import tensorflow as tf
import logging

class BayesianPredictor:
    """
    A class to perform Bayesian inference using Monte Carlo (MC) Dropout.

    This class takes a trained Keras model with Dropout layers and uses them
    to generate a distribution of predictions for a given input. The mean of
    this distribution is used as the final prediction, and the variance is
    used as a measure of the model's uncertainty.
    """
    def __init__(self, model, n_samples=100):
        """
        Initializes the BayesianPredictor.

        Args:
            model (tf.keras.Model): The trained Keras model. It is assumed
                that this model has one or more Dropout layers.
            n_samples (int): The number of stochastic forward passes to perform
                to generate the prediction distribution.
        """
        self.model = model
        self.n_samples = n_samples
        self.logger = logging.getLogger(__name__)

    def predict(self, X_image, X_timeseries):
        """
        Performs prediction with uncertainty estimation using MC Dropout.

        This method runs multiple stochastic forward passes through the network
        with dropout enabled to sample from the approximate posterior distribution.

        Args:
            X_image (np.ndarray): The image part of the input data.
            X_timeseries (np.ndarray): The time-series part of the input data.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The mean of the predictions across all samples.
                              Shape: (n_data_points,).
                - np.ndarray: The variance of the predictions across all samples,
                              representing the model's uncertainty.
                              Shape: (n_data_points,).
        """
        self.logger.info(f"Performing {self.n_samples} stochastic forward passes for uncertainty estimation...")
        
        # Create a list to store the predictions from each forward pass
        predictions_list = []

        # Loop to perform N stochastic forward passes
        for _ in range(self.n_samples):
            try:
                # The key to MC Dropout: set `training=True` during inference
                # to ensure that dropout layers are active.
                predictions = self.model([X_image, X_timeseries], training=True)
                predictions_list.append(predictions)
            except Exception as e:
                self.logger.error(f"Error during a stochastic forward pass: {e}")
                # Depending on the desired behavior, you might want to skip this sample
                # or halt the process. For now, we'll just log it.
                continue
        
        if not predictions_list:
            self.logger.error("No predictions were generated. Aborting.")
            # Return empty arrays with the correct number of dimensions
            return np.array([]), np.array([])

        # Stack the predictions along a new axis to create a [n_samples, n_data_points, 1] tensor
        predictions_stack = tf.stack(predictions_list, axis=0)
        
        # Squeeze the last dimension if it's 1
        if predictions_stack.shape[-1] == 1:
            predictions_stack = tf.squeeze(predictions_stack, axis=-1)

        # Calculate the mean across the samples (axis=0) to get the final prediction
        y_pred_mean = tf.reduce_mean(predictions_stack, axis=0).numpy()
        
        # Calculate the variance across the samples (axis=0) to get the uncertainty
        y_pred_uncertainty = tf.math.reduce_variance(predictions_stack, axis=0).numpy()

        self.logger.info("Uncertainty estimation complete.")
        return y_pred_mean, y_pred_uncertainty
