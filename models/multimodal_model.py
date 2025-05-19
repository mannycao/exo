"""
Multimodal fusion models for exoplanet detection.
"""

import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    Reshape, Input, Conv1D, MaxPooling1D, Concatenate
)

logger = logging.getLogger(__name__)


def build_multimodal_fusion_model(image_shape, timeseries_shape, num_classes=1):
    """
    Creates a multimodal fusion model that combines image and time series data.
    
    Args:
        image_shape: Shape of the image input (height, width)
        timeseries_shape: Shape of the time series input (length,)
        num_classes: Number of output classes (1 for binary classification)
    
    Returns:
        keras.models.Model: Compiled multimodal fusion model
    """
    # Image branch - CNN
    image_input = Input(shape=(image_shape[0], image_shape[1], 1))
    
    x_img = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', padding='same')(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', padding='same')(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Flatten()(x_img)
    x_img = Dense(128, activation='relu')(x_img)
    
    # Time series branch - 1D CNN
    ts_input = Input(shape=(timeseries_shape,))
    
    # Reshape for 1D convolution
    if isinstance(timeseries_shape, tuple):
        x_ts = Reshape((*timeseries_shape, 1))(ts_input)
    else:
        x_ts = Reshape((timeseries_shape, 1))(ts_input)
    
    x_ts = Conv1D(32, 5, activation='relu', padding='same')(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Conv1D(64, 3, activation='relu', padding='same')(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Flatten()(x_ts)
    x_ts = Dense(128, activation='relu')(x_ts)
    
    # Merge branches
    merged = Concatenate()([x_img, x_ts])
    merged_dense1 = Dense(256, activation='relu')(merged)
    merged_drop1 = Dropout(0.5)(merged_dense1)
    merged_dense2 = Dense(128, activation='relu')(merged_drop1)
    merged_drop2 = Dropout(0.3)(merged_dense2)
    
    # Output layer
    if num_classes == 1:
        output = Dense(1, activation='sigmoid')(merged_drop2)
    else:
        output = Dense(num_classes, activation='softmax')(merged_drop2)
    
    # Create model
    model = Model(inputs=[image_input, ts_input], outputs=output)
    
    # Compile model
    if num_classes == 1:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model


def build_ensemble_multimodal_model(image_shape, timeseries_shape, num_classes=1):
    """
    Creates an ensemble of multimodal models with different architectures.
    
    Args:
        image_shape: Shape of the image input (height, width)
        timeseries_shape: Shape of the time series input (length,)
        num_classes: Number of output classes (1 for binary classification)
    
    Returns:
        list: List of compiled multimodal models
    """
    models = []
    
    # Model 1: Standard fusion model
    models.append(build_multimodal_fusion_model(image_shape, timeseries_shape, num_classes))
    
    # Model 2: Deeper CNN branch
    # Image branch
    image_input = Input(shape=(image_shape[0], image_shape[1], 1))
    
    x_img = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x_img = Conv2D(32, (3, 3), activation='relu', padding='same')(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    
    x_img = Conv2D(64, (3, 3), activation='relu', padding='same')(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', padding='same')(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    
    x_img = Conv2D(128, (3, 3), activation='relu', padding='same')(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', padding='same')(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    
    x_img = Flatten()(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = Dropout(0.4)(x_img)
    
    # Time series branch
    ts_input = Input(shape=(timeseries_shape,))
    
    # Reshape for 1D convolution
    if isinstance(timeseries_shape, tuple):
        x_ts = Reshape((*timeseries_shape, 1))(ts_input)
    else:
        x_ts = Reshape((timeseries_shape, 1))(ts_input)
    
    x_ts = Conv1D(32, 5, activation='relu', padding='same')(x_ts)
    x_ts = Conv1D(32, 5, activation='relu', padding='same')(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    
    x_ts = Conv1D(64, 3, activation='relu', padding='same')(x_ts)
    x_ts = Conv1D(64, 3, activation='relu', padding='same')(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    
    x_ts = Flatten()(x_ts)
    x_ts = Dense(128, activation='relu')(x_ts)
    x_ts = Dropout(0.3)(x_ts)
    
    # Merge branches
    merged = Concatenate()([x_img, x_ts])
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.4)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    
    # Output layer
    if num_classes == 1:
        output = Dense(1, activation='sigmoid')(merged)
    else:
        output = Dense(num_classes, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[image_input, ts_input], outputs=output)
    
    # Compile model
    if num_classes == 1:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    models.append(model)
    
    # Model 3: Different architecture with additional layers
    # Image branch
    image_input = Input(shape=(image_shape[0], image_shape[1], 1))
    
    x_img = Conv2D(32, (5, 5), activation='relu', padding='same')(image_input)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', padding='same')(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', padding='same')(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Flatten()(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = Dropout(0.5)(x_img)
    x_img = Dense(128, activation='relu')(x_img)
    
    # Time series branch
    ts_input = Input(shape=(timeseries_shape,))
    
    # Reshape for 1D convolution
    if isinstance(timeseries_shape, tuple):
        x_ts = Reshape((*timeseries_shape, 1))(ts_input)
    else:
        x_ts = Reshape((timeseries_shape, 1))(ts_input)
    
    x_ts = Conv1D(32, 7, activation='relu', padding='same')(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Conv1D(64, 5, activation='relu', padding='same')(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Conv1D(128, 3, activation='relu', padding='same')(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Flatten()(x_ts)
    x_ts = Dense(128, activation='relu')(x_ts)
    x_ts = Dropout(0.4)(x_ts)
    
    # Merge branches
    merged = Concatenate()([x_img, x_ts])
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.4)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    
    # Output layer
    if num_classes == 1:
        output = Dense(1, activation='sigmoid')(merged)
    else:
        output = Dense(num_classes, activation='softmax')(merged)
    
    # Create model
    model = Model(inputs=[image_input, ts_input], outputs=output)
    
    # Compile model
    if num_classes == 1:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    models.append(model)
    
    return models


def combine_ensemble_predictions(predictions, method='average'):
    """
    Combine predictions from multiple models in an ensemble.
    
    Args:
        predictions: List of prediction arrays from different models
        method: Method to combine predictions ('average', 'max', or 'vote')
        
    Returns:
        numpy.ndarray: Combined predictions
    """
    import numpy as np
    
    if method == 'average':
        # Simple averaging of probabilities
        return np.mean(predictions, axis=0)
    elif method == 'max':
        # Take maximum probability
        return np.max(predictions, axis=0)
    elif method == 'vote':
        # For binary classification, convert to binary and take majority vote
        binary_preds = [pred > 0.5 for pred in predictions]
        return np.mean(binary_preds, axis=0) > 0.5
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def optimize_ensemble_weights(ensemble_models, X_val, y_val):
    """
    Optimize weights for ensemble model predictions to maximize performance.
    
    Args:
        ensemble_models: List of trained models
        X_val: Validation data (image, time series)
        y_val: Validation labels
    
    Returns:
        list: Optimized weights for each model
    """
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss, average_precision_score
    
    def objective(weights, models, X, y):
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Get predictions from each model
        preds = []
        for model in models:
            pred = model.predict(X)
            preds.append(pred)
        
        # Compute weighted predictions
        weighted_preds = np.zeros_like(preds[0])
        for i, pred in enumerate(preds):
            weighted_preds += weights[i] * pred
        
        # Optimize for negative average precision (to minimize)
        return -average_precision_score(y, weighted_preds)
    
    # Initial weights (equal)
    initial_weights = np.ones(len(ensemble_models)) / len(ensemble_models)
    
    # Constraints: weights must sum to 1 and be non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(ensemble_models))]
    
    # Run optimization
    result = minimize(
        objective, 
        initial_weights, 
        args=(ensemble_models, X_val, y_val),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Get optimized weights
    optimized_weights = result.x
    optimized_weights = optimized_weights / np.sum(optimized_weights)
    
    return optimized_weights


def combine_ensemble_predictions_weighted(predictions, weights=None, method='average'):
    """
    Combine predictions from multiple models in an ensemble using weighted averaging.
    
    Args:
        predictions: List of prediction arrays from different models
        weights: Weights for each model's predictions (optional)
        method: Method to combine predictions ('average', 'max', 'vote', or 'weighted')
        
    Returns:
        numpy.ndarray: Combined predictions
    """
    import numpy as np
    
    if method == 'weighted' and weights is not None:
        # Ensure weights sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Initialize with zeros
        weighted_preds = np.zeros_like(predictions[0])
        
        # Apply weights to each model's predictions
        for i, pred in enumerate(predictions):
            weighted_preds += weights[i] * pred
            
        return weighted_preds
    
    elif method == 'average':
        # Simple averaging of probabilities
        return np.mean(predictions, axis=0)
    
    elif method == 'max':
        # Take maximum probability
        return np.max(predictions, axis=0)
    
    elif method == 'vote':
        # For binary classification, convert to binary and take majority vote
        binary_preds = [pred > 0.5 for pred in predictions]
        return np.mean(binary_preds, axis=0) > 0.5
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def create_dual_threshold_predictions(ensemble_preds, high_precision_threshold=0.8, high_recall_threshold=0.3):
    """
    Create dual predictions with high precision and high recall thresholds.
    
    Args:
        ensemble_preds: Predicted probabilities from ensemble
        high_precision_threshold: Threshold for high precision predictions
        high_recall_threshold: Threshold for high recall predictions
    
    Returns:
        tuple: (high_precision_preds, high_recall_preds, confidence_level)
    """
    import numpy as np
    
    # High precision predictions (conservative)
    high_precision_preds = (ensemble_preds > high_precision_threshold).astype(int)
    
    # High recall predictions (more inclusive)
    high_recall_preds = (ensemble_preds > high_recall_threshold).astype(int)
    
    # Confidence level based on prediction probability
    confidence_level = np.ones_like(ensemble_preds)
    
    # Very high confidence: above high precision threshold
    confidence_level[ensemble_preds > high_precision_threshold] = 3
    
    # Medium confidence: between thresholds
    medium_mask = (ensemble_preds <= high_precision_threshold) & (ensemble_preds > high_recall_threshold)
    confidence_level[medium_mask] = 2
    
    # Low confidence: below both thresholds but not extremely low
    low_mask = (ensemble_preds <= high_recall_threshold) & (ensemble_preds > 0.1)
    confidence_level[low_mask] = 1
    
    # Very low confidence: extremely low probability
    confidence_level[ensemble_preds <= 0.1] = 0
    
    return high_precision_preds, high_recall_preds, confidence_level
