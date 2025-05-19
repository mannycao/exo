"""
CNN models for exoplanet detection.
"""

import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    Reshape, Input, Conv1D, MaxPooling1D, Concatenate
)

logger = logging.getLogger(__name__)


def build_transit_detection_model(input_shape):
    """
    Creates a convolutional neural network for transit detection.
    
    Args:
        input_shape: Input shape for the model (height, width)
    
    Returns:
        keras.models.Model: Compiled CNN model
    """
    model = Sequential([
        Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def build_characterization_model(input_shape, output_dim=3):
    """
    Creates a model for characterizing exoplanet properties.
    
    Args:
        input_shape: Input shape for the model (height, width)
        output_dim: Number of output dimensions (planet properties)
    
    Returns:
        keras.models.Model: Compiled characterization model
    """
    model = Sequential([
        Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.2),
        # Multiple outputs: radius, period, etc.
        Dense(output_dim, activation='linear')  # For multiple exoplanet properties
    ])
    
    model.compile(
        optimizer='adam', 
        loss='mse', 
        metrics=['mae']
    )
    
    return model


def build_multimodal_model(image_shape, timeseries_shape):
    """
    Creates a multimodal fusion model that combines image and time series data.
    
    Args:
        image_shape: Shape of the image input (height, width)
        timeseries_shape: Shape of the time series input (length,)
    
    Returns:
        keras.models.Model: Compiled multimodal model
    """
    # Image branch
    image_input = Input(shape=(image_shape[0], image_shape[1], 1))
    image_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    image_pool1 = MaxPooling2D((2, 2))(image_conv1)
    image_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(image_pool1)
    image_pool2 = MaxPooling2D((2, 2))(image_conv2)
    image_flat = Flatten()(image_pool2)
    image_dense = Dense(128, activation='relu')(image_flat)
    
    # Time series branch
    timeseries_input = Input(shape=timeseries_shape)
    ts_reshape = Reshape((timeseries_shape[0], 1))(timeseries_input)
    ts_conv1 = Conv1D(32, 5, activation='relu', padding='same')(ts_reshape)
    ts_pool1 = MaxPooling1D(2)(ts_conv1)
    ts_conv2 = Conv1D(64, 3, activation='relu', padding='same')(ts_pool1)
    ts_pool2 = MaxPooling1D(2)(ts_conv2)
    ts_flat = Flatten()(ts_pool2)
    ts_dense = Dense(128, activation='relu')(ts_flat)
    
    # Merge branches
    merged = Concatenate()([image_dense, ts_dense])
    merged_dense1 = Dense(256, activation='relu')(merged)
    merged_drop1 = Dropout(0.5)(merged_dense1)
    merged_dense2 = Dense(128, activation='relu')(merged_drop1)
    merged_drop2 = Dropout(0.3)(merged_dense2)
    output = Dense(1, activation='sigmoid')(merged_drop2)
    
    # Create model
    model = Model(inputs=[image_input, timeseries_input], outputs=output)
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model


def create_ensemble_model(models, weights=None):
    """
    Creates an ensemble model from multiple trained models.
    
    Args:
        models: List of trained models
        weights: List of weights for each model (optional)
    
    Returns:
        keras.models.Model: Compiled ensemble model
    """
    if not models:
        raise ValueError("No models provided for ensemble")
    
    # Default to equal weights if not specified
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    if len(weights) != len(models):
        raise ValueError("Number of weights must match number of models")
    
    # Get the inputs from the first model
    inputs = models[0].inputs
    
    # Get the outputs from all models
    outputs = [model.outputs[0] for model in models]
    
    # Create weighted average
    ensemble_output = outputs[0] * weights[0]
    for i in range(1, len(outputs)):
        ensemble_output = ensemble_output + outputs[i] * weights[i]
    
    # Create the ensemble model
    ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
    ensemble_model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return ensemble_model
