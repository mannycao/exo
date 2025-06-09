"""
Defines the Convolutional Neural Network (CNN) model for transit detection
from 2D image representations of light curve segments.
"""

import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
)

logger = logging.getLogger(__name__)

def build_transit_detection_model(input_shape):
    """
    Builds a more robust, deeper CNN model for classifying transit images.

    This architecture uses a standard pattern of stacking convolutional blocks,
    each containing Conv2D, Activation, Batch Normalization, and MaxPooling,
    followed by Dense layers for classification. This version is deeper to
    increase model capacity.

    Args:
        input_shape (tuple): The shape of the input images (e.g., (64, 64, 1)).

    Returns:
        tensorflow.keras.models.Model: The Keras model object.
    """
    if not isinstance(input_shape, tuple) or len(input_shape) < 2:
        raise ValueError(f"input_shape must be a tuple of length 2 or 3, but got {input_shape}")

    # If shape is (H, W), ensure a channel dimension exists
    if len(input_shape) == 2:
        input_shape = (*input_shape, 1)

    logger.info(f"Building Deep CNN model with input shape: {input_shape}")
    
    model = Sequential([
        Input(shape=input_shape),
        
        # --- Convolutional Block 1 ---
        Conv2D(32, (3, 3), padding='same', name='conv1a'),
        Activation('relu', name='act1a'),
        BatchNormalization(name='bn1a'),
        Conv2D(32, (3, 3), padding='same', name='conv1b'),
        Activation('relu', name='act1b'),
        BatchNormalization(name='bn1b'),
        MaxPooling2D(pool_size=(2, 2), name='pool1'),
        
        # --- Convolutional Block 2 ---
        Conv2D(64, (3, 3), padding='same', name='conv2a'),
        Activation('relu', name='act2a'),
        BatchNormalization(name='bn2a'),
        Conv2D(64, (3, 3), padding='same', name='conv2b'),
        Activation('relu', name='act2b'),
        BatchNormalization(name='bn2b'),
        MaxPooling2D(pool_size=(2, 2), name='pool2'),
        
        # --- Convolutional Block 3 ---
        Conv2D(128, (3, 3), padding='same', name='conv3a'),
        Activation('relu', name='act3a'),
        BatchNormalization(name='bn3a'),
        Conv2D(128, (3, 3), padding='same', name='conv3b'),
        Activation('relu', name='act3b'),
        BatchNormalization(name='bn3b'),
        MaxPooling2D(pool_size=(2, 2), name='pool3'),
        
        # --- Flatten and Dense Layers ---
        Flatten(name='flatten'),
        
        # --- Dense Block 1 ---
        Dense(512, name='dense1'), # Increased capacity
        Activation('relu', name='act_dense1'),
        BatchNormalization(name='bn_dense1'),
        Dropout(0.5, name='dropout1'), 
        
        # --- Dense Block 2 ---
        Dense(256, name='dense2'),
        Activation('relu', name='act_dense2'),
        BatchNormalization(name='bn_dense2'),
        Dropout(0.5, name='dropout2'),
        
        # --- Output Layer ---
        Dense(1, activation='sigmoid', name='output')
    ])
    
    model.summary(print_fn=logger.info)
    
    return model
