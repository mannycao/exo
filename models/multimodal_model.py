"""
Multimodal fusion models for exoplanet detection.
"""

import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    Reshape, Conv1D, MaxPooling1D, Concatenate, BatchNormalization
)

logger = logging.getLogger(__name__)


def build_multimodal_fusion_model(image_shape, timeseries_shape, num_classes=1):
    """
    Creates a simplified, more robust multimodal fusion model.
    """
    # --- Image branch (CNN) ---
    image_input = Input(shape=image_shape, name='image_input')
    
    x_img = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    
    x_img = Conv2D(32, (3, 3), activation='relu', padding='same')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    
    x_img = Flatten()(x_img)
    x_img = Dense(64, activation='relu')(x_img)
    x_img = Dropout(0.5)(x_img)
    
    # --- Time series branch (1D CNN) ---
    ts_input = Input(shape=timeseries_shape, name='timeseries_input')
    
    # Reshape for 1D convolution, assuming shape is (length, 1)
    x_ts = Reshape((timeseries_shape[0], 1))(ts_input)
    
    x_ts = Conv1D(16, 5, activation='relu', padding='same')(x_ts)
    x_ts = BatchNormalization()(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    
    x_ts = Conv1D(32, 5, activation='relu', padding='same')(x_ts)
    x_ts = BatchNormalization()(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    
    x_ts = Flatten()(x_ts)
    x_ts = Dense(64, activation='relu')(x_ts)
    x_ts = Dropout(0.5)(x_ts)
    
    # --- Merge branches ---
    merged = Concatenate()([x_img, x_ts])
    merged_dense = Dense(128, activation='relu')(merged)
    merged_dense = BatchNormalization()(merged_dense)
    merged_dense = Dropout(0.5)(merged_dense)
    
    # --- Output Layer ---
    output = Dense(1, activation='sigmoid', name='output')(merged_dense)
    
    # Create and compile model
    model = Model(inputs=[image_input, ts_input], outputs=output)
    
    # The compilation will happen in the model_trainer
    logger.info("Multimodal model built successfully.")
    model.summary(print_fn=logger.info)
    
    return model