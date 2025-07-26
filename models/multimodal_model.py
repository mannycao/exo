# models/multimodal_model.py

import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    Reshape, Conv1D, MaxPooling1D, Concatenate, BatchNormalization
)

logger = logging.getLogger(__name__)

def build_multimodal_fusion_model(image_shape, timeseries_shape, num_classes=1):
    """
    Creates a simplified, more robust multimodal fusion model
    using the optimized hyperparameters.
    """
    # --- Architectural Hyperparameters from Tuning ---
    DENSE_NEURONS = 256
    DROPOUT_RATE = 0.4
    # ---

    # --- Image branch (CNN) ---
    image_input = Input(shape=image_shape, name='image_input')
    
    x_img = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    
    x_img = Conv2D(32, (3, 3), activation='relu', padding='same')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    
    x_img = Flatten()(x_img)
    x_img = Dense(DENSE_NEURONS // 2, activation='relu')(x_img) # Using half for this branch
    x_img = Dropout(DROPOUT_RATE)(x_img)
    
    # --- Time series branch (1D CNN) ---
    ts_input = Input(shape=timeseries_shape, name='timeseries_input')
    
    x_ts = Reshape((timeseries_shape[0], 1))(ts_input)
    
    x_ts = Conv1D(16, 5, activation='relu', padding='same')(x_ts)
    x_ts = BatchNormalization()(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    
    x_ts = Conv1D(32, 5, activation='relu', padding='same')(x_ts)
    x_ts = BatchNormalization()(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    
    x_ts = Flatten()(x_ts)
    x_ts = Dense(DENSE_NEURONS // 2, activation='relu')(x_ts) # Using half for this branch
    x_ts = Dropout(DROPOUT_RATE)(x_ts)
    
    # --- Merge branches ---
    merged = Concatenate()([x_img, x_ts])
    merged_dense = Dense(DENSE_NEURONS, activation='relu')(merged) # Full dense layer after merge
    merged_dense = BatchNormalization()(merged_dense)
    merged_dense = Dropout(DROPOUT_RATE)(merged_dense)
    
    # --- Output Layer ---
    output = Dense(1, activation='sigmoid', name='output')(merged_dense)
    
    model = Model(inputs=[image_input, ts_input], outputs=output)
    
    logger.info("Multimodal model built with optimized parameters.")
    model.summary(print_fn=logger.info)
    
    return model