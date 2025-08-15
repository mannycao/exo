"""
Robust Fusion Model - Simplified architecture to prevent overfitting
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dropout, Dense,
    Conv2D, MaxPooling2D, Concatenate, LSTM, BatchNormalization, Multiply,
    GlobalAveragePooling2D, LayerNormalization, GlobalAveragePooling1D
)
import numpy as np


def create_robust_fusion_model(cfg, num_train_samples=1000, ttv_dim=7):
    """
    Creates a simplified, regularized fusion model to prevent overfitting.
    
    Key improvements:
    - Reduced model complexity
    - Aggressive regularization
    - Simpler architecture
    - Proper normalization
    """
    
    # --- Branch 1: Transit Photometry (Simplified CNN) ---
    transit_image_input = Input(shape=cfg.model.transit_image_shape, name='transit_image_input', dtype=tf.float32)
    transit_ts_input = Input(shape=cfg.model.transit_ts_shape, name='transit_ts_input', dtype=tf.float32)

    # Simplified timeseries branch
    x1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(transit_ts_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(0.4)(x1)
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)
    x1 = Dropout(0.5)(x1)

    # Simplified image branch
    x2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(transit_image_input)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.4)(x2)
    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dropout(0.5)(x2)
    
    transit_features = Concatenate(name='transit_features')([x1, x2])

    # --- Branch 2: Radial Velocity (Simplified LSTM) ---
    rv_input = Input(shape=cfg.model.rv_shape, name='rv_input', dtype=tf.float32)
    rv_mask_input = Input(shape=(1,), name='rv_mask_input', dtype=tf.float32)
    
    x3 = LSTM(16, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(rv_input)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.5)(x3)
    # Apply mask to zero-out features if data is missing
    rv_features = Multiply()([x3, rv_mask_input])

    # --- Branch 3: High-Resolution Imaging (Simplified Dense) ---
    imaging_input = Input(shape=(2,), name='imaging_input', dtype=tf.float32)
    imaging_mask_input = Input(shape=(1,), name='imaging_mask_input', dtype=tf.float32)
    
    x4 = Dense(8, activation='relu')(imaging_input)
    x4 = BatchNormalization()(x4)
    x4 = Dropout(0.5)(x4)
    x4_features = Dense(4, activation='relu')(x4)
    # Apply mask to zero-out features if data is missing
    imaging_features = Multiply()([x4_features, imaging_mask_input])

    # --- Branch 4: TTV Features (Simplified) ---
    ttv_input = Input(shape=(ttv_dim,), name='ttv_input', dtype=tf.float32)
    ttv_features = Dense(8, activation='relu')(ttv_input)
    ttv_features = BatchNormalization()(ttv_features)
    ttv_features = Dropout(0.5)(ttv_features)

    # --- Fusion Layer (Simplified) ---
    fused = Concatenate(name='fused_features')([
        transit_features, 
        rv_features, 
        imaging_features, 
        ttv_features
    ])
    
    # Simplified classification head with heavy regularization
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(fused)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    
    x = Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    
    output = Dense(1, activation='sigmoid')(x)

    # Create model
    inputs = {
        'transit_image_input': transit_image_input,
        'transit_ts_input': transit_ts_input,
        'rv_input': rv_input,
        'imaging_input': imaging_input,
        'rv_mask_input': rv_mask_input,
        'imaging_mask_input': imaging_mask_input,
        'ttv_input': ttv_input
    }
    
    model = Model(inputs=inputs, outputs=output, name="RobustFusionModel")
    
    return model
