# models/multimodal_model.py

import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    Reshape, Conv1D, MaxPooling1D, Concatenate, BatchNormalization
)
from tensorflow.keras.regularizers import l2

logger = logging.getLogger(__name__)

def build_multimodal_fusion_model(image_shape, timeseries_shape, feature_shape, num_classes=1):
    """
    Creates a more complex, three-branch multimodal fusion model.
    """
    # --- Architectural Hyperparameters ---
    DENSE_NEURONS = 128
    DROPOUT_RATE = 0.3
    L2_REG = 1e-5
    # ---

    # --- Image branch (CNN) ---
    image_input = Input(shape=image_shape, name='image_input')
    x_img = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(image_input)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D((2, 2))(x_img)
    x_img = Flatten()(x_img)
    x_img = Dense(DENSE_NEURONS // 2, activation='relu', kernel_regularizer=l2(L2_REG))(x_img)
    x_img = Dropout(DROPOUT_RATE)(x_img)
    
    # --- Time series branch (1D CNN) ---
    ts_input = Input(shape=timeseries_shape, name='timeseries_input')
    x_ts = Reshape((timeseries_shape[0], 1))(ts_input)
    x_ts = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_ts)
    x_ts = BatchNormalization()(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_ts)
    x_ts = BatchNormalization()(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_ts)
    x_ts = BatchNormalization()(x_ts)
    x_ts = MaxPooling1D(2)(x_ts)
    x_ts = Flatten()(x_ts)
    x_ts = Dense(DENSE_NEURONS // 2, activation='relu', kernel_regularizer=l2(L2_REG))(x_ts)
    x_ts = Dropout(DROPOUT_RATE)(x_ts)

    # --- Feature branch (Dense) ---
    feature_input = Input(shape=feature_shape, name='feature_input')
    x_feat = Dense(DENSE_NEURONS // 4, activation='relu', kernel_regularizer=l2(L2_REG))(feature_input)
    x_feat = Dropout(DROPOUT_RATE)(x_feat)
    
    # --- Merge branches ---
    merged = Concatenate()([x_img, x_ts, x_feat])
    merged_dense = Dense(DENSE_NEURONS, activation='relu', kernel_regularizer=l2(L2_REG))(merged)
    merged_dense = BatchNormalization()(merged_dense)
    merged_dense = Dropout(DROPOUT_RATE)(merged_dense)
    
    # --- Output Layer ---
    output = Dense(1, activation='sigmoid', name='output')(merged_dense)
    
    model = Model(inputs=[image_input, ts_input, feature_input], outputs=output)
    
    logger.info("Three-branch multimodal model built.")
    model.summary(print_fn=logger.info)
    
    return model