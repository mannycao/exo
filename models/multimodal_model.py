import logging
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Reshape, Conv1D, MaxPooling1D, Concatenate, BatchNormalization
)
from tensorflow.keras.regularizers import l2

# --- DR. V FIX: Robust Import Strategy ---
# This ensures we find 'config.py' whether running as a module, script, or from root.
try:
    import config as lite_config
except ImportError:
    try:
        from lite import config as lite_config
    except ImportError:
        # If running from inside models/ directory, look one level up
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        try:
            import config as lite_config
        except ImportError:
             # If running from project root but 'lite' isn't a package yet
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from lite import config as lite_config
# -----------------------------------------

logger = logging.getLogger(__name__)

def build_transformer_classifier_model(encoder_model: Model, num_classes=1):
    """
    Creates a classifier model using a pre-trained KerasTransformerEncoder.
    The encoder's output is flattened and passed through a dense classification head.
    """
    # Use config for feature length to avoid hardcoding
    input_tensor = Input(shape=(lite_config.FEATURE_VECTOR_LENGTH,), name='classifier_input')
    
    # Use the encoder
    x = encoder_model(input_tensor)
    
    # Classification head
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='sigmoid', name='classifier_output')(x)
    
    model = Model(inputs=input_tensor, outputs=output, name="transformer_classifier")
    logger.info("Transformer-based classifier model built.")
    # model.summary(print_fn=logger.info) # Optional: keep log clean
    return model

def build_multimodal_fusion_model(image_shape, timeseries_shape, feature_shape, num_classes=1):
    """
    Creates a more complex, three-branch multimodal fusion model.
    """
    # Constants from config or defaults
    L2_REG = 1e-4
    DROPOUT_RATE = 0.5
    DENSE_NEURONS = 256

    # --- Image branch (2D CNN) ---
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

    # --- Time-series branch (1D CNN) ---
    ts_input = Input(shape=timeseries_shape, name='timeseries_input')
    x_ts = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(ts_input)
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
    x = Dense(DENSE_NEURONS, activation='relu', kernel_regularizer=l2(L2_REG))(merged)
    x = Dropout(DROPOUT_RATE)(x)
    output = Dense(num_classes, activation='sigmoid', name='output')(x)

    model = Model(inputs=[image_input, ts_input, feature_input], outputs=output, name="multimodal_fusion")
    logger.info("Multimodal fusion model built.")
    return model
