# models/eo_sar_model.py

import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    Concatenate, BatchNormalization
)
from tensorflow.keras.regularizers import l2
import numpy as np

logger = logging.getLogger(__name__)

def build_multimodal_fusion_model(eo_shape, sar_shape, num_classes=1, return_branches=False):
    """
    Creates a two-branch multimodal fusion model for EO and SAR imagery.
    """
    # --- Architectural Hyperparameters ---
    DENSE_NEURONS = 128
    DROPOUT_RATE = 0.3
    L2_REG = 1e-5
    # ---

    # --- EO branch (Optical RGB CNN) ---
    eo_input = Input(shape=eo_shape, name='eo_input')
    x_eo = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(eo_input)
    x_eo = BatchNormalization()(x_eo)
    x_eo = MaxPooling2D((2, 2))(x_eo)
    x_eo = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_eo)
    x_eo = BatchNormalization()(x_eo)
    x_eo = MaxPooling2D((2, 2))(x_eo)
    x_eo = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_eo)
    x_eo = BatchNormalization()(x_eo)
    x_eo = MaxPooling2D((2, 2))(x_eo)
    x_eo = Flatten()(x_eo)
    x_eo = Dense(DENSE_NEURONS // 2, activation='relu', kernel_regularizer=l2(L2_REG))(x_eo)
    x_eo = Dropout(DROPOUT_RATE)(x_eo)
    
    # Branch A (EO) Prediction (before fusion)
    eo_branch_output = Dense(num_classes, activation='sigmoid', name='eo_output')(x_eo)

    # --- SAR branch (Grayscale CNN) ---
    sar_input = Input(shape=sar_shape, name='sar_input')
    # Mirroring the image branch structure, adapted for 1-channel input
    x_sar = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(sar_input)
    x_sar = BatchNormalization()(x_sar)
    x_sar = MaxPooling2D((2, 2))(x_sar)
    x_sar = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_sar)
    x_sar = BatchNormalization()(x_sar)
    x_sar = MaxPooling2D((2, 2))(x_sar)
    x_sar = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG))(x_sar)
    x_sar = BatchNormalization()(x_sar)
    x_sar = MaxPooling2D((2, 2))(x_sar)
    x_sar = Flatten()(x_sar)
    x_sar = Dense(DENSE_NEURONS // 2, activation='relu', kernel_regularizer=l2(L2_REG))(x_sar)
    x_sar = Dropout(DROPOUT_RATE)(x_sar)

    # Branch B (SAR) Prediction (before fusion)
    sar_branch_output = Dense(num_classes, activation='sigmoid', name='sar_output')(x_sar)

    # --- Merge branches ---
    merged = Concatenate()([x_eo, x_sar])
    merged_dense = Dense(DENSE_NEURONS, activation='relu', kernel_regularizer=l2(L2_REG))(merged)
    merged_dense = BatchNormalization()(merged_dense)
    merged_dense = Dropout(DROPOUT_RATE)(merged_dense)
    
    # --- Output Layer ---
    main_output = Dense(num_classes, activation='sigmoid', name='main_output')(merged_dense) # num_classes=1 for binary
    
    if return_branches:
        model = Model(inputs=[eo_input, sar_input], outputs=[main_output, eo_branch_output, sar_branch_output])
    else:
        model = Model(inputs=[eo_input, sar_input], outputs=main_output)
    
    logger.info("Two-branch EO/SAR multimodal model built.")
    # model.summary(print_fn=logger.info) # Commented out for cleaner console output during baseline test
    
    return model

def train_baseline():
    logger.info("Running baseline training to verify model shapes.")

    # Define dummy input shapes based on request
    DUMMY_EO_SHAPE = (224, 224, 3) # Optical RGB
    DUMMY_SAR_SHAPE = (224, 224, 1) # SAR Grayscale
    BATCH_SIZE = 4
    NUM_SAMPLES = 100
    EPOCHS = 1

    # Generate dummy data
    X_eo_dummy = np.random.rand(NUM_SAMPLES, *DUMMY_EO_SHAPE).astype(np.float32)
    X_sar_dummy = np.random.rand(NUM_SAMPLES, *DUMMY_SAR_SHAPE).astype(np.float32)
    y_dummy = np.random.randint(0, 2, (NUM_SAMPLES, 1)).astype(np.float32)

    # Build the model (without returning branches for baseline training)
    model = build_multimodal_fusion_model(DUMMY_EO_SHAPE, DUMMY_SAR_SHAPE, return_branches=False)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Run one epoch of training
    logger.info(f"Training model with dummy data for {EPOCHS} epoch(s)...")
    history = model.fit(
        {'eo_input': X_eo_dummy, 'sar_input': X_sar_dummy},
        y_dummy,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1 # Show progress bar
    )

    logger.info("Baseline training complete.")
    logger.info(f"Loss: {history.history['loss'][0]:.4f}, Accuracy: {history.history['accuracy'][0]:.4f}")

if __name__ == "__main__":
    # Setup basic logging for the script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_baseline()