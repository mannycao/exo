# FILE: models/model_trainer.py (Corrected)

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import config
from utils.visualization import visualize_learning_curves # Relies on the corrected visualization file

logger = logging.getLogger(__name__)


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for multi-class classification.
    Adjusts the standard cross-entropy loss to focus on hard-to-classify examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")
        
        epsilon = keras.backend.epsilon()
        y_pred = keras.ops.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross-entropy
        cross_entropy = -y_true * keras.ops.log(y_pred)
        
        # Calculate focal loss
        loss = alpha * keras.ops.power(1 - y_pred, gamma) * cross_entropy
        
        return keras.ops.sum(loss, axis=-1)
    return focal_loss_fixed


def train_enhanced_model(model, X_train, y_train, X_val, y_val, model_name, output_dir, 
                         use_focal_loss_config=False, use_class_weights_config=False, **kwargs):
    """
    Trains a Keras model with enhanced features like callbacks, focal loss, and class weights.
    """
    if model is None:
        logger.error(f"Cannot train a null model for {model_name}.")
        return None, None

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir_path / f"{model_name}_best.keras"
    history_path = output_dir_path / f"{model_name}_history.csv"

    # Callbacks for robust training
    checkpoint = ModelCheckpoint(
        filepath=str(model_path),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.LR_REDUCTION_FACTOR,
        patience=config.LR_REDUCTION_PATIENCE,
        verbose=1,
        mode='min'
    )
    
    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    class_weights = None
    if use_class_weights_config:
        # Calculate class weights to handle imbalance
        neg, pos = np.bincount(y_train.astype(int))
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1
        weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1
        class_weights = {0: weight_for_0, 1: weight_for_1}
        logger.info(f"Using class weights: {class_weights} for model {model_name}")

    loss_function = 'binary_crossentropy'
    if use_focal_loss_config:
        loss_function = focal_loss(
            gamma=config.FOCAL_LOSS_GAMMA, 
            alpha=config.FOCAL_LOSS_ALPHA
        )
        logger.info(f"Using focal loss for model {model_name}")
        
    optimizer = keras.optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=config.MODEL_METRICS
    )
    
    logger.info(f"Starting training for model: {model_name} with loss: {getattr(loss_function, '__name__', str(loss_function))}")

    # Determine if this is a single-input or multi-input model
    if isinstance(X_train, list):
        logger.info(f"Training multi-input model {model_name}.")
        train_data = X_train
        val_data = (X_val, y_val)
    else:
        logger.info(f"Training single-input model {model_name}.")
        train_data = X_train
        val_data = (X_val, y_val)

    history = model.fit(
        train_data, y_train,
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=val_data,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )

    # Save training history
    if history and history.history:
        pd.DataFrame(history.history).to_csv(history_path, index=False)
        try:
            # --- THIS IS THE CORRECTED FUNCTION CALL ---
            # The 'metrics' argument has been removed to match the function definition.
            visualize_learning_curves(
                history=history.history, 
                output_dir=str(output_dir_path), 
                filename=f"{model_name}_learning_curves.png"
            )
        except Exception as e:
            logger.error(f"Error generating learning curves for {model_name}: {e}")

    logger.info(f"Enhanced model training completed for {model_name}. Best model saved to: {model_path}")
    
    # Load the best performing model from the checkpoint
    best_model = keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': loss_function})
    
    return best_model, history
