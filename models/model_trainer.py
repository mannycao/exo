"""
Enhanced model training functions for exoplanet detection.
Uses focal loss and class weights for imbalanced data.
Includes training with confidence-weighted samples.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt 
from pathlib import Path # <<<<<<<<<<<< IMPORT ADDED HERE

# Attempt to import config from parent directory if this file is in models/
try:
    from .. import config # Assumes model_trainer.py is in models/ and config.py is in project root
except ImportError:
    # Fallback for direct execution or different structure
    try:
        import config
    except ImportError:
        # Define a dummy config if absolutely necessary, though this indicates a setup issue
        class DummyConfig:
            EPOCHS = 50
            BATCH_SIZE = 32
            MODEL_DIR = Path("./models_trained") # Default if config cannot be loaded
            EARLY_STOPPING_PATIENCE = 10
            LEARNING_RATE = 1e-3
        config = DummyConfig()
        logging.error("Could not import config.py in model_trainer.py. Using dummy config.")

# Assuming utils.metrics contains focal_loss and utils.visualization contains visualize_learning_curves
try:
    from utils.metrics import focal_loss 
    from utils.visualization import visualize_learning_curves 
except ImportError as e:
    logging.error(f"Could not import helper functions from utils: {e}. Ensure utils directory is accessible.")
    # Define dummy functions if imports fail, to prevent further NameErrors, though functionality will be lost
    def focal_loss(gamma=2., alpha=.25): # Dummy
        logging.warning("Using dummy focal_loss function due to import error.")
        return 'binary_crossentropy' 
    def visualize_learning_curves(history, metrics, filename, output_dir): # Dummy
        logging.warning("Using dummy visualize_learning_curves function due to import error.")
        pass

logger = logging.getLogger(__name__)


def train_enhanced_model(model, X_train, y_train, X_val, y_val, model_name,
                         epochs=None, batch_size=None, output_dir=None,
                         use_focal_loss_config=True, use_class_weights_config=True):
    """
    Trains an AI model with enhanced techniques for imbalanced data.
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    # This was the line causing the error: Path was not defined. It is now fixed by the import.
    output_dir_path = Path(output_dir) if output_dir else Path(config.MODEL_DIR) 
    output_dir_path.mkdir(parents=True, exist_ok=True) 

    checkpoint_path = str(output_dir_path / f"{model_name}_best.keras") 
    history_path = str(output_dir_path / f"{model_name}_history.csv")

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss', 
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, 
            patience=5, 
            min_lr=1e-7, 
            verbose=1
        )
    ]

    class_weight_dict = None
    if use_class_weights_config:
        y_train_flat = np.asarray(y_train).ravel()
        if len(np.unique(y_train_flat)) > 1: 
            classes = np.unique(y_train_flat)
            class_weights_values = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train_flat 
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights_values)}
            logger.info(f"Using class weights: {class_weight_dict} for model {model_name}")
        else:
            logger.warning(f"Only one class present in y_train for model {model_name}. Cannot compute class weights.")

    loss_function_to_use = 'binary_crossentropy'
    if use_focal_loss_config:
        logger.info(f"Using focal loss for model {model_name}")
        # Ensure focal_loss is callable or a string Keras understands
        fl = focal_loss(gamma=2.0, alpha=0.25)
        if callable(fl):
            loss_function_to_use = fl
        else: # Fallback if dummy focal_loss was used due to import error
            loss_function_to_use = 'binary_crossentropy'
            logger.warning("Focal loss function not callable, falling back to binary_crossentropy.")


    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE) 

    model.compile(
        optimizer=optimizer,
        loss=loss_function_to_use,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='roc_auc', curve='ROC'), 
            tf.keras.metrics.AUC(name='pr_auc', curve='PR')   
            ]
    )

    logger.info(f"Starting training for model: {model_name} with loss: {str(loss_function_to_use)}")
    
    y_train_fit = np.asarray(y_train).reshape(-1, 1)
    y_val_fit = np.asarray(y_val).reshape(-1, 1)

    # Determine if X_train is a list (for multimodal) or a single array
    if isinstance(X_train, list):
        logger.info(f"Training multimodal model {model_name} with {len(X_train)} inputs.")
        # Ensure X_val is also a list of the same length
        if not isinstance(X_val, list) or len(X_val) != len(X_train):
            logger.error("X_val must be a list of same length as X_train for multimodal input.")
            # Handle error appropriately, e.g., by raising an exception or returning
            raise ValueError("X_val structure mismatch for multimodal input.")
    else:
        logger.info(f"Training single-input model {model_name}.")


    history = model.fit(
        X_train, y_train_fit, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_fit), 
        callbacks=callbacks,
        class_weight=class_weight_dict if use_class_weights_config and class_weight_dict is not None else None,
        verbose=1
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path)

    try:
        visualize_learning_curves(
            history.history, 
            metrics=['loss', 'accuracy', 'precision', 'recall', 'roc_auc', 'pr_auc'], 
            filename=f"{model_name}_learning_curves.png",
            output_dir=str(output_dir_path)
        )
    except NameError: # In case visualize_learning_curves couldn't be imported
        logger.error("visualize_learning_curves is not defined. Skipping learning curve plot.")
    except Exception as e_vis:
        logger.error(f"Error generating learning curves for {model_name}: {e_vis}")


    logger.info(f"Enhanced model training completed for {model_name}. Best model saved to: {checkpoint_path}")
    return model, history


def train_with_confidence_weighted_samples(X_train, y_train, confidences, model,
                                           epochs=None, batch_size=None, output_dir=None, model_name="confidence_weighted_model"):
    """
    Train a model with confidence-weighted samples.
    (This function is from your original models/enhanced_trainer.py)
    
    Args:
        X_train: Training features
        y_train: Training labels
        confidences: Confidence scores for each label (must be same length as y_train)
        model: Model to train
        epochs, batch_size, output_dir, model_name: Standard training parameters
    
    Returns:
        tuple: (trained_model, history)
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    output_dir_path = Path(output_dir) if output_dir else Path(config.MODEL_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(output_dir_path / f"{model_name}_best.keras")
    history_path = str(output_dir_path / f"{model_name}_history.csv")

    if len(X_train) != len(confidences): # This might be an issue if X_train is a list for multimodal
        if isinstance(X_train, list):
            if len(X_train[0]) != len(confidences): # Check against the first element if X_train is list
                 raise ValueError("Length of X_train elements and confidences must match for multimodal.")
        else: # X_train is a single numpy array
            raise ValueError("Length of X_train and confidences must match.")


    # Create sample weights based on confidence
    sample_weights = np.asarray(confidences).copy()
    sample_weights = np.maximum(sample_weights, 0.01) # Apply minimum weight
    # Normalize weights (optional, but can help stabilize training if confidences vary wildly)
    # sample_weights = sample_weights / np.mean(sample_weights) 
    
    logger.info(f"Training model {model_name} with confidence-weighted samples.")

    # Assuming model is already compiled with optimizer, loss, metrics
    # If not, compile it here similar to train_enhanced_model

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    val_split_prop = 0.2 
    
    history = model.fit(
        X_train, np.asarray(y_train).reshape(-1,1), 
        sample_weight=sample_weights,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split_prop, 
        callbacks=callbacks,
        verbose=1
    )
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path)

    try:
        visualize_learning_curves(
            history.history,
            metrics=['loss', 'accuracy'] + [m for m in model.metrics_names if m not in ['loss', 'accuracy']],
            filename=f"{model_name}_learning_curves.png",
            output_dir=str(output_dir_path)
        )
    except NameError:
        logger.error("visualize_learning_curves is not defined. Skipping learning curve plot.")
    except Exception as e_vis:
        logger.error(f"Error generating learning curves for {model_name}: {e_vis}")

    logger.info(f"Confidence-weighted model training completed for {model_name}. Best model saved to: {checkpoint_path}")
    return model, history
