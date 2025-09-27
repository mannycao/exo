"""
Enhanced model training functions for exoplanet detection.
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
from sklearn.metrics import precision_score, recall_score, f1_score # Moved to module level

import config

logger = logging.getLogger(__name__)


def focal_loss(gamma=2., alpha=.25):
    """
    Creates a focal loss function for imbalanced classification problems.
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        gamma: Focusing parameter that reduces relative loss for well-classified examples
        alpha: Class weight parameter to address class imbalance
    
    Returns:
        function: Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        Implementation of Focal Loss for binary classification.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        """
        from tensorflow.keras import backend as K
        
        # Clip to prevent numerical instability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Calculate focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow((1.0 - p_t), gamma)
        
        # Apply weights to cross entropy
        loss = alpha_factor * modulating_factor * cross_entropy
        
        # Sum over all samples
        return K.mean(loss)
    
    return focal_loss_fixed


def train_enhanced_model(model, X_train, y_train, X_val, y_val, model_name,
                         epochs=None, batch_size=None, output_dir=None,
                         use_focal_loss=True, use_class_weights=True):
    """
    Trains an AI model with enhanced techniques for imbalanced data.
    
    Args:
        model: Keras model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name for the saved model
        epochs: Number of training epochs
        batch_size: Training batch size
        output_dir: Directory to save output files
        use_focal_loss: Whether to use focal loss for imbalanced data
        use_class_weights: Whether to use class weights for imbalanced data
    
    Returns:
        tuple: (trained_model, training_history)
    """
    # Use config values if not specified
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    output_dir = output_dir or config.MODEL_DIR
    
    # Set up file paths
    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.h5")
    history_path = os.path.join(output_dir, f"{model_name}_history.csv")
    
    # Enhanced callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path, 
            save_best_only=True, 
            monitor='val_loss'
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=config.EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Class weights for imbalanced data
    class_weight_dict = None
    if use_class_weights:
        # Calculate balanced class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        logger.info(f"Using class weights: {class_weight_dict}")
    
    # Recompile model with appropriate loss function
    loss_function_to_use = 'binary_crossentropy'
    if use_focal_loss:
        logger.info("Using focal loss for imbalanced classification")
        model.compile(
            optimizer='adam',
            loss=focal_loss(gamma=2, alpha=0.25),
            metrics=['accuracy']
        )
    else:
        # Standard binary cross-entropy
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    logger.info(f"Starting training for model: {model_name} with loss: {str(loss_function_to_use)}")
    
    y_train_fit = np.asarray(y_train).reshape(-1, 1)
    y_val_fit = np.asarray(y_val).reshape(-1, 1)

    # Determine if X_train is a list (for multimodal) or a single array
    if isinstance(X_train, list):
        logger.info(f"Training multimodal model {model_name} with {len(X_train)} inputs.")
        # Ensure X_val is also a list of the same length
        if not isinstance(X_val, list) or len(X_val) != len(X_train):
            logger.error("X_val structure mismatch for multimodal input.")
            # Handle error appropriately, e.g., by raising an exception or returning
            raise ValueError("X_val structure mismatch for multimodal input.")
        history = model.fit(
            X_train, y_train_fit, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(X_val, y_val_fit),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
    else:
        logger.info(f"Training single-input model {model_name}.")
        history = model.fit(
            X_train, y_train_fit, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(X_val, y_val_fit),
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path)

    # Manually calculate metrics with a custom threshold for validation
    y_pred_val_raw = model.predict(X_val)
    custom_threshold = 0.40 # Optimal threshold found from analysis
    y_pred_val_thresholded = (y_pred_val_raw >= custom_threshold).astype(int)

    val_precision_custom = precision_score(y_val_fit, y_pred_val_thresholded, zero_division=0)
    val_recall_custom = recall_score(y_val_fit, y_pred_val_thresholded, zero_division=0)
    val_f1_custom = f1_score(y_val_fit, y_pred_val_thresholded, zero_division=0)

    # Update history with custom threshold metrics for reporting
    history.history['val_precision_custom'] = [val_precision_custom] * len(history.history['val_loss'])
    history.history['val_recall_custom'] = [val_recall_custom] * len(history.history['val_loss'])
    history.history['val_f1_custom'] = [val_f1_custom] * len(history.history['val_loss'])
    
    # Save learning curves
    visualize_enhanced_learning_curves(
        history, 
        filename=f"{model_name}_learning_curves.png",
        output_dir=output_dir
    )
    
    logger.info(f"Enhanced model training completed for {model_name}. Best model saved to: {checkpoint_path}")
    return model, history


def visualize_enhanced_learning_curves(history, metrics=None, title=None, filename=None, output_dir=None):
    """
    Create improved visualizations of learning curves from model training history.
    
    Args:
        history: Keras history object or dictionary
        metrics: List of metrics to plot (optional)
        title: Plot title (optional)
        filename: Path to save the visualization (optional)
        output_dir: Directory to save the visualization (optional)
    """
    # Convert to dictionary if it's a Keras history object
    if hasattr(history, 'history'):
        history = history.history
    
    if not history:
        logger.warning("Empty training history provided")
        return
    
    # If metrics not specified, plot all except validation metrics
    if metrics is None:
        metrics = [m for m in history.keys() if not m.startswith('val_')]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training and validation metrics
    for metric in metrics:
        # Skip validation metrics in this loop
        if metric.startswith('val_'):
            continue
            
        val_metric = f'val_{metric}'
        
        # Plot on first subplot - training and validation metrics
        axes[0].plot(history[metric], label=f'Training {metric}', marker='o')
        
        if val_metric in history:
            axes[0].plot(history[val_metric], label=f'Validation {metric}', marker='x')
        
        axes[0].set_title('Training and Validation Metrics')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Plot learning rate if available
    if 'lr' in history:
        ax_lr = axes[1]
        ax_lr.plot(history['lr'], marker='o', label='Learning Rate')
        ax_lr.set_title('Learning Rate')
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_yscale('log')  # Log scale for learning rate
        ax_lr.grid(True, alpha=0.3)
    else:
        # Plot loss ratio
        axes[1].plot([history['loss'][i] / history['val_loss'][i] for i in range(len(history['loss']))], 
                      label='Train/Val Loss Ratio')
        axes[1].set_title('Train/Validation Loss Ratio')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Ratio')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure if a filename is provided
    if filename:
        if output_dir:
            full_path = os.path.join(output_dir, filename)
        else:
            full_path = filename
            
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()