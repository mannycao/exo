"""
Functions for training and evaluating machine learning models.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

import config

logger = logging.getLogger(__name__)


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=None, batch_size=None, 
               output_dir=None, class_weight=None):
    """
    Trains an AI model using improved metrics and loss functions.
    """
    # Use config values if not specified
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    output_dir = output_dir or config.MODEL_DIR
    
    # Set up file paths
    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.h5")
    history_path = os.path.join(output_dir, f"{model_name}_history.csv")
    
    # If class_weight not provided, compute it
    if class_weight is None:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weight_values = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weight = {i: weight for i, weight in enumerate(class_weight_values)}
        logger.info(f"Using computed class weights: {class_weight}")
    
    # Implement focal loss for better handling of imbalanced data
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            """
            Focal Loss for better handling of imbalanced data.
            FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
            """
            from tensorflow.keras import backend as K
            
            # Clip to prevent numerical instability
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            
            # Calculate focal loss
            cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            modulating_factor = K.pow((1.0 - p_t), gamma)
            
            loss = alpha_factor * modulating_factor * cross_entropy
            return K.mean(loss)
        return focal_loss_fixed
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=config.EARLY_STOPPING_PATIENCE, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Learning rate schedule
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=len(y_train) // batch_size * 5,  # 5 epochs
        decay_rate=0.9,
        staircase=True
    )
    
    # Use Adam optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Recompile model with improved loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    logger.info(f"Training model: {model_name}")
    
    # Handle multimodal model training (multiple inputs)
    if isinstance(X_train, list):
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path)
    
    # Save learning curves
    from utils.visualization import visualize_learning_curves
    visualize_learning_curves(
        history, 
        metrics=['loss', 'accuracy', 'precision', 'recall', 'auc'],
        filename=f"{model_name}_learning_curves.png",
        output_dir=output_dir
    )
    
    logger.info(f"Model training completed. Best model saved to: {checkpoint_path}")
    
    return model, history


def evaluate_model(model, X_test, y_test, model_name, output_dir=None):
    """
    Evaluates model performance with appropriate metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for saving results
        output_dir: Directory to save evaluation results
    
    Returns:
        dict: Evaluation metrics
    """
    output_dir = output_dir or config.MODEL_DIR
    
    logger.info(f"Evaluating model: {model_name}")
    
    # Basic evaluation
    evaluation = model.evaluate(X_test, y_test, verbose=1)
    metrics = {}
    
    # Map evaluation metrics to names
    for i, metric_name in enumerate(model.metrics_names):
        metrics[metric_name] = float(evaluation[i])
    
    # Predict probabilities
    y_pred_prob = model.predict(X_test)
    
    # For binary classification
    if y_test.ndim == 1 or y_test.shape[1] == 1:
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        ap_score = average_precision_score(y_test, y_pred_prob)
        metrics['average_precision'] = ap_score
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        metrics['roc_auc'] = roc_auc
        
        # Save precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'AP Score: {ap_score:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        pr_curve_path = os.path.join(output_dir, f'{model_name}_precision_recall_curve.png')
        plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC AUC: {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')  # Random baseline
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        roc_curve_path = os.path.join(output_dir, f'{model_name}_roc_curve.png')
        plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate additional classification metrics
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        from utils.metrics import confusion_matrix_with_metrics
        class_metrics = confusion_matrix_with_metrics(y_test, y_pred)
        
        # Add class metrics to overall metrics
        for key, value in class_metrics.items():
            if key != 'confusion_matrix':
                metrics[key] = value
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, f'{model_name}_metrics.csv'), index=False)
    
    logger.info(f"Model evaluation completed: {metrics}")
    
    return metrics


def prepare_train_test_split(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """
    Prepares train, validation, and test splits of the data.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of non-test data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Ensure y is a numpy array
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # First split: training+validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.ndim == 1 else None
    )
    
    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, 
        random_state=random_state, stratify=y_train_val if y_train_val.ndim == 1 else None
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def optimize_hyperparameters(build_model_fn, X, y, param_grid, n_iter=10, cv=3):
    """
    Optimizes hyperparameters for a Keras model using scikit-learn.
    
    Args:
        build_model_fn: Function that builds and returns a Keras model
        X: Features
        y: Labels
        param_grid: Dictionary of hyperparameter options
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
    
    Returns:
        dict: Best hyperparameters
    """
    try:
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import make_scorer, average_precision_score
        
        # Custom scorer that optimizes for average precision
        ap_scorer = make_scorer(average_precision_score, needs_proba=True)
        
        # Create the model wrapper
        model_wrapper = KerasClassifier(build_fn=build_model_fn, verbose=0)
        
        # Create the random search with custom scorer
        random_search = RandomizedSearchCV(
            estimator=model_wrapper,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=ap_scorer,
            verbose=1,
            n_jobs=1,  # Keras models don't work well with n_jobs > 1
            random_state=42
        )
        
        # Fit the random search
        logger.info("Starting hyperparameter optimization")
        random_search.fit(X, y)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best average precision score: {random_search.best_score_:.4f}")
        
        return random_search.best_params_
    
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {e}")
        return {}


def focal_loss(gamma=2., alpha=.25):
    """
    Creates a focal loss function for imbalanced classification problems.
    
    Args:
        gamma: Focusing parameter
        alpha: Balancing parameter
    
    Returns:
        function: Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        """
        Focal loss for binary classification.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        """
        import tensorflow as tf
        import keras.backend as K
        
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
