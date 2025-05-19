"""
Evaluation metrics for exoplanet detection models.
"""

import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


def calculate_binary_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculate comprehensive metrics for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC and PR curve metrics
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        roc_auc = 0.5  # Default for random classifier
        
    try:
        average_precision = average_precision_score(y_true, y_pred_probs)
    except Exception:
        average_precision = 0.0  # Default
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    
    # Calculate metrics at different thresholds
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred_probs)
    
    # Find optimal F1 threshold
    f1_scores = []
    for prec, rec in zip(precision_curve, recall_curve):
        f1_scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else threshold
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'npv': npv,
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'optimal_threshold': optimal_threshold
    }
    
    return metrics


def calculate_detection_metrics(true_transits, detected_transits, window=5):
    """
    Calculate metrics for transit detection performance.
    
    Args:
        true_transits: List of true transit indices
        detected_transits: List of detected transit indices
        window: Window size for matching transits
    
    Returns:
        dict: Dictionary of detection metrics
    """
    # Initialize counters
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    # Mark detected transits
    detected = [False] * len(detected_transits)
    
    # First pass: count true positives and false negatives
    for true_idx in true_transits:
        # Check if this true transit was detected
        matched = False
        for i, detected_idx in enumerate(detected_transits):
            if abs(true_idx - detected_idx) <= window and not detected[i]:
                # Match found
                tp += 1
                detected[i] = True
                matched = True
                break
        
        # If no match found, it's a false negative
        if not matched:
            fn += 1
    
    # Count false positives (unmatched detections)
    fp = sum(1 for d in detected if not d)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate metrics for regression tasks (e.g., planet property estimation).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary of regression metrics
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        # Handle infinities and NaNs
        if np.isinf(mape) or np.isnan(mape):
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # R-squared
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }
    
    return metrics


def calculate_multimodal_improvement(cnn_metrics, multimodal_metrics):
    """
    Calculate the improvement of multimodal model over CNN model.
    
    Args:
        cnn_metrics: Dictionary of CNN model metrics
        multimodal_metrics: Dictionary of multimodal model metrics
    
    Returns:
        dict: Dictionary of improvement metrics
    """
    improvement = {}
    
    # Find common metrics
    common_metrics = set(cnn_metrics.keys()) & set(multimodal_metrics.keys())
    
    for metric in common_metrics:
        if isinstance(cnn_metrics[metric], (int, float)) and isinstance(multimodal_metrics[metric], (int, float)):
            if cnn_metrics[metric] != 0:  # Avoid division by zero
                pct_change = ((multimodal_metrics[metric] - cnn_metrics[metric]) / cnn_metrics[metric]) * 100
                improvement[f"{metric}_pct_change"] = pct_change
            
            absolute_change = multimodal_metrics[metric] - cnn_metrics[metric]
            improvement[f"{metric}_absolute_change"] = absolute_change
    
    return improvement


def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=1000, confidence=0.95):
    """
    Calculate confidence intervals for metrics using bootstrapping.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metric_func: Function to calculate metric
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% confidence)
    
    Returns:
        tuple: (metric_value, lower_bound, upper_bound)
    """
    import numpy as np
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Calculate the base metric value
    base_metric = metric_func(y_true, y_pred)
    
    # Generate bootstrap samples
    bootstrap_metrics = []
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
        bootstrap_y_true = y_true[bootstrap_indices]
        bootstrap_y_pred = y_pred[bootstrap_indices]
        
        # Calculate metric on bootstrap sample
        bootstrap_metric = metric_func(bootstrap_y_true, bootstrap_y_pred)
        bootstrap_metrics.append(bootstrap_metric)
    
    # Calculate confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    
    lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
    upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
    
    return base_metric, lower_bound, upper_bound


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Generate a comprehensive classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Names of the target classes
    
    Returns:
        str: Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=target_names)
def calculate_precision_recall_curve_with_thresholds(y_true, y_pred_probs, thresholds=None, return_thresholds=False):
    """
    Calculate precision-recall values at various thresholds.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Predicted probabilities
        thresholds: Specific thresholds to evaluate (optional)
        return_thresholds: Whether to return thresholds
    
    Returns:
        dict: Precision-recall values at different thresholds
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import numpy as np
    
    # Get the full precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_probs)
    ap_score = average_precision_score(y_true, y_pred_probs)
    
    # Create a dictionary for the full curve
    full_curve = {
        'precision': precision,
        'recall': recall,
        'thresholds': pr_thresholds,
        'ap_score': ap_score
    }
    
    # If specific thresholds aren't provided, create a range of thresholds
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Calculate metrics at specific thresholds
    threshold_results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        
        # Calculate precision and recall
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()
        false_negatives = ((y_pred == 0) & (y_true == 1)).sum()
        
        if (true_positives + false_positives) > 0:
            precision_val = true_positives / (true_positives + false_positives)
        else:
            precision_val = 0.0
            
        if (true_positives + false_negatives) > 0:
            recall_val = true_positives / (true_positives + false_negatives)
        else:
            recall_val = 0.0
        
        f1_score = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        # Calculate F2 score (emphasizes recall more than precision)
        beta = 2
        f2_score = (1 + beta**2) * precision_val * recall_val / ((beta**2 * precision_val) + recall_val) if (precision_val + recall_val) > 0 else 0
        
        threshold_results.append({
            'threshold': threshold,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_score,
            'f2_score': f2_score,
            'predicted_positives': int(true_positives + false_positives),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        })
    
    # Find the optimal threshold for F1 score
    f1_scores = [result['f1_score'] for result in threshold_results]
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = threshold_results[optimal_f1_idx]['threshold']
    
    # Find the optimal threshold for F2 score (recall-focused)
    f2_scores = [result['f2_score'] for result in threshold_results]
    optimal_f2_idx = np.argmax(f2_scores)
    optimal_f2_threshold = threshold_results[optimal_f2_idx]['threshold']
    
    # Create summary with optimization results
    summary = {
        'ap_score': ap_score,
        'optimal_f1_threshold': optimal_f1_threshold,
        'optimal_f1_score': threshold_results[optimal_f1_idx]['f1_score'],
        'optimal_f2_threshold': optimal_f2_threshold,
        'optimal_f2_score': threshold_results[optimal_f2_idx]['f2_score'],
        'threshold_metrics': threshold_results,
        'full_curve': full_curve
    }
    
    if return_thresholds:
        return summary, thresholds
    else:
        return summary


def visualize_threshold_analysis(threshold_metrics, output_dir=None, filename=None):
    """
    Create a visualization of precision, recall, F1, and F2 scores at different thresholds.
    
    Args:
        threshold_metrics: Dictionary of metrics at different thresholds
        output_dir: Directory to save the visualization (optional)
        filename: Name of the output file (optional)
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    # Extract metrics from the results
    thresholds = [m['threshold'] for m in threshold_metrics]
    precision = [m['precision'] for m in threshold_metrics]
    recall = [m['recall'] for m in threshold_metrics]
    f1_scores = [m['f1_score'] for m in threshold_metrics]
    f2_scores = [m['f2_score'] for m in threshold_metrics]
    
    plt.figure(figsize=(12, 8))
    
    # Plot metrics vs threshold
    plt.subplot(2, 1, 1)
    plt.plot(thresholds, precision, 'b-', label='Precision')
    plt.plot(thresholds, recall, 'r-', label='Recall')
    plt.plot(thresholds, f1_scores, 'g-', label='F1 Score')
    plt.plot(thresholds, f2_scores, 'y-', label='F2 Score')
    
    # Find optimal F1 and F2 thresholds
    f1_optimal_idx = np.argmax(f1_scores)
    f2_optimal_idx = np.argmax(f2_scores)
    
    # Mark optimal thresholds
    plt.axvline(x=thresholds[f1_optimal_idx], color='g', linestyle='--', 
                label=f'Optimal F1 Threshold: {thresholds[f1_optimal_idx]:.2f}')
    plt.axvline(x=thresholds[f2_optimal_idx], color='y', linestyle='--', 
                label=f'Optimal F2 Threshold: {thresholds[f2_optimal_idx]:.2f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot precision vs recall
    plt.subplot(2, 1, 2)
    plt.plot(recall, precision, 'b-o')
    
    # Add threshold annotations
    for i, threshold in enumerate(thresholds):
        # Only annotate some thresholds to avoid clutter
        if i % 2 == 0:
            plt.annotate(f'{threshold:.1f}', 
                         (recall[i], precision[i]),
                         xytext=(5, 5),
                         textcoords='offset points')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall at Different Thresholds')
    plt.grid(True, alpha=0.3)
    
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


def confusion_matrix_with_metrics(y_true, y_pred):
    """
    Calculate confusion matrix and related metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        dict: Confusion matrix and metrics
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract values from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = fn = fp = tn = 0
        
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle division by zero
    if tp + fp > 0:
        precision = precision_score(y_true, y_pred, zero_division=0)
    else:
        precision = 0.0
        
    if tp + fn > 0:
        recall = recall_score(y_true, y_pred, zero_division=0)
    else:
        recall = 0.0
        
    if precision + recall > 0:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        f1 = 0.0
    
    # Calculate additional metrics
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0
        
    # Calculate F2 score (emphasizes recall)
    if precision + recall > 0:
        beta = 2
        f2 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
    else:
        f2 = 0.0
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'f2_score': f2,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
