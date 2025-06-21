"""
Evaluation metrics for exoplanet detection models.
"""

import logging
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
from pathlib import Path # <<<<<<<<<<<< ADDED THIS IMPORT

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report
)
from tensorflow.keras import backend as K 

logger = logging.getLogger(__name__)


def focal_loss(gamma=2., alpha=.25):
    """
    Creates a focal loss function for imbalanced classification problems.
    Reference: https://arxiv.org/abs/1708.02002

    Args:
        gamma: Focusing parameter that reduces relative loss for well-classified examples.
        alpha: Class weight parameter to address class imbalance.

    Returns:
        function: Focal loss function.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = K.pow((1.0 - p_t), gamma)
        loss = alpha_factor * modulating_factor * cross_entropy
        return K.mean(loss)
    return focal_loss_fixed


def calculate_binary_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculate comprehensive metrics for binary classification.
    """
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    if len(y_true) == 0:
        logger.warning("Empty y_true array for metric calculation.")
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'specificity': 0.0, 'npv': 0.0, 'roc_auc': 0.0, 'average_precision': 0.0,
            'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0,
            'optimal_threshold': threshold
        }
    if len(y_pred_probs) == 0: 
        logger.warning("Empty y_pred_probs array for metric calculation.")
        unique_true_labels = np.unique(y_true)
        ap_default = 0.0
        if len(unique_true_labels) == 1 : 
             if unique_true_labels[0] == 1 : ap_default = 1.0
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'specificity': 0.0, 'npv': 0.0, 'roc_auc': 0.5 if len(unique_true_labels) > 1 else 0.0, 
            'average_precision': ap_default,
            'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0,
            'optimal_threshold': threshold
        }

    y_pred = (y_pred_probs >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = 0.5 
    average_precision = 0.0 
    unique_true_labels = np.unique(y_true)

    if len(unique_true_labels) > 1: 
        try:
            roc_auc = roc_auc_score(y_true, y_pred_probs)
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC (y_true may contain only one class): {e}")
        try:
            average_precision = average_precision_score(y_true, y_pred_probs)
        except ValueError as e:
            logger.warning(f"Could not calculate Average Precision (y_true may contain only one class): {e}")
    elif len(unique_true_labels) == 1: 
        if unique_true_labels[0] == 1: 
            average_precision = 1.0 
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) 
    if cm.size == 4: 
        tn, fp, fn, tp = cm.ravel()
    else: 
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0 

    optimal_threshold_val = threshold 
    if len(unique_true_labels) > 1 and len(y_true) > 1:
        try:
            precision_curve, recall_curve, pr_thresholds_sklearn = precision_recall_curve(y_true, y_pred_probs)
            f1_scores_curve = []
            for i in range(len(pr_thresholds_sklearn)):
                prec_val = precision_curve[i] 
                rec_val = recall_curve[i]   
                if (prec_val + rec_val) > 0:
                    f1_scores_curve.append(2 * prec_val * rec_val / (prec_val + rec_val))
                else:
                    f1_scores_curve.append(0.0)
            
            if f1_scores_curve: 
                optimal_idx = np.argmax(f1_scores_curve)
                optimal_threshold_val = pr_thresholds_sklearn[optimal_idx]
            else: 
                 logger.warning("No valid F1 scores generated from PR curve for optimal threshold calculation.")
        except Exception as e:
            logger.warning(f"Could not determine optimal threshold from PR curve: {e}", exc_info=True)

    metrics = {
        'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall),
        'f1_score': float(f1), 'specificity': float(specificity), 'npv': float(npv),
        'roc_auc': float(roc_auc), 'average_precision': float(average_precision),
        'true_positives': int(tp), 'false_positives': int(fp),
        'true_negatives': int(tn), 'false_negatives': int(fn),
        'optimal_threshold': float(optimal_threshold_val)
    }
    return metrics


def calculate_detection_metrics(true_transits, detected_transits, window=5):
    """
    Calculate metrics for transit detection performance.
    """
    tp = 0; fp = 0; fn = 0
    detected_mask = [False] * len(detected_transits) 

    for true_idx in true_transits:
        matched = False
        for i, detected_idx in enumerate(detected_transits):
            if abs(true_idx - detected_idx) <= window and not detected_mask[i]:
                tp += 1
                detected_mask[i] = True
                matched = True
                break
        if not matched:
            fn += 1
    
    fp = sum(1 for d_mask_val in detected_mask if not d_mask_val) 

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'precision': precision, 'recall': recall, 'f1_score': f1
    }


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate metrics for regression tasks.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0 : return {'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'r2': 0}

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    mape = np.nan 
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    elif len(y_true) > 0 : 
        mape = np.mean(np.abs(y_pred)) * 100 

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2}


def calculate_multimodal_improvement(cnn_metrics, multimodal_metrics):
    """
    Calculate the improvement of multimodal model over CNN model.
    """
    improvement = {}
    if not cnn_metrics or not multimodal_metrics: return improvement

    common_metrics = set(cnn_metrics.keys()) & set(multimodal_metrics.keys())
    for metric in common_metrics:
        cnn_val = cnn_metrics.get(metric)
        mm_val = multimodal_metrics.get(metric)
        if isinstance(cnn_val, (int, float)) and isinstance(mm_val, (int, float)):
            if cnn_val != 0:
                pct_change = ((mm_val - cnn_val) / abs(cnn_val)) * 100 
                improvement[f"{metric}_pct_change"] = pct_change
            absolute_change = mm_val - cnn_val
            improvement[f"{metric}_absolute_change"] = absolute_change
    return improvement


def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=1000, confidence=0.95):
    """
    Calculate confidence intervals for metrics using bootstrapping.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return (0,0,0) if callable(metric_func) and metric_func.__name__ not in ['roc_auc_score', 'average_precision_score'] else (0.5, 0.5, 0.5)

    base_metric = metric_func(y_true, y_pred)
    bootstrap_metrics = []
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
        bootstrap_y_true = y_true[bootstrap_indices]
        bootstrap_y_pred = y_pred[bootstrap_indices]
        try:
            bootstrap_metric = metric_func(bootstrap_y_true, bootstrap_y_pred)
            bootstrap_metrics.append(bootstrap_metric)
        except ValueError: 
            continue
    
    if not bootstrap_metrics: 
        return base_metric, base_metric, base_metric 

    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
    upper_bound = np.percentile(bootstrap_metrics, upper_percentile)
    
    return base_metric, lower_bound, upper_bound


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Generate a comprehensive classification report.
    """
    if len(y_true) == 0 or len(y_pred) == 0: return "No data to report."
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def calculate_precision_recall_curve_with_thresholds(y_true, y_pred_probs, thresholds_input=None, return_thresholds=False):
    """
    Calculate precision-recall values at various thresholds.
    """
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    if len(y_true) == 0 or len(y_pred_probs) == 0:
        logger.warning("Empty y_true or y_pred_probs for PR curve calculation.")
        empty_curve = {'precision': [], 'recall': [], 'thresholds': [], 'ap_score': 0.0}
        empty_summary = {
            'ap_score': 0.0, 'optimal_f1_threshold': 0.5, 'optimal_f1_score': 0.0,
            'optimal_f2_threshold': 0.5, 'optimal_f2_score': 0.0,
            'threshold_metrics': [], 'full_curve': empty_curve
        }
        return empty_summary if not return_thresholds else (empty_summary, [])

    ap_score = 0.0
    full_curve_thresholds = np.array([])
    precision_sklearn_curve = np.array([1.0]) 
    recall_sklearn_curve = np.array([0.0])    

    if len(np.unique(y_true)) > 1: 
        try:
            precision_sklearn_curve, recall_sklearn_curve, full_curve_thresholds = precision_recall_curve(y_true, y_pred_probs)
            ap_score = average_precision_score(y_true, y_pred_probs)
        except ValueError as e:
            logger.warning(f"Could not calculate sklearn PR curve or AP score (y_true may have only one class): {e}")
    elif len(y_true)>0 and np.all(y_true == 1): 
        ap_score = 1.0 
    
    full_curve = {
        'precision': precision_sklearn_curve.tolist(),
        'recall': recall_sklearn_curve.tolist(),
        'thresholds': full_curve_thresholds.tolist(), 
        'ap_score': ap_score
    }
    
    if thresholds_input is None:
        thresholds_to_evaluate = np.linspace(0.05, 0.95, 19).tolist() 
    else:
        thresholds_to_evaluate = list(thresholds_input) 
    
    threshold_results = []
    if not thresholds_to_evaluate : thresholds_to_evaluate = [0.5] 

    for th_val in thresholds_to_evaluate:
        y_pred_binary = (y_pred_probs >= th_val).astype(int)
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        beta = 2
        f2 = (1 + beta**2) * prec * rec / ((beta**2 * prec) + rec) if (prec + rec) > 0 else 0.0
        threshold_results.append({
            'threshold': th_val, 'precision': prec, 'recall': rec, 'f1_score': f1, 'f2_score': f2,
            'true_positives': int(tp), 'false_positives': int(fp), 'false_negatives': int(fn),
            'predicted_positives': int(tp + fp)
        })
    
    optimal_f1_th, optimal_f1_val = 0.5, 0.0
    optimal_f2_th, optimal_f2_val = 0.5, 0.0

    if threshold_results:
        f1_s = np.array([r['f1_score'] for r in threshold_results])
        if len(f1_s) > 0 :
            optimal_f1_idx = np.argmax(f1_s)
            optimal_f1_th = threshold_results[optimal_f1_idx]['threshold']
            optimal_f1_val = threshold_results[optimal_f1_idx]['f1_score']
        f2_s = np.array([r['f2_score'] for r in threshold_results])
        if len(f2_s) > 0:
            optimal_f2_idx = np.argmax(f2_s)
            optimal_f2_th = threshold_results[optimal_f2_idx]['threshold']
            optimal_f2_val = threshold_results[optimal_f2_idx]['f2_score']
            
    summary = {
        'ap_score': ap_score,
        'optimal_f1_threshold': optimal_f1_th,
        'optimal_f1_score': optimal_f1_val,
        'optimal_f2_threshold': optimal_f2_th,
        'optimal_f2_score': optimal_f2_val,
        'threshold_metrics': threshold_results,
        'full_curve': full_curve 
    }
    return summary if not return_thresholds else (summary, thresholds_to_evaluate)


def visualize_threshold_analysis(threshold_metrics_list, output_dir=None, filename=None):
    """
    Create a visualization of precision, recall, F1, and F2 scores at different thresholds.
    """
    if not threshold_metrics_list: 
        logger.warning("No threshold metrics provided to visualize_threshold_analysis.")
        return

    thresholds = [m['threshold'] for m in threshold_metrics_list]
    precision = [m['precision'] for m in threshold_metrics_list]
    recall = [m['recall'] for m in threshold_metrics_list]
    f1_scores = [m['f1_score'] for m in threshold_metrics_list]
    f2_scores = [m['f2_score'] for m in threshold_metrics_list]
    
    plt.figure(figsize=(14, 10)) 
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(thresholds, precision, 'b-o', label='Precision', markersize=5)
    ax1.plot(thresholds, recall, 'r-s', label='Recall', markersize=5)
    ax1.plot(thresholds, f1_scores, 'g-^', label='F1 Score', markersize=5)
    ax1.plot(thresholds, f2_scores, 'y-d', label='F2 Score', markersize=5)
    
    if f1_scores:
        f1_optimal_idx = np.argmax(f1_scores)
        ax1.axvline(x=thresholds[f1_optimal_idx], color='g', linestyle='--', 
                    label=f'Optimal F1 Th: {thresholds[f1_optimal_idx]:.2f} (F1={f1_scores[f1_optimal_idx]:.2f})')
    if f2_scores:
        f2_optimal_idx = np.argmax(f2_scores)
        ax1.axvline(x=thresholds[f2_optimal_idx], color='y', linestyle=':', 
                    label=f'Optimal F2 Th: {thresholds[f2_optimal_idx]:.2f} (F2={f2_scores[f2_optimal_idx]:.2f})')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs. Classification Threshold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.5)
    ax1.set_ylim(0, 1.05) 
    
    ax2 = plt.subplot(2, 1, 2)
    sorted_indices = np.argsort(recall)
    ax2.plot(np.array(recall)[sorted_indices], np.array(precision)[sorted_indices], 'b-o', markersize=5)
    
    annotation_indices = list(range(0, len(thresholds), max(1, len(thresholds)//5))) 
    if f1_scores and f1_optimal_idx not in annotation_indices: annotation_indices.append(f1_optimal_idx)

    for i in annotation_indices:
        if i < len(thresholds): 
            ax2.annotate(f'{thresholds[i]:.2f}', 
                         (recall[i], precision[i]),
                         xytext=(5, -5 if i % 2 == 0 else 5), 
                         textcoords='offset points',
                         fontsize=8)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve (from evaluated thresholds)')
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout(pad=3.0) 
    
    if filename:
        # This was the line causing the error: Path was not defined. It is now fixed by the import.
        full_path = Path(output_dir) / filename if output_dir else Path(filename) 
        full_path.parent.mkdir(parents=True, exist_ok=True) 
        plt.savefig(str(full_path), dpi=300, bbox_inches='tight')
        logger.info(f"Saved threshold analysis plot to {full_path}")
        plt.close()
    else:
        plt.show()


def confusion_matrix_with_metrics(y_true, y_pred):
    """
    Calculate confusion matrix and related metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'confusion_matrix': np.array([[0,0],[0,0]]).tolist(), 'accuracy': 0, 'precision': 0, 'recall': 0,
            'specificity': 0, 'f1_score': 0, 'f2_score': 0,
            'true_positives': 0, 'false_positives': 0, 'true_negatives': 0, 'false_negatives': 0
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else: 
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    beta = 2
    f2 = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'confusion_matrix': cm.tolist(), 
        'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall),
        'specificity': float(specificity), 'f1_score': float(f1), 'f2_score': float(f2),
        'true_positives': int(tp), 'false_positives': int(fp), 
        'true_negatives': int(tn), 'false_negatives': int(fn)
    }