"""
Evaluation metrics for exoplanet detection models.
"""

import logging
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
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
        modulating_factor = K.pow(1.0 - p_t, gamma)
        loss = alpha_factor * modulating_factor * cross_entropy
        return K.mean(loss, axis=-1)
    return focal_loss_fixed


def get_uncertainty_metrics(y_true, y_pred_probs):
    """
    Calculates metrics related to model calibration and uncertainty.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred_probs (np.ndarray): Predicted probabilities.

    Returns:
        dict: A dictionary containing the ECE and Brier score.
    """
    if len(y_true) == 0 or len(y_pred_probs) == 0:
        return {'expected_calibration_error': -1, 'brier_score': -1}

    # Expected Calibration Error (ECE)
    prob_true, prob_pred = calibration_curve(y_true, y_pred_probs, n_bins=10, strategy='uniform')
    ece = np.mean(np.abs(prob_true - prob_pred))

    # Brier Score
    brier = brier_score_loss(y_true, y_pred_probs)

    return {
        'expected_calibration_error': ece,
        'brier_score': brier
    }

