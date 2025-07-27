# validation/cross_validator.py

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# --- Add the project root to the system path ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from data.real_data_fetcher import smart_data_fetcher
from legacy_pipeline import prepare_multimodal_data
from models.multimodal_model import build_multimodal_fusion_model
import tensorflow as tf

# Setup a simple logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_kfold_cross_validation(light_curve_files, n_splits=5, threshold=0.5):
    """
    Performs K-Fold cross-validation with class weights to handle imbalance.
    """
    logger.info(f"Preparing full dataset for K-Fold Cross-Validation with threshold={threshold}...")
    
    dummy_labels = pd.DataFrame(light_curve_files)
    X_image, X_timeseries, y = prepare_multimodal_data(light_curve_files, dummy_labels)

    if X_image is None or y is None:
        logger.error("Dataset creation failed. Aborting cross-validation.")
        return

    if X_image.ndim == 3:
        X_image = np.expand_dims(X_image, axis=-1)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_reports = []
    
    logger.info(f"Starting {n_splits}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(y)), y)):
        logger.info(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        X_img_train, X_img_val = X_image[train_idx], X_image[val_idx]
        X_ts_train, X_ts_val = X_timeseries[train_idx], X_timeseries[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- THIS IS THE FIX ---
        # Calculate class weights for the current training fold to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        logger.info(f"Using class weights for Fold {fold + 1}: {class_weight_dict}")
        # ^^^^^^^^^^^^^^^^^^^^^^^^^

        model = build_multimodal_fusion_model(
            image_shape=X_img_train.shape[1:],
            timeseries_shape=X_ts_train.shape[1:]
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        # Apply the class weights during training
        model.fit(
            [X_img_train, X_ts_train], y_train,
            epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
            validation_data=([X_img_val, X_ts_val], y_val),
            class_weight=class_weight_dict, 
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        )

        y_pred_probs = model.predict([X_img_val, X_ts_val])
        y_pred = (y_pred_probs > threshold).astype(int)
        
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        fold_reports.append(report)
        logger.info(f"Fold {fold + 1} Report (Threshold={threshold}):\n{classification_report(y_val, y_pred, zero_division=0)}")

    if not fold_reports:
        logger.error("Cross-validation failed to produce any results.")
        return

    # Calculate and print the average metrics
    avg_precision = np.mean([r['1']['precision'] for r in fold_reports])
    avg_recall = np.mean([r['1']['recall'] for r in fold_reports])
    avg_f1 = np.mean([r['1']['f1-score'] for r in fold_reports])
    avg_accuracy = np.mean([r['accuracy'] for r in fold_reports])
    
    print("\n" + "="*50)
    print(f"      FINAL CROSS-VALIDATION SUMMARY (Threshold={threshold})")
    print("="*50)
    print(f"Number of Folds: {n_splits}")
    print(f"Average Accuracy:  {avg_accuracy:.4f}")
    print("\n--- Average Metrics for Planet Class (1) ---")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1-Score:  {avg_f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run final K-Fold Cross-Validation on the deep learning model.')
    parser.add_argument('--planets_dir', type=str, required=True, help="Directory for confirmed planet light curves.")
    parser.add_argument('--false_positives_dir', type=str, required=True, help="Directory for false positive light curves.")
    parser.add_argument('--n_splits', type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Classification threshold for converting probabilities to labels.")
    args = parser.parse_args()
    
    all_files = smart_data_fetcher(args.planets_dir, args.false_positives_dir)
    
    if all_files:
        run_kfold_cross_validation(all_files, n_splits=args.n_splits, threshold=args.threshold)
    else:
        logger.error("No files found for cross-validation.")