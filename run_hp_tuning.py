# run_hp_tuning.py

import logging
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

from data.real_data_fetcher import smart_data_fetcher
from legacy_pipeline import prepare_multimodal_data, build_simple_model

def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """Main function to run manual hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Run manual hyperparameter optimization for the exoplanet model.')
    parser.add_argument('--planets_dir', type=str, required=True, help="Directory for confirmed planet light curves.")
    parser.add_argument('--false_positives_dir', type=str, required=True, help="Directory for false positive light curves.")
    parser.add_argument('--n_iter', type=int, default=10, help='Number of parameter combinations to try.')
    parser.add_argument('--cv', type=int, default=3, help='Number of cross-validation folds.')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting MANUAL hyperparameter optimization...")

    # --- Step 1: Prepare Data ---
    logger.info("Fetching local data files...")
    typed_light_curve_files = smart_data_fetcher(
        confirmed_planet_dir=args.planets_dir,
        false_positive_dir=args.false_positives_dir
    )
    if not typed_light_curve_files:
        logger.error("No light curve files found. Aborting.")
        return

    exoplanet_labels = pd.DataFrame(typed_light_curve_files)
    logger.info(f"Preparing multimodal dataset from {len(typed_light_curve_files)} files...")
    X_image, X_timeseries, y = prepare_multimodal_data(typed_light_curve_files, exoplanet_labels)

    if X_image is None:
        logger.error("Failed to create dataset. Aborting.")
        return

    # Combine features for the simple model
    X_image_flat = X_image.reshape(X_image.shape[0], -1)
    X_combined = np.concatenate([X_image_flat, X_timeseries], axis=1)

    # --- Step 2: Define Hyperparameter Grid ---
    param_grid = {
        'dense_neurons': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'batch_size': [16, 32, 64],
        'epochs': [15, 25, 40]
    }

    # --- Step 3: Manual Random Search Loop ---
    results = []
    best_score = -1
    best_params = {}

    for i in range(args.n_iter):
        params = {key: random.choice(value) for key, value in param_grid.items()}
        logger.info(f"\n--- Trial {i+1}/{args.n_iter} ---")
        logger.info(f"Parameters: {params}")

        # K-Fold Cross-Validation
        kfold = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_combined, y)):
            logger.info(f"  -- Fold {fold+1}/{args.cv} --")
            X_train, X_val = X_combined[train_idx], X_combined[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Build, compile, and train the model
            model = build_simple_model(
                dense_neurons=params['dense_neurons'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate'],
                meta={'n_features_in_': X_combined.shape[1]}
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_data=(X_val, y_val),
                verbose=0, # Set to 1 for more detail
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
            )
            
            # Evaluate on the validation fold
            score = model.evaluate(X_val, y_val, verbose=0)[1] # Get accuracy
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        logger.info(f"-> Average Validation Accuracy for Trial {i+1}: {avg_score:.4f}")
        results.append({'params': params, 'score': avg_score})

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # --- Step 4: Display Final Results ---
    logger.info("\n" + "="*50)
    logger.info("Hyperparameter Optimization Finished")
    logger.info("="*50)
    logger.info(f"Best Validation Accuracy: {best_score:.4f}")
    logger.info("Best parameters found:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

if __name__ == "__main__":
    main()