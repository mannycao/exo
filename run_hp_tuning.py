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
from pathlib import Path
import sys
from tqdm import tqdm # ADDED THIS IMPORT

# NEW IMPORTS for data preparation and parallelization
import config
from data.dataset_generator import create_dataset
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Moved build_simple_model from legacy_pipeline.py
def build_simple_model(dense_neurons=128, dropout_rate=0.3, learning_rate=0.001, meta=None):
    """
    Builds a simplified single-input model for hyperparameter tuning.
    The 'meta' parameter is required by scikeras to pass metadata like input shapes.
    """
    n_features_in_ = meta["n_features_in_"]
    
    model = Sequential([
        Input(shape=(n_features_in_,)),
        Dense(dense_neurons, activation='relu'),
        Dropout(dropout_rate),
        Dense(dense_neurons // 2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def setup_logging():
    """
    Configure basic logging.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def train_and_evaluate_fold(fold_data, params, X_combined_shape):
    """
    Helper function to train and evaluate a single fold.
    """
    X_train, X_val, y_train, y_val, fold_info = fold_data
    logger = logging.getLogger(__name__) # Get logger for this thread

    logger.info(f"  -- Fold {fold_info} --")

    # Build, compile, and train the model
    model = build_simple_model(
        dense_neurons=params['dense_neurons'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate'],
        meta={'n_features_in_': X_combined_shape[1]}
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
    return score

def main():
    """
    Main function to run manual hyperparameter optimization.
    """
    parser = argparse.ArgumentParser(description='Run manual hyperparameter optimization for the exoplanet model.')
    parser.add_argument('--planets_dir', type=str, required=True, help="Directory for confirmed planet light curves.")
    parser.add_argument('--false_positives_dir', type=str, required=True, help="Directory for false positive light curves.")
    parser.add_argument('--n_iter', type=int, default=10, help='Number of parameter combinations to try.')
    parser.add_argument('--cv', type=int, default=3, help='Number of cross-validation folds.')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(), # New argument for parallelization
                        help='Maximum number of concurrent workers for cross-validation folds.')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting MANUAL hyperparameter optimization...")

    # --- Step 1: Prepare Data ---
    logger.info("Fetching local data files...")
    # Add project root to sys.path for imports
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.real_data_fetcher import smart_data_fetcher # Moved import here to avoid circular dependency with config

    typed_light_curve_files = smart_data_fetcher(
        confirmed_planet_dir=args.planets_dir,
        false_positive_dir=args.false_positives_dir
    )
    if not typed_light_curve_files:
        logger.error("No light curve files found. Aborting.")
        return

    # Load exoplanet metadata for period information
    metadata_path = project_root / "data" / "metadata" / "exoplanet_labels.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}. Aborting.")
        return
    exoplanet_metadata_df = pd.read_csv(metadata_path)

    # NEW DATA PREPARATION LOGIC (similar to run_bayesian_inference.py)
    processed_data_temp_dir = Path("./temp_hp_processed_data") # Temporary directory for processed data
    processed_data_temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing multimodal dataset from {len(typed_light_curve_files)} files...")
    create_dataset(
        file_paths=[item['file_path'] for item in typed_light_curve_files],
        labels=[item['type'] for item in typed_light_curve_files],
        output_dir=processed_data_temp_dir,
        metadata_df=exoplanet_metadata_df,
        image_size=config.IMAGE_SIZE # Use IMAGE_SIZE from config
    )

    try:
        X_image = np.load(processed_data_temp_dir / 'X_images.npy')
        X_timeseries = np.load(processed_data_temp_dir / 'X_timeseries.npy')
        y = np.load(processed_data_temp_dir / 'y_labels.npy')
    except FileNotFoundError:
        logger.error("Could not find multimodal dataset files in temp directory. Aborting.")
        shutil.rmtree(processed_data_temp_dir) # Clean up even on error
        return

    # Clean up the temporary processed data directory after loading
    shutil.rmtree(processed_data_temp_dir)

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

    # --- Step 3: Manual Random Search Loop (Parallelized) ---
    results = []
    best_score = -1
    best_params = {}

    for i in range(args.n_iter):
        params = {key: random.choice(value) for key, value in param_grid.items()}
        logger.info(f"\n--- Trial {i+1}/{args.n_iter} ---\nParameters: {params}")

        # K-Fold Cross-Validation
        kfold = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        
        fold_data_list = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_combined, y)):
            X_train, X_val = X_combined[train_idx], X_combined[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            fold_data_list.append((X_train, X_val, y_train, y_val, f"{fold+1}/{args.cv}"))

        # Parallelize fold training
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Use partial to pass fixed arguments to the helper function
            train_func = partial(train_and_evaluate_fold, params=params, X_combined_shape=X_combined.shape)
            
            # Map the function to the list of fold data
            fold_scores = list(tqdm(executor.map(train_func, fold_data_list), 
                                    total=len(fold_data_list), 
                                    desc=f"Trial {i+1} Folds"))

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