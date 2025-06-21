# FILE: evaluate_model.py

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# Configure basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path, file_name):
    """Safely loads a .npy file from the given path."""
    file_to_load = data_path / file_name
    if not file_to_load.exists():
        logger.error(f"Data file not found: {file_to_load}")
        return None
    try:
        return np.load(file_to_load, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error loading data from {file_to_load}: {e}")
        return None

def evaluate_saved_model(run_directory):
    """
    Loads a trained model and test data from a pipeline run directory
    and evaluates its performance.
    """
    run_path = Path(run_directory)
    if not run_path.is_dir():
        logger.error(f"Provided run directory does not exist: {run_path}")
        return

    logger.info(f"--- Starting Evaluation for Pipeline Run: {run_path.name} ---")

    # --- 1. Load the Trained Model ---
    model_path = run_path / "trained_models" / "transit_detector_cnn_best.keras"
    if not model_path.exists():
        logger.error(f"Saved model not found at: {model_path}")
        return

    logger.info(f"Loading model from: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
        # We re-compile the model to get evaluation metrics
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        logger.error(f"Error loading Keras model: {e}", exc_info=True)
        return

    # --- 2. Load the Test Datasets ---
    # The test data is saved by the pipeline runner.
    # Note: Your pipeline saves a list [X_img_test, X_ts_test], so we load the list.
    logger.info("Loading test datasets...")
    X_test_list = load_data(run_path, "X_test.npy")
    y_test = load_data(run_path, "y_test.npy")

    if X_test_list is None or y_test is None:
        logger.error("Could not load test data. Aborting evaluation.")
        return
        
    # We only need the image data for the CNN model
    X_test_images = X_test_list[0]
    logger.info(f"Successfully loaded {len(X_test_images)} test images and {len(y_test)} test labels.")

    # --- 3. Evaluate the Model ---
    logger.info("\n--- Model Evaluation on Test Set ---")
    loss, accuracy = model.evaluate(X_test_images, y_test, verbose=0)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- 4. Generate Predictions and Classification Report ---
    logger.info("\n--- Generating Classification Report ---")
    y_pred_probs = model.predict(X_test_images).ravel()
    # Convert probabilities to binary predictions using a 0.5 threshold
    y_pred_binary = (y_pred_probs > 0.5).astype(int)

    # Print the classification report
    print(classification_report(y_test, y_pred_binary, target_names=['Noise (Class 0)', 'Planet (Class 1)']))

    # --- 5. Display Confusion Matrix ---
    logger.info("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred_binary)
    cm_df = pd.DataFrame(cm,
                         index=['Actual Noise', 'Actual Planet'],
                         columns=['Predicted Noise', 'Predicted Planet'])
    print(cm_df)
    print("\nTN | FP")
    print("FN | TP")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained exoplanet detection model.")
    parser.add_argument(
        "run_directory",
        type=str,
        help="Path to the pipeline run directory containing the model and test data (e.g., 'results/pipeline_run_20250614_181328')."
    )
    args = parser.parse_args()
    
    evaluate_saved_model(args.run_directory)

