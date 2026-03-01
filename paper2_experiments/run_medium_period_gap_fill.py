
import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys

import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config # Import config after sys.path is updated

from data.dataset_generator import create_dataset
from discovery_stack.wrappers import BayesianWrapper
from models.model_trainer_utils import focal_loss

def main():
    """
    Main function to run the medium-period gap-fill experiment.
    """
    # 1. File Indexing
    data_dir = "/Users/emmanuel/proj/phd/kepler_fits_files"
    file_index = {}
    print(f"Indexing files in {data_dir}...")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".fits"):
                match = re.search(r'kplr(\d+)', file)
                if match:
                    target_id = int(match.group(1))
                    file_index[target_id] = os.path.join(root, file)
    print(f"Indexed {len(file_index)} files.")
    indexed_ids = set(file_index.keys())

    # 2. Target Selection
    metadata_path = os.path.join(project_root, "data_files", "full_metadata_v2.csv")
    print(f"Loading metadata from {metadata_path}...")
    metadata_df = pd.read_csv(metadata_path)
    
    # Filter metadata to only include targets for which we have light curve files
    all_light_curve_ids = set(file_index.keys())
    metadata_df_filtered_by_files = metadata_df[metadata_df['target_id'].isin(all_light_curve_ids)]

    medium_period_targets = metadata_df_filtered_by_files[
        (metadata_df_filtered_by_files['period'] > 20) & # Filter for medium periods (20-100 days)
        (metadata_df_filtered_by_files['period'] < 100) & # Filter for medium periods (20-100 days)
        (metadata_df_filtered_by_files['source'] == 'KEPLER')
    ]
    print(f"Metadata contains {len(medium_period_targets)} Medium-Period targets (after file availability filter and modified period filter).")

    # 3. Intersection
    # The filtering above already ensures intersection.
    metadata_ids = set(medium_period_targets['target_id'].astype(int))
    target_ids_to_process = list(indexed_ids.intersection(metadata_ids)) # Re-calculate intersection with newly filtered metadata_ids
    print(f"Found {len(target_ids_to_process)} Medium-Period files ready to process.")

    # 4. Inference Loop
    # Define input shapes (from data/dataset_generator.py and config.py)
    FIXED_LENGTH = config.FIXED_LENGTH # 2048
    IMAGE_SIZE = config.IMAGE_SIZE # (64, 64)
    FEATURE_VECTOR_LENGTH = config.FEATURE_VECTOR_LENGTH # 1024

    # 4. Inference Loop
    print(f"Loading Joint model from {config.MODEL_PATH} with custom objects (FocalLoss)...")
    custom_objects = {'focal_loss_fixed': focal_loss(gamma=config.FOCAL_LOSS_GAMMA, alpha=config.FOCAL_LOSS_ALPHA)}
    joint_model = tf.keras.models.load_model(config.MODEL_PATH, custom_objects=custom_objects)
    print("Joint model loaded.")

    # Create the "1D" (Time-series) Model from the Joint model
    # This extracts the time-series branch and appends a new sigmoid output layer.
    ts_input_tensor = joint_model.inputs[1] # timeseries_input is the second input
    ts_branch_output_tensor = joint_model.get_layer('dropout_1').output # Output of dropout layer after dense in TS branch
    model_1d_base = tf.keras.Model(inputs=ts_input_tensor, outputs=ts_branch_output_tensor, name='1d_base')
    model_1d = tf.keras.Sequential([model_1d_base, tf.keras.layers.Dense(1, activation='sigmoid', name='1d_output')])
    print("1D model (time-series branch with new sigmoid) constructed.")

    # Create the "2D" (Image) Model from the Joint model
    # This extracts the image branch and appends a new sigmoid output layer.
    image_input_tensor = joint_model.inputs[0] # image_input is the first input
    image_branch_output_tensor = joint_model.get_layer('dropout').output # Output of dropout layer after dense in Image branch
    model_2d_base = tf.keras.Model(inputs=image_input_tensor, outputs=image_branch_output_tensor, name='2d_base')
    model_2d = tf.keras.Sequential([model_2d_base, tf.keras.layers.Dense(1, activation='sigmoid', name='2d_output')])
    print("2D model (image branch with new sigmoid) constructed.")

    # --- IMPORTANT NOTE ON 1D/2D MODELS ---
    # The '1D' and '2D' models constructed above reuse the trained weights of the respective
    # branches from the 'joint_model'. However, the final `Dense(1, activation='sigmoid')`
    # layers added to `model_1d` and `model_2d` are *untrained*.
    # For accurate comparison and "disagreement" calculation, it is crucial to understand
    # that these 1D and 2D predictions reflect the branch's feature extraction capability
    # followed by an untrained classification head.
    # If truly independently trained 1D and 2D models are required, they should be loaded
    # from separate saved files. This implementation assumes the intent is to analyze
    # the branch outputs within the context of the overall multimodal model\'s trained
    # feature extractors.


    results = []
    output_csv_path = os.path.join(os.path.dirname(__file__), "survey_results_medium_period.csv")

    for target_id in tqdm(target_ids_to_process, desc="Processing targets"):
        file_path = file_index[target_id]
        
        try:
            # Prepare file_info for create_dataset
            # create_dataset expects a list of dictionaries, each with 'file_path' and 'type'.
            # 'type' is not used for prediction, so a dummy value is fine.
            file_info_for_dataset = [{'file_path': file_path, 'type': 'UNKNOWN'}]
            
            # Create a temporary output directory for create_dataset if it doesn't exist
            temp_dataset_output_dir = os.path.join(project_root, ".temp_dataset_output")
            os.makedirs(temp_dataset_output_dir, exist_ok=True)

            # Atomic Inference - The create_dataset function returns X_ts, X_img, X_features, y, ...
            X_ts_batch, X_img_batch, X_features_batch, _, _, _ = create_dataset(
                file_info_for_dataset, 
                output_dir=temp_dataset_output_dir, 
                image_size=IMAGE_SIZE # Use IMAGE_SIZE from config
            )

            if X_ts_batch is None:
                continue # Skip to the next target_id in the loop

            # create_dataset returns batches, but we are processing one file at a time,
            # so we take the first element from each batch.
            X_ts = X_ts_batch[0][np.newaxis, ...] # Add batch dimension
            X_img = X_img_batch[0][np.newaxis, ...] # Add batch dimension
            X_features = X_features_batch[0][np.newaxis, ...] # Add batch dimension

            # Predict with each model
            # All models expect a batch dimension.
            p_joint = joint_model.predict([X_img, X_ts, X_features])[0][0]
            p_1d = model_1d.predict(X_ts)[0][0]
            p_2d = model_2d.predict(X_img)[0][0]

            # Calculate disagreement
            disagreement_1d = abs(p_joint - p_1d)
            disagreement_2d = abs(p_joint - p_2d)
            
            results.append({
                "target_id": target_id,
                "p_joint": p_joint,
                "p_1d": p_1d,
                "p_2d": p_2d,
                "disagreement_1d": disagreement_1d,
                "disagreement_2d": disagreement_2d,
            })

        except Exception as e:
            print(f"Error processing target {target_id} from {file_path}: {e}")

    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_path, index=False)
        print(f"Survey results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
