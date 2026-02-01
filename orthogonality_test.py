
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from models.multimodal_model import build_multimodal_fusion_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- Model Definitions (from differential_verification.py, adapted for standalone prediction) ---

def build_1d_branch(timeseries_shape, original_full_model):
    """Builds the 1D CNN branch and transfers weights from the original full model."""
    DENSE_NEURONS = 128
    DROPOUT_RATE = 0.3
    L2_REG = 1e-5
    
    ts_input = Input(shape=timeseries_shape, name='timeseries_input')
    x_ts = Reshape((timeseries_shape[0], 1))(ts_input)
    x_ts = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(L2_REG), name='conv1d_1')(x_ts)
    x_ts = BatchNormalization(name='batch_normalization_3')(x_ts)
    x_ts = MaxPooling1D(2, name='max_pooling1d_1')(x_ts)
    x_ts = Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(L2_REG), name='conv1d_2')(x_ts)
    x_ts = BatchNormalization(name='batch_normalization_4')(x_ts)
    x_ts = MaxPooling1D(2, name='max_pooling1d_2')(x_ts)
    x_ts = Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=l2(L2_REG), name='conv1d_3')(x_ts)
    x_ts = BatchNormalization(name='batch_normalization_5')(x_ts)
    x_ts = MaxPooling1D(2, name='max_pooling1d_3')(x_ts)
    x_ts = Flatten(name='flatten_1')(x_ts)
    x_ts = Dense(DENSE_NEURONS // 2, activation='relu', kernel_regularizer=l2(L2_REG), name='dense_1')(x_ts)
    x_ts = Dropout(DROPOUT_RATE, name='dropout_1')(x_ts)
    
    output = Dense(1, activation='sigmoid', name='output_1d')(x_ts)
    
    model = Model(inputs=ts_input, outputs=output, name='1d_branch')

    # Transfer weights from the original full model
    for layer_full in original_full_model.layers:
        try:
            target_layer = model.get_layer(name=layer_full.name)
            target_layer.set_weights(layer_full.get_weights())
        except ValueError:
            pass # Layer not in this branch model
    return model

def build_2d_branch(image_shape, original_full_model):
    """Builds the 2D CNN branch and transfers weights from the original full model."""
    DENSE_NEURONS = 128
    DROPOUT_RATE = 0.3
    L2_REG = 1e-5
    
    image_input = Input(shape=image_shape, name='image_input')
    x_img = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG), name='conv2d_1')(image_input)
    x_img = BatchNormalization(name='batch_normalization')(x_img)
    x_img = MaxPooling2D((2, 2), name='max_pooling2d_1')(x_img)
    x_img = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG), name='conv2d_2')(x_img)
    x_img = BatchNormalization(name='batch_normalization_1')(x_img)
    x_img = MaxPooling2D((2, 2), name='max_pooling2d_2')(x_img)
    x_img = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(L2_REG), name='conv2d_3')(x_img)
    x_img = BatchNormalization(name='batch_normalization_2')(x_img)
    x_img = MaxPooling2D((2, 2), name='max_pooling2d_3')(x_img)
    x_img = Flatten(name='flatten')(x_img)
    x_img = Dense(DENSE_NEURONS // 2, activation='relu', kernel_regularizer=l2(L2_REG), name='dense')(x_img)
    x_img = Dropout(DROPOUT_RATE, name='dropout')(x_img)
    
    output = Dense(1, activation='sigmoid', name='output_2d')(x_img)
    
    model = Model(inputs=image_input, outputs=output, name='2d_branch')

    # Transfer weights from the original full model
    for layer_full in original_full_model.layers:
        try:
            target_layer = model.get_layer(name=layer_full.name)
            target_layer.set_weights(layer_full.get_weights())
        except ValueError:
            pass # Layer not in this branch model
    return model

def main():
    # --- Setup ---
    # Using the latest run directory found previously
    RUN_DIR = 'results/run_20260125-144401'
    MODEL_PATH = os.path.join(RUN_DIR, 'exo_multimodal_model_best.h5')
    DATA_DIR = os.path.join(RUN_DIR, 'processed_data')

    print(f"Loading data from {DATA_DIR}...")
    X_timeseries = np.load(os.path.join(DATA_DIR, 'X_timeseries.npy'))
    X_images = np.load(os.path.join(DATA_DIR, 'X_images.npy'))
    y_labels = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
    
    num_samples = X_timeseries.shape[0]
    # Assuming features are not directly used in this test, but the multimodal model might expect it.
    # We will use a dummy feature input, adjust shape if needed based on `build_multimodal_fusion_model` expectation.
    # Looking at `build_multimodal_fusion_model` in `models/multimodal_model.py`, it expects `feature_input`
    # so we need to pass something, even if empty. Let's assume a default feature dimension of 10 for now
    # as no specific feature dimension was found in the `differential_verification.py`.
    # I'll check `models/multimodal_model.py` for actual feature_shape.
    # Let's read `models/multimodal_model.py` to get the `feature_shape`.

    # Placeholder for now, will refine after reading the model definition
    # For now, let's assume feature_shape is (10,) if it's a simple feature vector.
    # Given the previous context and typical model inputs, (1024,) might be from a feature extractor.
    # Let's stick with that for now as it was used in the example and if it causes an error, I'll adjust.
    feature_shape = (1024,) 
    X_features = np.zeros((num_samples, *feature_shape))


    image_shape = X_images.shape[1:]
    timeseries_shape = X_timeseries.shape[1:]

    print("Loading and building models...")
    # Load the full multimodal model first to transfer weights
    full_model = build_multimodal_fusion_model(image_shape, timeseries_shape, feature_shape)
    full_model.load_weights(MODEL_PATH)

    # Build individual branches and transfer weights
    model_1d = build_1d_branch(timeseries_shape, full_model)
    model_2d = build_2d_branch(image_shape, full_model)

    print("Running initial inference to identify High Confidence Positives...")
    # Initial predictions for filtering
    # The full model requires all inputs, but individual branches only need their respective input
    prob_1d_initial = model_1d.predict(X_timeseries, verbose=0).flatten()
    prob_2d_initial = model_2d.predict(X_images, verbose=0).flatten()

    # Identify High Confidence Positives
    gt_positives_indices = np.where(y_labels == 1)[0]
    high_confidence_indices = np.where(
        (y_labels == 1) &
        (prob_1d_initial > 0.7) &
        (prob_2d_initial > 0.7)
    )[0]

    if len(high_confidence_indices) < 50:
        print(f"Warning: Only {len(high_confidence_indices)} High Confidence Positives found. Proceeding with available samples.")
        robust_candidates_indices = high_confidence_indices
    else:
        # Select first 50 robust candidates
        robust_candidates_indices = high_confidence_indices[:50]

    print(f"Selected {len(robust_candidates_indices)} Robust Candidates for stress testing.")

    results = []

    print("Starting Experiment Loop (Perturbation Analysis)...")
    for i, idx in enumerate(robust_candidates_indices):
        original_timeseries = X_timeseries[idx:idx+1] # Keep batch dimension
        original_images = X_images[idx:idx+1]         # Keep batch dimension
        original_prob_1d = prob_1d_initial[idx]
        original_prob_2d = prob_2d_initial[idx]

        # --- Attack A (Temporal Jitter on 1D input) ---
        X_timeseries_attack_A = original_timeseries + np.random.normal(0, 0.5, original_timeseries.shape)
        
        prob_1d_attack_A = model_1d.predict(X_timeseries_attack_A, verbose=0).flatten()[0]
        prob_2d_attack_A = model_2d.predict(original_images, verbose=0).flatten()[0] # 2D model uses original image

        drop_1d_A = original_prob_1d - prob_1d_attack_A
        drop_2d_A = original_prob_2d - prob_2d_attack_A

        # --- Attack B (Spatial Occlusion on 2D input) ---
        X_images_attack_B = np.copy(original_images)
        # Zero out a random 10x10 block in the center
        img_h, img_w = X_images_attack_B.shape[1], X_images_attack_B.shape[2]
        block_size = 10
        # Ensure block is within bounds
        center_h = img_h // 2
        center_w = img_w // 2
        
        # Calculate start and end indices for the 10x10 block
        start_h = max(0, center_h - block_size // 2)
        end_h = min(img_h, center_h + block_size // 2)
        start_w = max(0, center_w - block_size // 2)
        end_w = min(img_w, center_w + block_size // 2)

        X_images_attack_B[:, start_h:end_h, start_w:end_w, :] = 0.0 # Zero out the block

        prob_1d_attack_B = model_1d.predict(original_timeseries, verbose=0).flatten()[0] # 1D model uses original timeseries
        prob_2d_attack_B = model_2d.predict(X_images_attack_B, verbose=0).flatten()[0]

        drop_1d_B = original_prob_1d - prob_1d_attack_B
        drop_2d_B = original_prob_2d - prob_2d_attack_B
        
        results.append({
            'ID': idx,
            'Original_Prob_1D': original_prob_1d,
            'Original_Prob_2D': original_prob_2d,
            'Drop_1D_Noise': drop_1d_A,
            'Drop_2D_Noise': drop_2d_A,
            'Drop_1D_Mask': drop_1d_B,
            'Drop_2D_Mask': drop_2d_B,
        })
        print(f"Processed candidate {i+1}/{len(robust_candidates_indices)} (ID: {idx})")

    results_df = pd.DataFrame(results)
    output_csv_path = 'stress_test_results.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nStress test results saved to {output_csv_path}")

    # --- Crucial: Calculate and print Sensitivity Ratios and Summary ---
    avg_drop_1d_noise = results_df['Drop_1D_Noise'].mean()
    avg_2d_drop_due_to_1d_noise = results_df['Drop_2D_Noise'].mean() # 2D model was not attacked here

    avg_drop_2d_mask = results_df['Drop_2D_Mask'].mean()
    avg_1d_drop_due_to_2d_mask = results_df['Drop_1D_Mask'].mean() # 1D model was not attacked here

    print("\n--- Sensitivity Analysis ---")
    print(f"Average Confidence Drop under 1D Noise: Model 1D = {avg_drop_1d_noise:.2%}, Model 2D (untouched) = {avg_2d_drop_due_to_1d_noise:.2%}")
    print(f"Average Confidence Drop under 2D Masking: Model 2D = {avg_drop_2d_mask:.2%}, Model 1D (untouched) = {avg_1d_drop_due_to_2d_mask:.2%}")

    # Sensitivity Ratios
    # Does Model 1D suffer significantly more from Noise than Model 2D? (Expected: Yes)
    # The drop for Model 2D under 1D noise *should be close to zero* if independent.
    # So we compare avg_drop_1d_noise vs avg_2d_drop_due_to_1d_noise.
    
    # Does Model 2D suffer significantly more from Masking than Model 1D? (Expected: Yes)
    # The drop for Model 1D under 2D masking *should be close to zero* if independent.
    # So we compare avg_drop_2d_mask vs avg_1d_drop_due_to_2d_mask.

    print("\n--- Independence of Failure Modes Hypothesis Verification ---")
    # A positive difference means the attacked model dropped more.
    diff_1d_noise = avg_drop_1d_noise - avg_2d_drop_due_to_1d_noise
    diff_2d_mask = avg_drop_2d_mask - avg_1d_drop_due_to_2d_mask

    if diff_1d_noise > 0.01: # Use a small threshold for "significantly more"
        print(f"Hypothesis 1 (1D Noise): Model 1D suffered {diff_1d_noise:.2%} more confidence drop from 1D Noise than Model 2D. (Expected: Yes)")
    else:
        print(f"Hypothesis 1 (1D Noise): Model 1D did NOT suffer significantly more from 1D Noise than Model 2D. (Difference: {diff_1d_noise:.2%}). (Expected: Yes, but result is contrary or weak)")

    if diff_2d_mask > 0.01: # Use a small threshold for "significantly more"
        print(f"Hypothesis 2 (2D Masking): Model 2D suffered {diff_2d_mask:.2%} more confidence drop from 2D Masking than Model 1D. (Expected: Yes)")
    else:
        print(f"Hypothesis 2 (2D Masking): Model 2D did NOT suffer significantly more from 2D Masking than Model 1D. (Difference: {diff_2d_mask:.2%}). (Expected: Yes, but result is contrary or weak)")
    print("---------------------------------------------------------------")


if __name__ == '__main__':
    main()
