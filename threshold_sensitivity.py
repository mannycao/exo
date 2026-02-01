
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_threshold_sensitivity_analysis():
    # --- Setup ---
    disagreement_csv_path = 'representation_disagreement.csv'

    if not os.path.exists(disagreement_csv_path):
        print(f"Error: {disagreement_csv_path} not found. Please ensure Experiment 1 (differential_verification.py) has been run to generate it.")
        # For the purpose of this script, if the file is missing, we'll try to generate it.
        # This assumes the necessary data and models are available in the expected RUN_DIR.
        print("Attempting to regenerate representation_disagreement.csv...")
        # We need to replicate the logic from differential_verification.py or orthogonality_test.py
        # to generate the required CSV.

        # Re-using logic from orthogonality_test.py / differential_verification.py
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, Conv2D, MaxPooling2D, BatchNormalization
        from tensorflow.keras.regularizers import l2
        from models.multimodal_model import build_multimodal_fusion_model

        # Suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')

        # --- Model Definitions (from differential_verification.py, adapted for standalone prediction) ---

        def build_1d_branch_for_regen(timeseries_shape, original_full_model):
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

        def build_2d_branch_for_regen(image_shape, original_full_model):
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

        RUN_DIR = 'results/run_20260125-144401' # Using the latest run directory found previously
        MODEL_PATH = os.path.join(RUN_DIR, 'exo_multimodal_model_best.h5')
        DATA_DIR = os.path.join(RUN_DIR, 'processed_data')

        print(f"Loading data from {DATA_DIR} to regenerate {disagreement_csv_path}...")
        X_timeseries = np.load(os.path.join(DATA_DIR, 'X_timeseries.npy'))
        X_images = np.load(os.path.join(DATA_DIR, 'X_images.npy'))
        y_labels = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
        
        num_samples = X_timeseries.shape[0]
        feature_shape = (1024,)
        X_features = np.zeros((num_samples, *feature_shape))

        image_shape = X_images.shape[1:]
        timeseries_shape = X_timeseries.shape[1:]

        print("Loading and building models for regeneration...")
        full_model = build_multimodal_fusion_model(image_shape, timeseries_shape, feature_shape)
        full_model.load_weights(MODEL_PATH)

        model_1d_regen = build_1d_branch_for_regen(timeseries_shape, full_model)
        model_2d_regen = build_2d_branch_for_regen(image_shape, full_model)

        print("Running inference to regenerate disagreement data...")
        prob_1d = model_1d_regen.predict(X_timeseries, verbose=0).flatten()
        prob_2d = model_2d_regen.predict(X_images, verbose=0).flatten()

        delta = np.abs(prob_1d - prob_2d)

        results_df_regen = pd.DataFrame({
            'Target_ID': range(len(y_labels)),
            'Ground_Truth_Label': y_labels,
            'Prob_1D': prob_1d,
            'Prob_2D': prob_2d,
            'Delta': delta
        })
        results_df_regen.to_csv(disagreement_csv_path, index=False)
        print(f"Successfully regenerated {disagreement_csv_path}.")
        df = results_df_regen
    else:
        df = pd.read_csv(disagreement_csv_path)
        print(f"Loaded existing {disagreement_csv_path}.")

    # --- Sweep Loop ---
    tau_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    analysis_results = []

    for tau in tau_values:
        oracle_flags = df[df['Delta'] > tau]
        true_flags = oracle_flags[oracle_flags['Ground_Truth_Label'] == 0]
        false_flags = oracle_flags[oracle_flags['Ground_Truth_Label'] == 1]

        count_true_flags = len(true_flags)
        count_false_flags = len(false_flags)
        total_flags = count_true_flags + count_false_flags

        precision = count_true_flags / total_flags if total_flags > 0 else 0.0
        yield_count = count_true_flags

        analysis_results.append({
            'Threshold': tau,
            'Precision': precision,
            'Yield': yield_count,
            'False_Alarms': count_false_flags
        })

    results_df = pd.DataFrame(analysis_results)
    output_csv_path = 'threshold_sensitivity.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nSensitivity analysis results saved to {output_csv_path}")

    # --- Print Summary Table ---
    print("\n--- Threshold Sensitivity Analysis Summary ---")
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("--------------------------------------------")

    # --- Identify "Sweet Spot" ---
    sweet_spot_threshold = None
    best_yield_at_sweet_spot = -1
    
    # Filter for thresholds less than 0.5 and precision >= 0.9
    potential_sweet_spots = results_df[
        (results_df['Threshold'] < 0.5) & 
        (results_df['Precision'] >= 0.9)
    ].sort_values(by='Yield', ascending=False) # Prioritize higher yield

    if not potential_sweet_spots.empty:
        # The first entry after sorting by Yield (descending) will be the best sweet spot
        sweet_spot_threshold = potential_sweet_spots.iloc[0]['Threshold']
        best_yield_at_sweet_spot = potential_sweet_spots.iloc[0]['Yield']
        sweet_spot_precision = potential_sweet_spots.iloc[0]['Precision']
        print(f"\n--- Sweet Spot Identified ---")
        print(f"Threshold (tau) lower than 0.5 that catches more bugs without dropping Precision below 90%:")
        print(f"  Sweet Spot Threshold: {sweet_spot_threshold:.1f}")
        print(f"  Corresponding Precision: {sweet_spot_precision:.2%}")
        print(f"  Corresponding Yield (True Flags): {int(best_yield_at_sweet_spot)}")
        print("---------------------------")
    else:
        print("\nNo 'Sweet Spot' found (Threshold < 0.5 and Precision >= 90%).")

    # --- Generate Plotting Code ---
    print("\n--- Plotting Code (copy and run in a Python environment with matplotlib) ---")
    plotting_code = f"""
import pandas as pd
import matplotlib.pyplot as plt

# Load the generated results
results_df = pd.read_csv('{output_csv_path}')

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Disagreement Threshold (tau)')
ax1.set_ylabel('Precision', color=color)
ax1.plot(results_df['Threshold'], results_df['Precision'], color=color, marker='o', label='Precision')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([-0.05, 1.05]) # Ensure y-axis covers full probability range

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Yield (True Flags)', color=color)  # we already handled the x-label with ax1
ax2.plot(results_df['Threshold'], results_df['Yield'], color=color, marker='x', label='Yield')
ax2.tick_params(axis='y', labelcolor=color)

# Add a horizontal line for 90% precision target
ax1.axhline(y=0.90, color='gray', linestyle='--', label='90% Precision Target')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Disagreement Threshold Sensitivity Analysis')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.grid(True)
plt.show()
"""
    print(plotting_code)
    print("------------------------------------------------------------------")

if __name__ == '__main__':
    run_threshold_sensitivity_analysis()
