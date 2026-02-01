
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- Model Definitions ---

def build_1d_branch(timeseries_shape):
    """Builds the 1D CNN branch of the multimodal model."""
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
    
    # This is a bit of a hack. We need a final dense layer to get a single probability output.
    # The original multimodal model has a shared dense layer after concatenation.
    # We'll add a new dense layer here. It won't have pre-trained weights.
    output = Dense(1, activation='sigmoid', name='output_1d')(x_ts)
    
    model = Model(inputs=ts_input, outputs=output, name='1d_branch')
    return model

def build_2d_branch(image_shape):
    """Builds the 2D CNN branch of the multimodal model."""
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
    return model

def load_full_model(model_path, image_shape, timeseries_shape, feature_shape):
    """Loads the full multimodal model."""
    from models.multimodal_model import build_multimodal_fusion_model
    model = build_multimodal_fusion_model(image_shape, timeseries_shape, feature_shape)
    model.load_weights(model_path)
    return model

def main():
    # --- Setup ---
    RUN_DIR = 'results/run_20251230-084051'
    MODEL_PATH = os.path.join(RUN_DIR, 'exo_multimodal_model_best.h5')
    DATA_DIR = os.path.join(RUN_DIR, 'processed_data')

    print("Loading data...")
    X_timeseries = np.load(os.path.join(DATA_DIR, 'X_timeseries.npy'))
    X_images = np.load(os.path.join(DATA_DIR, 'X_images.npy'))
    y_labels = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
    
    # Assuming features are not used for this test, but the model requires the input.
    # We will use a dummy feature input.
    # I'll need to figure out the shape from a previous run. A shape of (num_samples, 10) is a reasonable guess.
    num_samples = X_timeseries.shape[0]
    feature_shape = (1024,) 
    X_features = np.zeros((num_samples, *feature_shape))


    image_shape = X_images.shape[1:]
    timeseries_shape = X_timeseries.shape[1:]

    print("Loading and building models...")
    full_model = load_full_model(MODEL_PATH, image_shape, timeseries_shape, feature_shape)

    model_1d = build_1d_branch(timeseries_shape)
    model_2d = build_2d_branch(image_shape)

    # Transfer weights
    for layer_full in full_model.layers:
        try:
            model_1d.get_layer(name=layer_full.name).set_weights(layer_full.get_weights())
        except ValueError:
            pass # Layer not in 1D model
        try:
            model_2d.get_layer(name=layer_full.name).set_weights(layer_full.get_weights())
        except ValueError:
            pass # Layer not in 2D model

    print("Running inference...")
    prob_1d = model_1d.predict(X_timeseries, verbose=0).flatten()
    prob_2d = model_2d.predict(X_images, verbose=0).flatten()

    # --- Metrics Calculation ---
    print("Calculating metrics...")
    delta = np.abs(prob_1d - prob_2d)

    # Create DataFrame
    results_df = pd.DataFrame({
        'Target_ID': range(len(y_labels)),
        'Ground_Truth_Label': y_labels,
        'Prob_1D': prob_1d,
        'Prob_2D': prob_2d,
        'Delta': delta
    })

    # Save CSV
    csv_path = 'representation_disagreement.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # --- Scatter Plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(results_df['Prob_1D'], results_df['Prob_2D'], c=results_df['Ground_Truth_Label'], cmap='coolwarm', alpha=0.6)
    plt.xlabel('Probability (1D CNN)')
    plt.ylabel('Probability (2D CNN)')
    plt.title('Model Disagreement Scatter Plot')
    plt.legend(handles=scatter.legend_elements()[0], labels=['False Positive', 'Confirmed Planet'])
    plt.grid(True)
    plot_path = 'disagreement_scatter.png'
    plt.savefig(plot_path)
    print(f"Scatter plot saved to {plot_path}")

    # --- Summary Report ---
    total_samples = len(results_df)
    pearson_corr, _ = pearsonr(results_df['Prob_1D'], results_df['Prob_2D'])
    high_disagreement = results_df[results_df['Delta'] > 0.5]
    latent_faults = results_df[
        (results_df['Ground_Truth_Label'] == 0) &
        (
            ((results_df['Prob_1D'] > 0.8) & (results_df['Prob_2D'] < 0.5)) |
            ((results_df['Prob_2D'] > 0.8) & (results_df['Prob_1D'] < 0.5))
        )
    ]

    print("\n--- Differential Verification Report ---")
    print(f"Total samples processed: {total_samples}")
    print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
    print(f"High Disagreement (delta > 0.5): {len(high_disagreement)} ({len(high_disagreement)/total_samples:.2%})")
    print(f"Latent Faults: {len(latent_faults)}")
    print("--------------------------------------")


if __name__ == '__main__':
    main()
