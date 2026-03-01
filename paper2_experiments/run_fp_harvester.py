import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
from glob import glob
from pathlib import Path
import logging # Moved this import to the top
from astropy.io import fits, ascii
from skimage.transform import resize
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle


# --- Local Project Imports ---
# Assuming config.py is in the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

# Assuming data.augmentation_utils exists, though not directly used in create_dataset here
# from data.augmentation_utils import advanced_phase_folding
# Assuming detection.periodicity_analyzer and detection.transit_detector exist
# from detection.periodicity_analyzer import analyze_periodicity
# from detection.transit_detector import find_transits_bls, estimate_planet_properties


# --- Constants (for files not specified in config.py) ---
FP_CSV_PATH = 'false_positives.csv' # This is in the project root
METADATA_FILENAME = 'full_metadata_v2.csv'
KEPLER_FITS_SUBDIR_NAME = 'kepler_fits_files' # Name of the subdirectory for Kepler FITS files

# --- Configuration ---
NUM_TARGETS_TO_HARVEST = 500

# Global logger for this script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Keras Model Definitions (copied from convert_cacl_to_keras.py) ---
@tf.keras.utils.register_keras_serializable(name="KerasTransformerEncoder")
class KerasTransformerEncoder(tf.keras.Model):
    def __init__(self, input_dim, head_size, num_heads, ff_dim, num_layers, **kwargs):
        super(KerasTransformerEncoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        self.input_proj = tf.keras.layers.Dense(head_size * num_heads)

        self.attention_layers = []
        self.attention_dropout_layers = []
        self.attention_norm_layers = []
        self.ffn_conv1d_1_layers = []
        self.ffn_dropout_layers = []
        self.ffn_conv1d_2_layers = []
        self.ffn_norm_layers = []

        for _ in range(num_layers):
            self.attention_layers.append(tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size))
            self.attention_dropout_layers.append(tf.keras.layers.Dropout(0.1))
            self.attention_norm_layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.ffn_conv1d_1_layers.append(tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu"))
            self.ffn_dropout_layers.append(tf.keras.layers.Dropout(0.1))
            self.ffn_conv1d_2_layers.append(tf.keras.layers.Conv1D(filters=head_size * num_heads, kernel_size=1))
            self.ffn_norm_layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))

    def call(self, inputs):
        x = self.input_proj(inputs)
        x = tf.expand_dims(x, axis=1) # [B, D] -> [B, 1, D]

        for i in range(self.num_layers):
            # MultiHeadAttention
            attention_output = self.attention_layers[i](x, x)
            attention_output = self.attention_dropout_layers[i](attention_output)
            x = self.attention_norm_layers[i](x + attention_output)

            # Feed Forward Network
            ffn_output = self.ffn_conv1d_1_layers[i](x)
            ffn_output = self.ffn_dropout_layers[i](ffn_output)
            ffn_output = self.ffn_conv1d_2_layers[i](ffn_output)
            x = self.ffn_norm_layers[i](x + ffn_output)
            
        return tf.squeeze(x, axis=1) # [B, 1, D] -> [B, D]

    def get_config(self):
        config = super(KerasTransformerEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# --- Data Processing Functions (adapted from data/dataset_generator.py) ---
def create_dataset_for_inference(file_path):
    """
    Processes a single light curve file for inference, returning 1D time-series, 2D image, and engineered features.
    """
    try:
        if file_path.endswith(('.fits', '.npz')):
            with fits.open(file_path, mode='readonly', ignore_missing_simple=True) as hdul:
                data = hdul[1].data
                time_lc = data.field('TIME')
                flux_lc = data.field('PDCSAP_FLUX')
        elif file_path.endswith('.tbl'):
            table = ascii.read(file_path)
            time_lc = table['TIME'].value
            flux_lc = table['PDCSAP_FLUX'].value
            logger.debug(f"Successfully read .tbl file: {os.path.basename(file_path)}")
        else:
            logger.warning(f"Skipping {os.path.basename(file_path)}: Unsupported file type.")
            return None, None, None

        finite_mask = np.isfinite(flux_lc) & np.isfinite(time_lc)
        time_lc = time_lc[finite_mask]
        flux_lc = flux_lc[finite_mask]

        if len(flux_lc) < 100: # Minimum data points for meaningful analysis
            logger.warning(f"Skipping {os.path.basename(file_path)}: Not enough finite data points ({len(flux_lc)} < 100).")
            return None, None, None

        # Normalize flux
        flux_norm = (flux_lc - np.nanmedian(flux_lc)) / (np.nanstd(flux_lc) + 1e-8) # Add epsilon to prevent division by zero

        # 1D Time-series Segment (X_ts)
        start = max(0, len(flux_norm) // 2 - config.FIXED_LENGTH // 2)
        segment = flux_norm[start : start + config.FIXED_LENGTH]
        if len(segment) < config.FIXED_LENGTH:
            segment = np.pad(segment, (0, config.FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
        X_ts = segment.astype(np.float32)

        # Feature Engineering (X_features - using LombScargle power as an example)
        # Placeholder for period as it's typically derived from transit detection, here we'll use a dummy or skip
        # For simplicity in this inference function, we'll use a basic feature vector
        # This part should ideally align with what the original model was trained on
        if len(time_lc) > 1:
            # Using a simplified LombScargle for feature vector, without full transit detection logic
            try:
                frequency, power = LombScargle(time_lc, flux_lc).autopower(minimum_frequency=1/1000, maximum_frequency=1/0.1)
                # Resize power spectrum to config.FEATURE_VECTOR_LENGTH
                feature_vector = resize(power, (config.FEATURE_VECTOR_LENGTH,), preserve_range=True, anti_aliasing=False).astype(np.float32)
            except Exception as e:
                logger.warning(f"LombScargle failed for {os.path.basename(file_path)}: {e}. Using zeros for features.")
                feature_vector = np.zeros(config.FEATURE_VECTOR_LENGTH, dtype=np.float32)
        else:
            feature_vector = np.zeros(config.FEATURE_VECTOR_LENGTH, dtype=np.float32)
        X_features = feature_vector

        # 2D Image Generation (X_img - simplified phase folding for fixed input)
        # This part assumes a known period or approximates it if not detected, for consistency
        # For this harvester, we'll create a simple 2D representation from the 1D segment
        img_1d = segment[:config.IMAGE_SIZE[0] * config.IMAGE_SIZE[1]]
        if len(img_1d) < config.IMAGE_SIZE[0] * config.IMAGE_SIZE[1]:
            img_1d = np.pad(img_1d, (0, config.IMAGE_SIZE[0] * config.IMAGE_SIZE[1] - len(img_1d)), 'constant', constant_values=0)
        img_2d = img_1d.reshape(config.IMAGE_SIZE)
        X_img = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8) # Normalize to 0-1
        X_img = X_img[..., np.newaxis].astype(np.float32) # Add channel dimension

        return X_img, X_ts, X_features

    except Exception as e:
        logger.error(f"FAILED to process {os.path.basename(file_path)}. Error: {e}. Skipping.", exc_info=True)
        return None, None, None


def run_fp_harvester():
    logger.info("Starting False Positive Harvester for Paper 2 experiments.")

    # 0. Load the Multimodal Keras Model
    MODEL_PATH = config.MODEL_PATH
    if not MODEL_PATH.exists():
        logger.error(f"Multimodal Keras model not found at {MODEL_PATH}. Aborting.")
        return

    # Custom objects needed for loading Keras models that use custom layers
    custom_objects = {'KerasTransformerEncoder': KerasTransformerEncoder}
    
    try:
        multimodal_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        logger.info(f"Multimodal Keras model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load multimodal Keras model from {MODEL_PATH}: {e}. Aborting.")
        return

    # 1. Global Indexing: Walk through all data directories to find FITS files
    logger.info(f"Globally indexing FITS files in {config.DATA_DIR} and {config.BASE_DIR / KEPLER_FITS_SUBDIR_NAME}...")
    file_index = {}
    
    # Load false_positives.csv from project root
    fp_csv_path_root = config.BASE_DIR / FP_CSV_PATH
    fp_df_from_csv = pd.read_csv(fp_csv_path_root)
    fp_kepid_list_from_csv = set(fp_df_from_csv['kepid'].astype(int).tolist())

    # Search in config.DATA_DIR (e.g., data_files/light_curves)
    all_fits_files_data_dir = glob(os.path.join(config.DATA_DIR, '**', '*.fits'), recursive=True)
    # Search in config.BASE_DIR / KEPLER_FITS_SUBDIR_NAME (e.g., kepler_fits_files)
    all_fits_files_kepler_dir = glob(os.path.join(config.BASE_DIR / KEPLER_FITS_SUBDIR_NAME, '**', '*.fits'), recursive=True)

    all_fits_files = all_fits_files_data_dir + all_fits_files_kepler_dir
    
    for f_path in tqdm(all_fits_files, desc="Building file index"):
        p = Path(f_path)
        
        # Extract KEPID from filename (e.g., kplr010848459-2013131215648_llc.fits)
        match = re.search(r'kplr(\d+)-', p.name)
        if match:
            kepid = int(match.group(1))
            if kepid in fp_kepid_list_from_csv: # Only index FITS files that are in our false_positives.csv list
                file_index[kepid] = f_path
        else:
            logger.debug(f"Could not extract KEPID from filename: {p.name}")

    logger.info(f"Indexed {len(file_index)} total FITS files that are relevant to False Positives.")

    # 2. Metadata Target List: Load and filter metadata for False Positives
    logger.info(f"Loading metadata from {config.DATA_DIR / METADATA_FILENAME}...")
    metadata_df = pd.read_csv(config.DATA_DIR / METADATA_FILENAME)
    
    verified_fps_metadata = metadata_df[metadata_df['true_label'] == 'FALSE_POSITIVE']
    verified_fp_ids_from_metadata = set(verified_fps_metadata['target_id'].astype(int).tolist())
    logger.info(f"Metadata has {len(verified_fp_ids_from_metadata)} False Positives.")

    # 3. The Intersection: Find common IDs and apply limit
    common_ids = list(file_index.keys() & verified_fp_ids_from_metadata)
    
    logger.info(f"Found {len(common_ids)} False Positive files across all folders that match metadata.")

    selected_ids = common_ids
    if len(common_ids) > NUM_TARGETS_TO_HARVEST:
        logger.info(f"Limiting to {NUM_TARGETS_TO_HARVEST} randomly sampled targets.")
        selected_ids = random.sample(common_ids, NUM_TARGETS_TO_HARVEST)
    else:
        logger.info(f"Processing all {len(common_ids)} available False Positive targets.")
    
    logger.info(f"Harvesting {len(selected_ids)} targets for inference...")

    # 4. Inference Loop
    results_df = pd.DataFrame(columns=['kepid', 'p_joint', 'p_1d', 'p_2d', 'disagreement'])

    # Prepare zero-arrays for partial predictions
    # This requires processing at least one valid file to get the shapes
    first_valid_kepid = None
    for kepid in selected_ids:
        if kepid in file_index:
            first_valid_kepid = kepid
            break

    if first_valid_kepid is None:
        logger.error("No valid FITS files found for selected False Positives. Aborting.")
        return

    dummy_img, dummy_ts, dummy_features = create_dataset_for_inference(file_index[first_valid_kepid])
    if dummy_img is None:
        logger.error("Failed to create dummy inputs for zero arrays from first valid file. Aborting.")
        return

    zeros_img = np.expand_dims(np.zeros_like(dummy_img), axis=0)
    zeros_ts = np.expand_dims(np.zeros_like(dummy_ts), axis=0)
    zeros_features = np.expand_dims(np.zeros_like(dummy_features), axis=0)

    for kepid in tqdm(selected_ids, desc="Running inference"):
        file_path = file_index.get(kepid)
        if file_path is None:
            logger.warning(f"File path for KEPID {kepid} not found in index. Skipping.")
            continue
        
        try:
            X_img_input, X_ts_input, X_features_input = create_dataset_for_inference(file_path)
            
            if X_img_input is None or X_ts_input is None or X_features_input is None:
                logger.warning(f"Failed to create dataset inputs for KEPID {kepid}. Skipping.")
                continue

            # Ensure batch dimension for prediction
            X_img_input = np.expand_dims(X_img_input, axis=0)
            X_ts_input = np.expand_dims(X_ts_input, axis=0)
            X_features_input = np.expand_dims(X_features_input, axis=0)

            # Atomic Inference
            p_joint = multimodal_model.predict([X_img_input, X_ts_input, X_features_input], verbose=0)[0][0]
            
            # For p_1d, zero out the image input and feature input
            p_1d = multimodal_model.predict([zeros_img, X_ts_input, zeros_features], verbose=0)[0][0]
            
            # For p_2d, zero out the time-series input and feature input
            p_2d = multimodal_model.predict([X_img_input, zeros_ts, zeros_features], verbose=0)[0][0]
            
            disagreement = abs(p_1d - p_2d)

            # Append results to DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([{
                'kepid': kepid,
                'p_joint': p_joint,
                'p_1d': p_1d,
                'p_2d': p_2d,
                'disagreement': disagreement
            }])], ignore_index=True)

        except Exception as e:
            logger.error(f"Error processing KEPID {kepid} from {file_path}: {e}", exc_info=True)
            continue

    # Save results to CSV
    os.makedirs(config.SURVEY_RESULTS_DIR, exist_ok=True)
    results_df.to_csv(config.SURVEY_RESULTS_DIR / 'survey_results_fps.csv', index=False)
    logger.info(f"Inference results saved to {config.SURVEY_RESULTS_DIR / 'survey_results_fps.csv'}")

if __name__ == '__main__':
    run_fp_harvester()
