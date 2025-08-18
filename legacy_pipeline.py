# legacy_pipeline.py

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

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

def preprocess_light_curve(file_path):
    """Loads and preprocesses light curve FITS files."""
    try:
        from astropy.io import fits
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            time, flux = data['TIME'], data['PDCSAP_FLUX']
            mask = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[mask], flux[mask]
            
            if len(time) == 0:
                return None, None
                
            normalized_flux = (flux - np.median(flux)) / (np.std(flux) + 1e-8)
            return time, normalized_flux
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def phase_fold(time, flux, period, t0):
    """
    Phase-folds a light curve given time, flux, period, and reference epoch.
    """
    if period <= 0:
        raise ValueError("Period must be positive.")
    
    phase = ((time - t0) % period) / period
    return phase, flux

def create_image_representations(time, flux, period, t0, img_size=(64, 64)):
    """
    Creates a 2D image representation from a light curve using phase folding.
    """
    if time is None or flux is None or len(time) == 0 or len(flux) == 0:
        return None

    try:
        folded_phase, folded_flux = phase_fold(time, flux, period, t0)
        
        # Sort by phase for consistent image representation
        sort_indices = np.argsort(folded_phase)
        folded_phase_sorted = folded_phase[sort_indices]
        folded_flux_sorted = folded_flux[sort_indices]

        # Create a 2D image from the phase-folded light curve
        # This is a simplified approach; more sophisticated methods might involve binning or interpolation
        # For now, we'll just reshape the sorted folded flux
        img_1d_folded = folded_flux_sorted[:img_size[0] * img_size[1]]
        if len(img_1d_folded) < img_size[0] * img_size[1]:
            img_1d_folded = np.pad(img_1d_folded, (0, img_size[0] * img_size[1] - len(img_1d_folded)), 'constant', constant_values=0)
        
        img_2d = img_1d_folded.reshape(img_size)

        # Normalize image to [0, 1] for the CNN
        img_norm = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8)
        
        return img_norm
    except Exception as e:
        print(f"Error creating image representation: {e}")
        return None

def extract_transit_features(time, flux, window_size=2048):
    """Extracts a fixed-length segment from the light curve."""
    if flux is None or len(flux) < 100:
        return None
    
    if len(flux) < window_size:
        pad_width = window_size - len(flux)
        flux = np.pad(flux, (0, pad_width), 'constant', constant_values=0)

    center_idx = len(flux) // 2
    start_idx = max(0, center_idx - window_size // 2)
    segment = flux[start_idx : start_idx + window_size]
    return [segment]

def prepare_multimodal_data(light_curve_files, metadata_df):
    """
    Prepares multimodal dataset from a list of file dictionaries.
    """
    image_data = []
    timeseries_data = []
    labels = []
    successful_files = []
    
    for item in light_curve_files:
        file_path = item['file_path']
        label_type = item['type']
        
        try:
            time, flux = preprocess_light_curve(file_path)
            if time is None or flux is None:
                continue
            
            # --- Determine Period and t0 for Phase Folding ---
            period = None
            t0 = 0.0 # Default t0 for now

            if label_type == 'confirmed_planet':
                filename_parts = os.path.basename(file_path).split('-')
                if len(filename_parts) > 0:
                    kepler_id = filename_parts[0]
                    if kepler_id.startswith('kplr'):
                        kepler_id = kepler_id[4:]
                    
                    matching_rows = metadata_df[metadata_df['pl_name'].str.contains(kepler_id, na=False)]
                    if not matching_rows.empty:
                        period = matching_rows['pl_orbper'].iloc[0]
                        if pd.isna(period):
                            period = np.random.uniform(1, 100) # Fallback
                            print(f"Warning: Period for {kepler_id} is NaN. Using random period: {period:.2f}")
                    else:
                        period = np.random.uniform(1, 100) # Fallback if no match
                        print(f"Warning: No period found for {kepler_id} in metadata. Using random period: {period:.2f}")
                else:
                    period = np.random.uniform(1, 100) # Fallback if filename parsing fails
                    print(f"Warning: Could not parse Kepler ID from {os.path.basename(file_path)}. Using random period: {period:.2f}")
            elif label_type == 'false_positive':
                period = np.random.uniform(1, 100) # Random period for false positives
            
            if period is None: # Safeguard
                period = np.random.uniform(1, 100)
                print(f"Warning: Period is None after all attempts. Using random period: {period:.2f}")

            # --- 1D Time-Series Preparation (from original light curve) ---
            # Normalize the entire light curve first
            flux_norm = (flux - np.median(flux)) / (np.std(flux) + 1e-8)
            
            # For simplicity, we'll take a center chunk of the light curve
            # A more advanced approach would use the transit detection logic
            FIXED_LENGTH = 2048 # Define a fixed length for time-series segments
            start = max(0, len(flux_norm) // 2 - FIXED_LENGTH // 2)
            segment = flux_norm[start : start + FIXED_LENGTH]

            # Pad if segment is shorter than FIXED_LENGTH
            if len(segment) < FIXED_LENGTH:
                segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
            
            timeseries_data.append(segment)
            
            # --- 2D Image Preparation (Phase-folded) ---
            images = create_image_representations(time, flux, period, t0)
            if images is None:
                continue

            image_data.append(images)
            labels.append(1 if label_type == 'confirmed_planet' else 0)
            successful_files.append(item)
                
        except Exception as e:
            print(f"Error processing {item} for multimodal data: {e}")
            continue
    
    if not image_data:
        return None, None, None, None
        
    return np.array(image_data), np.array(timeseries_data), np.array(labels), successful_files