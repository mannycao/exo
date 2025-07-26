# legacy_pipeline.py

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

def create_image_representations(segments, img_size=(64, 64)):
    """Convert 1D segments to 2D image representations."""
    if segments is None or len(segments) == 0:
        return None
    
    images = []
    for segment in segments:
        if len(segment) > 0:
            segment_2d = segment.reshape(1, -1)
            img_resized = tf.image.resize(tf.expand_dims(segment_2d, axis=-1), img_size)
            images.append(img_resized)
    
    return np.array(images).squeeze() if images else None


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

def prepare_multimodal_data(light_curve_files, exoplanet_labels):
    """Prepares multimodal dataset from a list of file dictionaries."""
    image_data = []
    timeseries_data = []
    labels = []
    
    for item in light_curve_files:
        file_path = item['file_path']
        label_type = item['type']
        try:
            time, flux = preprocess_light_curve(file_path)
            if time is None or flux is None:
                continue
            
            transit_segments = extract_transit_features(time, flux)
            if transit_segments is None:
                continue

            segment = transit_segments[0]
            images = create_image_representations([segment])
            if images is None:
                continue

            image_data.append(images)
            timeseries_data.append(segment)
            labels.append(1 if label_type == 'confirmed_planet' else 0)
                
        except Exception as e:
            print(f"Error processing {item} for multimodal data: {e}")
            continue
    
    if not image_data:
        return None, None, None
        
    return np.array(image_data), np.array(timeseries_data), np.array(labels)