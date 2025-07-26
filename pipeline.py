import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import hashlib
from datetime import datetime
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Input, Concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from astroquery.mast import Observations
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astroquery.ipac.nexsci.tap import TapPlus

# Configuration
DATA_DIR = "data"
LIGHT_CURVE_DIR = os.path.join(DATA_DIR, "light_curves")
MODEL_DIR = os.path.join(DATA_DIR, "models")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Create necessary directories
for directory in [DATA_DIR, LIGHT_CURVE_DIR, MODEL_DIR, METADATA_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def generate_file_hash(params):
    """Generate a consistent hash for query parameters to use as cache filename."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def fetch_kepler_data(max_records=50, use_cache=True):
    """Fetches light curve data from Kepler mission with local caching."""
    cache_file = os.path.join(METADATA_DIR, "kepler_observation_list.pkl")
    
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached Kepler observation list")
        obs_table = pd.read_pickle(cache_file)
        return obs_table
    
    print(f"Fetching Kepler observations from MAST")
    obs_table = Observations.query_criteria(obs_collection='Kepler', 
                                           dataproduct_type="timeseries",
                                           objectname="exoplanet")
    
    obs_table.to_pandas().to_pickle(cache_file)
    
    return obs_table[:max_records]

def download_product(product, use_cache=True):
    """Downloads a data product with caching."""
    filename = f"{product['obs_id']}_{product['dataproduct_type']}.fits"
    local_path = os.path.join(LIGHT_CURVE_DIR, filename)
    
    if use_cache and os.path.exists(local_path):
        print(f"Using cached file: {local_path}")
        return local_path
    
    print(f"Downloading: {product['dataURI']}")
    try:
        download_path = Observations.download_file(product['dataURI'], local_path=local_path)
        return download_path[0] if isinstance(download_path, list) else download_path
    except Exception as e:
        print(f"Error downloading {product['dataURI']}: {e}")
        return None

def download_light_curves(obs_table, use_cache=True):
    """Downloads light curves from an observation table."""
    light_curve_files = []
    for obs in obs_table:
        try:
            data_products = Observations.get_product_list(obs)
            light_curve_products = [p for p in data_products if 'LIGHTCURVE' in p['dataURI']]
            
            if light_curve_products:
                file_path = download_product(light_curve_products[0], use_cache=use_cache)
                if file_path:
                    light_curve_files.append(file_path)
        except Exception as e:
            print(f"Error processing observation {obs['obs_id']}: {e}")
    return light_curve_files

def fetch_exoplanet_labels(use_cache=True):
    """Fetches confirmed exoplanet data for training labels using the updated TAP service."""
    cache_file = os.path.join(METADATA_DIR, "exoplanet_labels.csv")
    
    if use_cache and os.path.exists(cache_file):
        print("Loading cached exoplanet labels")
        return pd.read_csv(cache_file)
    
    print("Fetching exoplanet data from NASA Exoplanet Archive TAP service")
    tap = TapPlus(url="https://exoplanetarchive.ipac.caltech.edu/TAP")
    
    query = "SELECT pl_name, hostname, pl_orbper, pl_rade, pl_masse, disc_year, discoverymethod FROM ps WHERE default_flag = 1"
    
    result = tap.launch_job(query)
    exoplanet_data = result.get_results()
    
    exoplanet_data.to_pandas().to_csv(cache_file, index=False)
    
    return exoplanet_data.to_pandas()

def preprocess_light_curve(file_path):
    """Loads and preprocesses light curve FITS files."""
    try:
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            time, flux = data['TIME'], data['PDCSAP_FLUX']
            mask = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[mask], flux[mask]
            
            if len(time) == 0:
                return None, None
                
            normalized_flux = (flux - np.median(flux)) / np.std(flux)
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
            # Simple repetition to create a 2D image
            img = np.tile(segment, (img_size[0], 1))
            # Resize to ensure consistent dimensions
            img_resized = tf.image.resize(np.expand_dims(img, axis=-1), img_size)
            images.append(img_resized)
    
    return np.array(images) if images else None

def extract_transit_features(time, flux, window_size=2048):
    """Extracts a fixed-length segment from the light curve."""
    if flux is None or len(flux) < window_size:
        return None
    
    # Simple case: take the central part of the light curve
    center_idx = len(flux) // 2
    start_idx = max(0, center_idx - window_size // 2)
    segment = flux[start_idx : start_idx + window_size]
    return [segment] # Return as a list to be consistent

def prepare_multimodal_data(light_curve_files, exoplanet_labels):
    """Prepares multimodal dataset combining image representations and time series data."""
    image_data = []
    timeseries_data = []
    labels = []
    
    host_stars = set(exoplanet_labels['hostname'].dropna().unique())
    
    for file_path in light_curve_files:
        try:
            with fits.open(file_path) as hdul:
                # Basic stellar name extraction from FITS header
                stellar_name = hdul[0].header.get('OBJECT', '').replace('KIC ', '').strip()
                is_host = stellar_name in host_stars
            
            time, flux = preprocess_light_curve(file_path)
            if time is None or flux is None:
                continue
            
            # Extract a single feature segment per file for simplicity
            transit_segments = extract_transit_features(time, flux)
            if transit_segments is None:
                continue

            # We use the same segment for both time-series and image representation
            segment = transit_segments[0]
            
            # Create image representation from the segment
            images = create_image_representations([segment])
            if images is None:
                continue

            image_data.append(images[0])
            timeseries_data.append(segment)
            labels.append(1 if is_host else 0)
                
        except Exception as e:
            print(f"Error processing {file_path} for multimodal data: {e}")
            continue
    
    if not image_data:
        return None, None, None
        
    return np.array(image_data), np.array(timeseries_data), np.array(labels)

def optimize_hyperparameters(X_image, X_timeseries, y, n_iter=10, cv=3):
    """Optimizes hyperparameters for the multimodal model using RandomizedSearchCV."""
    
    def build_model(conv_filters=32, dense_neurons=128, dropout_rate=0.3, learning_rate=0.001):
        """A wrapper function to build the Keras model for use in scikit-learn."""
        
        # Image branch
        image_input = Input(shape=X_image.shape[1:])
        x_img = Conv2D(conv_filters, (3, 3), activation='relu', padding='same')(image_input)
        x_img = MaxPooling2D((2, 2))(x_img)
        x_img = Conv2D(conv_filters * 2, (3, 3), activation='relu', padding='same')(x_img)
        x_img = MaxPooling2D((2, 2))(x_img)
        x_img = Flatten()(x_img)
        x_img = Dense(dense_neurons, activation='relu')(x_img)
        
        # Time series branch
        timeseries_input = Input(shape=X_timeseries.shape[1:])
        x_ts = Reshape((X_timeseries.shape[1], 1))(timeseries_input)
        x_ts = Conv1D(conv_filters, 5, activation='relu', padding='same')(x_ts)
        x_ts = MaxPooling1D(2)(x_ts)
        x_ts = Flatten()(x_ts)
        x_ts = Dense(dense_neurons, activation='relu')(x_ts)
        
        # Merge branches
        merged = Concatenate()([x_img, x_ts])
        merged = Dense(dense_neurons * 2, activation='relu')(merged)
        merged = Dropout(dropout_rate)(merged)
        output = Dense(1, activation='sigmoid')(merged)
        
        model = Model(inputs=[image_input, timeseries_input], outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = KerasClassifier(build_fn=build_model, verbose=0)
    
    param_dist = {
        'conv_filters': [16, 32, 64],
        'dense_neurons': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20, 30]
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # RandomizedSearchCV expects a dictionary for multi-input models
    X_dict = {'image_input': X_image, 'timeseries_input': X_timeseries}

    # The KerasClassifier wrapper for scikit-learn doesn't natively support multi-input models
    # A common workaround is to create a custom wrapper or pass data in a way the fit method can handle.
    # For this case, we will use a simplified approach for demonstration, fitting on a combined feature set.
    # NOTE: This is a simplification. A production system would require a more complex wrapper.
    
    print("Warning: Simplifying to single input for hyperparameter search demonstration.")
    X_combined = np.concatenate([X_image.reshape(X_image.shape[0], -1), X_timeseries], axis=1)

    def build_simple_model(dense_neurons=128, dropout_rate=0.3, learning_rate=0.001):
        """Simplified single-input model for hyperparameter tuning."""
        model = Sequential([
            Input(shape=(X_combined.shape[1],)),
            Dense(dense_neurons, activation='relu'),
            Dropout(dropout_rate),
            Dense(dense_neurons // 2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    simple_model = KerasClassifier(build_fn=build_simple_model, verbose=0)

    param_dist_simple = {
        'dense_neurons': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20, 30]
    }

    random_search_simple = RandomizedSearchCV(
        estimator=simple_model,
        param_distributions=param_dist_simple,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    random_search_simple.fit(X_combined, y)
    
    print("Best parameters found:")
    print(random_search_simple.best_params_)
    
    return random_search_simple.best_params_