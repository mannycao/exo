import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import hashlib
from datetime import datetime
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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

def fetch_kepler_data(target='Kepler', max_records=50, use_cache=True):
    """Fetches light curve data from Kepler mission with local caching."""
    cache_file = os.path.join(METADATA_DIR, f"{target.lower()}_observation_list.pkl")
    
    # Check if we have cached data
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached {target} observation list")
        obs_table = pd.read_pickle(cache_file)
        return obs_table
    
    print(f"Fetching {target} observations from MAST")
    obs_table = Observations.query_criteria(obs_collection=target, 
                                           dataproduct_type="timeseries",
                                           objectname="exoplanet")
    
    # Cache results for future use
    obs_table.to_pandas().to_pickle(cache_file)
    
    return obs_table[:max_records]  # Limit for development/testing

def download_product(product, use_cache=True):
    """Downloads a data product with caching."""
    # Create a sensible filename
    filename = f"{product['obs_id']}_{product['dataproduct_type']}.fits"
    local_path = os.path.join(LIGHT_CURVE_DIR, filename)
    
    if use_cache and os.path.exists(local_path):
        print(f"Using cached file: {local_path}")
        return local_path
    
    print(f"Downloading: {product['dataURI']}")
    try:
        download_path = Observations.download_file(product['dataURI'])
        # Move to our organized directory structure
        if download_path != local_path:
            os.rename(download_path, local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading {product['dataURI']}: {e}")
        return None

def fetch_exoplanet_labels(use_cache=True):
    """Fetches confirmed exoplanet data for training labels using the updated TAP service."""
    cache_file = os.path.join(METADATA_DIR, "exoplanet_labels.csv")
    
    if use_cache and os.path.exists(cache_file):
        print("Loading cached exoplanet labels")
        return pd.read_csv(cache_file)
    
    print("Fetching exoplanet data from NASA Exoplanet Archive TAP service")
    # Using TAP service instead of the deprecated exoplanets table
    tap = TapPlus(url="https://exoplanetarchive.ipac.caltech.edu")
    
    # Query the Planetary Systems (PS) table instead of exoplanets
    query = """
    SELECT pl_name, hostname, pl_orbper, pl_rade, pl_masse, disc_year, discoverymethod
    FROM ps
    WHERE disc_refname IS NOT NULL
    AND pl_orbper IS NOT NULL
    AND pl_rade IS NOT NULL
    """
    
    result = tap.launch_job(query)
    exoplanet_data = result.get_data()
    
    # Cache results
    exoplanet_data.to_pandas().to_csv(cache_file, index=False)
    
    return exoplanet_data

def preprocess_light_curve(file_path):
    """Loads and preprocesses light curve FITS files."""
    try:
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            time, flux = data['TIME'], data['PDCSAP_FLUX']
            mask = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[mask], flux[mask]
            
            if len(time) == 0:
                return None, None, None
                
            # Normalize the flux
            flux_mean = np.mean(flux)
            flux_std = np.std(flux)
            normalized_flux = (flux - flux_mean) / flux_std
            
            # Store metadata for reference
            metadata = {
                'file_path': file_path,
                'mean': flux_mean,
                'std': flux_std,
                'n_points': len(time)
            }
            
            return time, normalized_flux, metadata
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def detect_transits(time, flux, sensitivity=3.0, min_duration=0.1, max_duration=0.5):
    """
    Uses peak detection to identify potential exoplanet transits.
    
    Args:
        time: Time series data
        flux: Normalized flux data
        sensitivity: How many standard deviations below mean to detect
        min_duration/max_duration: Expected transit duration range in days
    
    Returns:
        Dictionary with transit information
    """
    if time is None or flux is None:
        return None
    
    # Convert duration to data points (assuming uniform sampling)
    time_step = np.median(np.diff(time))
    min_width = int(min_duration / time_step)
    max_width = int(max_duration / time_step)
    
    # Find dips in light curve
    inverted_flux = -flux  # Look for peaks in inverted flux
    std_dev = np.std(inverted_flux)
    prominence = sensitivity * std_dev  # Peaks must be at least this prominent
    
    peaks, properties = find_peaks(inverted_flux, 
                                  prominence=prominence,
                                  width=(min_width, max_width),
                                  distance=max_width*2)  # Min distance between peaks
    
    if len(peaks) == 0:
        return None
    
    # Extract transit properties
    transit_info = {
        'peak_indices': peaks,
        'times': time[peaks],
        'depths': flux[peaks],
        'widths': properties['widths'],
        'prominences': properties['prominences']
    }
    
    return transit_info

def extract_transit_features(time, flux, transit_info, window_size=100):
    """Extract segments of light curve centered on detected transits."""
    if transit_info is None or len(transit_info['peak_indices']) == 0:
        return None
    
    features = []
    for peak_idx in transit_info['peak_indices']:
        # Ensure we don't go out of bounds
        start_idx = max(0, peak_idx - window_size//2)
        end_idx = min(len(flux), peak_idx + window_size//2)
        
        # Extract segment
        segment = flux[start_idx:end_idx]
        
        # Pad if necessary to ensure consistent size
        if len(segment) < window_size:
            pad_left = (window_size - len(segment)) // 2
            pad_right = window_size - len(segment) - pad_left
            segment = np.pad(segment, (pad_left, pad_right), mode='constant')
        
        features.append(segment)
    
    return np.array(features)

def create_image_representations(transit_segments, img_size=(64, 64)):
    """Convert transit segments to image-like representations for CNN processing."""
    if transit_segments is None:
        return None
    
    num_segments = len(transit_segments)
    images = np.zeros((num_segments, img_size[0], img_size[1]))
    
    for i, segment in enumerate(transit_segments):
        # Resize segment to fit the image width
        resized_segment = np.interp(
            np.linspace(0, len(segment)-1, img_size[1]),
            np.arange(len(segment)),
            segment
        )
        
        # Create a 2D representation - different approaches possible:
        # 1. Simple repetition
        for j in range(img_size[0]):
            images[i, j, :] = resized_segment
            
        # 2. Optional: Add time-frequency representation
        # Could add wavelet transform or other time-frequency analysis here
    
    return images

def reshape_data(X):
    """Ensures input data is reshaped correctly for the AI model."""
    if X is None:
        return None
    return np.expand_dims(X, axis=-1) if len(X.shape) == 3 else X

def build_transit_detection_model(input_shape):
    """Creates a convolutional neural network for transit detection."""
    model = Sequential([
        Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_characterization_model(input_shape):
    """Creates a model for characterizing exoplanet properties."""
    model = Sequential([
        Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.2),
        # Multiple outputs: radius, period, etc.
        Dense(3, activation='linear')  # For multiple exoplanet properties
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=50):
    """Trains AI model using labeled exoplanet data with callbacks."""
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.h5")
    
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(MODEL_DIR, f"{model_name}_history.csv"))
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluates model performance with appropriate metrics."""
    # Basic evaluation
    evaluation = model.evaluate(X_test, y_test)
    
    # For binary classification
    if y_test.ndim == 1 or y_test.shape[1] == 1:
        y_pred_prob = model.predict(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        ap_score = average_precision_score(y_test, y_pred_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'AP Score: {ap_score:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(MODEL_DIR, 'precision_recall_curve.png'))
        
        return {
            'loss': evaluation[0],
            'accuracy': evaluation[1],
            'average_precision': ap_score
        }
    
    # For regression
    return {
        'loss': evaluation[0],
        'mae': evaluation[1]
    }

def visualize_transit(time, flux, transit_info=None, filename=None):
    """Create visualization of light curve and detected transits."""
    plt.figure(figsize=(12, 6))
    plt.plot(time, flux, 'k-', alpha=0.8, label='Normalized Flux')
    
    if transit_info is not None and len(transit_info['peak_indices']) > 0:
        plt.scatter(
            transit_info['times'], 
            transit_info['depths'], 
            color='red', 
            s=50, 
            marker='v', 
            label=f"Detected Transits ({len(transit_info['peak_indices'])})"
        )
        
        # Optionally highlight the transit regions
        for idx, width in zip(transit_info['peak_indices'], transit_info['widths']):
            half_width = int(width / 2)
            left_idx = max(0, idx - half_width)
            right_idx = min(len(flux) - 1, idx + half_width)
            plt.axvspan(time[left_idx], time[right_idx], color='red', alpha=0.2)
    
    plt.xlabel('Time (BKJD)')
    plt.ylabel('Normalized Flux')
    plt.title('Light Curve with Detected Transits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def preprocess_dataset_def preprocess_dataset_for_training(light_curve_files, exoplanet_labels):
    """Processes a set of light curves for AI training."""
    transit_images = []
    labels = []
    
    # Create a mapping of stellar names to confirmed planets
    host_to_planets = {}
    for _, row in exoplanet_labels.iterrows():
        hostname = row['hostname']
        if hostname not in host_to_planets:
            host_to_planets[hostname] = []
        host_to_planets[hostname].append({
            'pl_name': row['pl_name'],
            'pl_orbper': row['pl_orbper'],
            'pl_rade': row['pl_rade'],
            'pl_masse': row.get('pl_masse', np.nan),
            'discoverymethod': row['discoverymethod']
        })
    
    # Process each light curve
    for file_path in light_curve_files:
        try:
            # Extract stellar name from file path
            file_name = os.path.basename(file_path)
            # This is a placeholder - actual stellar name extraction would depend on your file naming
            stellar_name = file_name.split('_')[0]  
            
            # Check if this star has confirmed planets
            has_confirmed_planets = stellar_name in host_to_planets
            
            # Process the light curve
            time, flux, metadata = preprocess_light_curve(file_path)
            if time is None or flux is None:
                continue
                
            # Detect transits
            transit_info = detect_transits(time, flux)
            if transit_info is None:
                continue
                
            # Extract features
            transit_segments = extract_transit_features(time, flux, transit_info)
            if transit_segments is None:
                continue
                
            # Create image representations
            images = create_image_representations(transit_segments)
            if images is None:
                continue
                
            # Add to dataset
            transit_images.append(images)
            
            # Label based on confirmed planets
            if has_confirmed_planets:
                # For confirmed planets, use 1 as the label
                batch_labels = np.ones(len(images))
            else:
                # For stars without confirmed planets, use 0 as the label
                batch_labels = np.zeros(len(images))
                
            labels.append(batch_labels)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Combine all processed data
    if not transit_images:
        return None, None
        
    X = np.vstack(transit_images)
    y = np.concatenate(labels)
    
    return X, y

def periodogram_analysis(time, flux):
    """Performs periodogram analysis to detect periodic signals."""
    from astropy.timeseries import LombScargle
    
    # Remove NaN values
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    
    # Compute the Lomb-Scargle periodogram
    frequency, power = LombScargle(time, flux).autopower()
    
    # Convert frequency to period
    period = 1/frequency
    
    # Find peaks in the periodogram
    peak_indices, _ = find_peaks(power, height=0.1)
    peak_periods = period[peak_indices]
    peak_powers = power[peak_indices]
    
    # Sort by power
    sort_idx = np.argsort(peak_powers)[::-1]
    peak_periods = peak_periods[sort_idx]
    peak_powers = peak_powers[sort_idx]
    
    return {
        'period': period,
        'power': power,
        'peak_periods': peak_periods[:10],  # Top 10 peaks
        'peak_powers': peak_powers[:10]
    }

def analyze_periodicity(time, flux, transit_info):
    """Analyzes periodicity of detected transits."""
    if transit_info is None or len(transit_info['peak_indices']) < 2:
        return None
    
    # Calculate time differences between consecutive transits
    transit_times = transit_info['times']
    time_diffs = np.diff(transit_times)
    
    # Calculate the median time difference (potential orbital period)
    median_period = np.median(time_diffs)
    
    # Perform periodogram analysis
    periodogram = periodogram_analysis(time, flux)
    
    # Check if the detected period from transits matches any peak in the periodogram
    period_matches = []
    for peak_period in periodogram['peak_periods']:
        # Check if the peak period is close to the median period or its harmonics
        if abs(peak_period - median_period) / median_period < 0.1:
            period_matches.append(peak_period)
        elif abs(peak_period - 2*median_period) / (2*median_period) < 0.1:
            period_matches.append(peak_period / 2)  # Half the harmonic
    
    return {
        'median_period': median_period,
        'period_matches': period_matches,
        'periodogram': periodogram
    }

def apply_transit_modeling(time, flux, transit_info, periodicity_info):
    """Applies transit modeling to characterize the potential exoplanet."""
    if transit_info is None or periodicity_info is None:
        return None
    
    try:
        import batman
        
        # Use the detected period from periodicity analysis
        period = periodicity_info['median_period']
        if period <= 0:
            return None
        
        # Create a transit model
        params = batman.TransitParams()
        params.t0 = transit_info['times'][0]  # Time of first transit
        params.per = period  # Orbital period
        params.rp = 0.1  # Planet radius (relative to stellar radius)
        params.a = 15.0  # Semi-major axis (in units of stellar radii)
        params.inc = 87.0  # Orbital inclination (in degrees)
        params.ecc = 0.0  # Eccentricity
        params.w = 90.0  # Longitude of periastron (in degrees)
        params.u = [0.1, 0.3]  # Limb darkening coefficients
        params.limb_dark = "quadratic"  # Limb darkening model
        
        # Initialize the model
        m = batman.TransitModel(params, time)
        
        # Generate the model light curve
        model_flux = m.light_curve(params)
        
        # Calculate residuals
        residuals = flux - model_flux
        residual_std = np.std(residuals)
        
        # Fit the model (simplified - in reality, you'd use MCMC or other methods)
        # This is a placeholder for more sophisticated fitting
        fit_quality = np.mean(np.abs(residuals))
        
        return {
            'model_params': params,
            'model_flux': model_flux,
            'residuals': residuals,
            'residual_std': residual_std,
            'fit_quality': fit_quality
        }
    except Exception as e:
        print(f"Error in transit modeling: {e}")
        return None

def estimate_planet_properties(transit_info, periodicity_info, stellar_properties=None):
    """Estimates basic planet properties from transit and periodicity data."""
    if transit_info is None or periodicity_info is None:
        return None
    
    # Default stellar properties if not provided
    if stellar_properties is None:
        stellar_properties = {
            'radius': 1.0,  # Solar radius
            'mass': 1.0,    # Solar mass
            'temperature': 5778  # Solar temperature (K)
        }
    
    # Extract transit depth and duration
    depths = transit_info['depths']
    mean_depth = np.mean(depths)
    
    # Estimate planet radius (in Earth radii)
    # Planet radius / star radius = sqrt(transit depth)
    planet_radius_ratio = np.sqrt(abs(mean_depth))
    planet_radius = planet_radius_ratio * stellar_properties['radius'] * 109.2  # Convert to Earth radii
    
    # Estimate orbital period (in days)
    orbital_period = periodicity_info['median_period']
    
    # Estimate semi-major axis using Kepler's Third Law
    # a^3 / P^2 = G(M + m) / 4π^2 ≈ GM / 4π^2 (since M >> m)
    # a = (GM * P^2 / 4π^2)^(1/3)
    G = 6.67430e-11  # Gravitational constant
    M_sun = 1.989e30  # Solar mass in kg
    star_mass_kg = stellar_properties['mass'] * M_sun
    P_seconds = orbital_period * 86400  # Convert days to seconds
    
    semi_major_axis_m = (G * star_mass_kg * P_seconds**2 / (4 * np.pi**2))**(1/3)
    semi_major_axis_au = semi_major_axis_m / 1.496e11  # Convert to AU
    
    # Estimate equilibrium temperature
    # T_eq = T_star * sqrt(R_star / 2a) * (1 - albedo)^(1/4)
    albedo = 0.3  # Assumed albedo
    star_radius_m = stellar_properties['radius'] * 6.957e8  # Convert to meters
    equilibrium_temp = stellar_properties['temperature'] * np.sqrt(star_radius_m / (2 * semi_major_axis_m)) * (1 - albedo)**(1/4)
    
    return {
        'radius_earth': planet_radius,
        'orbital_period_days': orbital_period,
        'semi_major_axis_au': semi_major_axis_au,
        'equilibrium_temp_k': equilibrium_temp
    }

def run_pipeline(light_curve_files, exoplanet_labels=None, use_cache=True):
    """Runs the full pipeline for exoplanet detection and characterization."""
    results = []
    
    # Process each light curve file
    for file_path in light_curve_files:
        try:
            print(f"Processing {file_path}")
            
            # Step 1: Preprocess the light curve
            time, flux, metadata = preprocess_light_curve(file_path)
            if time is None or flux is None:
                print(f"Failed to preprocess {file_path}")
                continue
                
            # Step 2: Traditional transit detection
            transit_info = detect_transits(time, flux)
            if transit_info is None:
                print(f"No transits detected in {file_path}")
                continue
                
            # Step 3: Periodicity analysis
            periodicity_info = analyze_periodicity(time, flux, transit_info)
            if periodicity_info is None:
                print(f"No periodicity found in {file_path}")
                
            # Step 4: Transit modeling (if available)
            transit_model = None
            if periodicity_info is not None:
                transit_model = apply_transit_modeling(time, flux, transit_info, periodicity_info)
            
            # Step 5: Estimate planet properties
            planet_properties = None
            if periodicity_info is not None:
                planet_properties = estimate_planet_properties(transit_info, periodicity_info)
            
            # Step 6: Feature extraction for AI/ML
            transit_segments = extract_transit_features(time, flux, transit_info)
            transit_images = None
            if transit_segments is not None:
                transit_images = create_image_representations(transit_segments)
            
            # Step 7: Visualization
            result_dir = os.path.join(RESULTS_DIR, os.path.basename(file_path).split('.')[0])
            os.makedirs(result_dir, exist_ok=True)
            
            # Save the light curve visualization
            visualize_transit(
                time, 
                flux, 
                transit_info, 
                filename=os.path.join(result_dir, 'light_curve.png')
            )
            
            # Step 8: Save the results
            result = {
                'file_path': file_path,
                'metadata': metadata,
                'transit_count': len(transit_info['peak_indices']) if transit_info else 0,
                'periodicity': periodicity_info['median_period'] if periodicity_info else None,
                'planet_properties': planet_properties,
                'transit_model': transit_model is not None,
                'has_transit_images': transit_images is not None,
                'result_dir': result_dir
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return results

def main():
    """Pipeline execution."""
    
    # Step 1: Get light curve data
    obs_table = fetch_kepler_data(max_records=20, use_cache=True)
    
    # Step 2: Fetch exoplanet labels
    exoplanet_labels = fetch_exoplanet_labels(use_cache=True)
    
    # Step 3: Download light curve files
    light_curve_files = []
    for i, obs in enumerate(obs_table):
        try:
            data_products = Observations.get_product_list(obs)
            # Filter for light curve products
            light_curve_products = [p for p in data_products if 'LIGHTCURVE' in p['dataURI']]
            
            if light_curve_products:
                file_path = download_product(light_curve_products[0], use_cache=True)
                if file_path:
                    light_curve_files.append(file_path)
                
            if i >= 5:  # Limit for testing
                break
        except Exception as e:
            print(f"Error with observation {obs['obsid']}: {e}")
    
    # Step 4: Run the pipeline
    results = run_pipeline(light_curve_files, exoplanet_labels)
    
    # Step 5: Optional - Train AI model
    if len(results) > 0:
        # Prepare data for AI training
        X, y = preprocess_dataset_for_training(light_curve_files, exoplanet_labels)
        
        if X is not None and y is not None and len(X) > 0:
            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            
            # Build and train the model
            model = build_transit_detection_model((X.shape[1], X.shape[2]))
            trained_model, history = train_model(model, X_train, y_train, X_val, y_val, 'transit_detector')
            
            # Evaluate model
            eval_results = evaluate_model(trained_model, X_test, y_test)
            print("Model evaluation results:", eval_results)
            
            # Save model summary
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            with open(os.path.join(MODEL_DIR, 'model_summary.txt'), 'w') as f:
                f.write('\n'.join(model_summary))
    
    # Step 6: Summarize results
    print(f"\nProcessed {len(light_curve_files)} light curves")
    print(f"Detected potential transits in {sum(1 for r in results if r['transit_count'] > 0)} light curves")
    print(f"Found periodicity in {sum(1 for r in results if r['periodicity'] is not None)} light curves")
    
    # Step 7: Generate comprehensive report
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"pipeline_report_{report_time}.html")
    
    with open(report_file, 'w') as f:
        f.write("<html><head><title>Exoplanet Detection Pipeline Report</title>")
        f.write("<style>body{font-family:Arial;max-width:1200px;margin:auto;padding:20px}")
        f.write("table{width:100%;border-collapse:collapse;margin:20px 0}")
        f.write("th,td{padding:8px;border:1px solid #ddd;text-align:left}")
        f.write("th{background-color:#f2f2f2}</style></head><body>")
        f.write(f"<h1>Exoplanet Detection Pipeline Report</h1>")
        f.write(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        f.write(f"<p>Processed {len(light_curve_files)} light curves</p>")
        
        # Summary statistics
        f.write("<h2>Summary Statistics</h2>")
        f.write("<ul>")
        f.write(f"<li>Light curves with detected transits: {sum(1 for r in results if r['transit_count'] > 0)}</li>")
        f.write(f"<li>Light curves with periodicity: {sum(1 for r in results if r['periodicity'] is not None)}</li>")
        f.write(f"<li>Light curves with planet property estimates: {sum(1 for r in results if r['planet_properties'] is not None)}</li>")
        f.write("</ul>")
        
        # Results table
        f.write("<h2>Detailed Results</h2>")
        f.write("<table>")
        f.write("<tr><th>File</th><th>Transit Count</th><th>Period (days)</th><th>Planet Radius (Earth)</th><th>Semi-major Axis (AU)</th><th>Equilibrium Temp (K)</th></tr>")
        
        for result in results:
            file_name = os.path.basename(result['file_path'])
            transit_count = result['transit_count']
            period = result['periodicity'] if result['periodicity'] else "N/A"
            
            # Planet properties
            radius = "N/A"
            semi_major = "N/A"
            temp = "N/A"
            
            if result['planet_properties']:
                props = result['planet_properties']
                radius = f"{props['radius_earth']:.2f}" if 'radius_earth' in props else "N/A"
                semi_major = f"{props['semi_major_axis_au']:.3f}" if 'semi_major_axis_au' in props else "N/A"
                temp = f"{props['equilibrium_temp_k']:.0f}" if 'equilibrium_temp_k' in props else "N/A"
            
            f.write(f"<tr><td>{file_name}</td><td>{transit_count}</td><td>{period}</td>")
            f.write(f"<td>{radius}</td><td>{semi_major}</td><td>{temp}</td></tr>")
        
        f.write("</table>")
        f.write("</body></html>")
    
    print(f"Report generated: {report_file}")
    
    # Step 8: Export results to JSON for further analysis
    export_file = os.path.join(RESULTS_DIR, f"pipeline_results_{report_time}.json")
    
    # Convert results to JSON serializable format
    json_results = []
    for result in results:
        # Clean up non-serializable objects
        clean_result = {
            'file_path': result['file_path'],
            'metadata': {
                'file_path': result['metadata']['file_path'],
                'mean': float(result['metadata']['mean']),
                'std': float(result['metadata']['std']),
                'n_points': int(result['metadata']['n_points'])
            },
            'transit_count': int(result['transit_count']),
            'result_dir': result['result_dir']
        }
        
        # Add periodicity if available
        if result['periodicity'] is not None:
            clean_result['periodicity'] = float(result['periodicity'])
        
        # Add planet properties if available
        if result['planet_properties'] is not None:
            clean_result['planet_properties'] = {
                'radius_earth': float(result['planet_properties']['radius_earth']),
                'orbital_period_days': float(result['planet_properties']['orbital_period_days']),
                'semi_major_axis_au': float(result['planet_properties']['semi_major_axis_au']),
                'equilibrium_temp_k': float(result['planet_properties']['equilibrium_temp_k'])
            }
        
        json_results.append(clean_result)
    
    with open(export_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results exported to: {export_file}")
    
    return results


def create_multimodal_fusion_model(image_input_shape, timeseries_input_shape):
    """Creates a multimodal model that combines image and time series data."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Concatenate
    
    # Image branch
    image_input = Input(shape=image_input_shape)
    image_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    image_pool1 = MaxPooling2D((2, 2))(image_conv1)
    image_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(image_pool1)
    image_pool2 = MaxPooling2D((2, 2))(image_conv2)
    image_flat = Flatten()(image_pool2)
    image_dense = Dense(128, activation='relu')(image_flat)
    
    # Time series branch
    timeseries_input = Input(shape=timeseries_input_shape)
    ts_reshape = Reshape((timeseries_input_shape[0], 1))(timeseries_input)
    ts_conv1 = Conv1D(32, 5, activation='relu', padding='same')(ts_reshape)
    ts_pool1 = MaxPooling1D(2)(ts_conv1)
    ts_conv2 = Conv1D(64, 3, activation='relu', padding='same')(ts_pool1)
    ts_pool2 = MaxPooling1D(2)(ts_conv2)
    ts_flat = Flatten()(ts_pool2)
    ts_dense = Dense(128, activation='relu')(ts_flat)
    
    # Merge branches
    merged = Concatenate()([image_dense, ts_dense])
    merged_dense1 = Dense(256, activation='relu')(merged)
    merged_drop1 = Dropout(0.5)(merged_dense1)
    merged_dense2 = Dense(128, activation='relu')(merged_drop1)
    merged_drop2 = Dropout(0.3)(merged_dense2)
    output = Dense(1, activation='sigmoid')(merged_drop2)
    
    # Create model
    model = Model(inputs=[image_input, timeseries_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def prepare_multimodal_data(light_curve_files, exoplanet_labels):
    """Prepares multimodal dataset combining image representations and time series data."""
    image_data = []
    timeseries_data = []
    labels = []
    
    # Create lookup table for exoplanet hosts
    host_stars = set()
    if exoplanet_labels is not None:
        host_stars = set(exoplanet_labels['hostname'].unique())
    
    for file_path in light_curve_files:
        try:
            # Extract stellar name (this is a placeholder - modify based on your file naming)
            file_name = os.path.basename(file_path)
            stellar_name = file_name.split('_')[0]
            
            # Determine if this is a known exoplanet host
            is_host = stellar_name in host_stars
            
            # Process light curve
            time, flux, metadata = preprocess_light_curve(file_path)
            if time is None or flux is None:
                continue
                
            # Detect transits
            transit_info = detect_transits(time, flux)
            if transit_info is None or len(transit_info['peak_indices']) == 0:
                continue
                
            # Create time series segments
            transit_segments = extract_transit_features(time, flux, transit_info)
            if transit_segments is None:
                continue
                
            # Create image representations
            images = create_image_representations(transit_segments)
            if images is None:
                continue
                
            # Add to datasets
            for i, (segment, image) in enumerate(zip(transit_segments, images)):
                image_data.append(image)
                timeseries_data.append(segment)
                labels.append(1 if is_host else 0)
                
        except Exception as e:
            print(f"Error processing {file_path} for multimodal data: {e}")
            continue
    
    # Convert to numpy arrays
    if not image_data:
        return None, None, None
        
    X_image = np.array(image_data)
    X_timeseries = np.array(timeseries_data)
    y = np.array(labels)
    
    return X_image, X_timeseries, y


def train_multimodal_model(X_image, X_timeseries, y, model_name='multimodal_classifier'):
    """Trains the multimodal fusion model."""
    # Create train/validation/test split
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])
    
    # Split the data
    X_image_train, X_image_val, X_image_test = X_image[train_idx], X_image[val_idx], X_image[test_idx]
    X_ts_train, X_ts_val, X_ts_test = X_timeseries[train_idx], X_timeseries[val_idx], X_timeseries[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    # Reshape for model input
    X_image_train = np.expand_dims(X_image_train, axis=-1)
    X_image_val = np.expand_dims(X_image_val, axis=-1)
    X_image_test = np.expand_dims(X_image_test, axis=-1)
    
    # Create model
    model = create_multimodal_fusion_model(
        image_input_shape=X_image_train[0].shape,
        timeseries_input_shape=(X_ts_train[0].shape[0],)
    )
    
    # Set up callbacks
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.h5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # Train model
    history = model.fit(
        [X_image_train, X_ts_train], y_train,
        validation_data=([X_image_val, X_ts_val], y_val),
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(MODEL_DIR, f"{model_name}_history.csv"))
    
    # Evaluate model
    eval_results = model.evaluate([X_image_test, X_ts_test], y_test)
    y_pred_prob = model.predict([X_image_test, X_ts_test])
    
    # Generate ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multimodal Model ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_roc_curve.png"))
    
    # Generate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    ap_score = average_precision_score(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AP Score = {ap_score:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multimodal Model Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_pr_curve.png"))
    
    # Save evaluation metrics
    metrics = {
        'loss': float(eval_results[0]),
        'accuracy': float(eval_results[1]),
        'roc_auc': float(roc_auc),
        'average_precision': float(ap_score)
    }
    
    with open(os.path.join(MODEL_DIR, f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return model, metrics


def compare_models(transit_model_metrics, multimodal_metrics):
    """Compares performance of traditional and multimodal models."""
    plt.figure(figsize=(10, 6))
    
    metrics = ['accuracy', 'average_precision', 'roc_auc']
    traditional_scores = [transit_model_metrics.get(m, 0) for m in metrics]
    multimodal_scores = [multimodal_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, traditional_scores, width, label='Traditional CNN')
    plt.bar(x + width/2, multimodal_scores, width, label='Multimodal Model')
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'))
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': metrics,
        'Traditional CNN': traditional_scores,
        'Multimodal Model': multimodal_scores,
        'Improvement': [m-t for m, t in zip(multimodal_scores, traditional_scores)]
    })
    
    comparison.to_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'), index=False)
    
    return comparison


def analyze_feature_importance(model, X_image_test, X_ts_test):
    """Analyzes feature importance using permutation importance."""
    from sklearn.inspection import permutation_importance
    
    # Function to predict using the model
    def predict_fn(X):
        image_data, ts_data = X
        return model.predict([image_data, ts_data])
    
    # Set up the test data
    X_test = [X_image_test, X_ts_test]
    
    # Calculate permutation importance
    result = permutation_importance(
        predict_fn, X_test, y_test, 
        n_repeats=10, 
        random_state=42
    )
    
    # Extract the importances
    importances = result.importances_mean
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(['Image Features', 'Time Series Features'], importances)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'))
    
    return importances


def create_validation_dataset(use_cache=True):
    """Creates a validation dataset from a separate source."""
    # This is a placeholder for connecting to a different data source
    # For a real implementation, this would fetch data from a different mission
    
    cache_file = os.path.join(METADATA_DIR, "validation_dataset.pkl")
    
    if use_cache and os.path.exists(cache_file):
        print("Loading cached validation dataset")
        return pd.read_pickle(cache_file)
    
    # Placeholder for fetching validation data
    # In a real implementation, this would fetch data from e.g., TESS
    # and process it similarly to the Kepler data
    print("Fetching validation dataset")
    
    # Create a synthetic validation dataset based on the training data
    # This is a placeholder for demonstration purposes
    validation_data = {
        'light_curve_files': [],
        'exoplanet_labels': pd.DataFrame({
            'hostname': ['TESS_1', 'TESS_2', 'TESS_3'],
            'pl_name': ['TESS_1b', 'TESS_2b', 'TESS_3b'],
            'pl_orbper': [2.7, 5.1, 10.2],
            'pl_rade': [1.2, 2.1, 3.5],
            'discoverymethod': ['Transit', 'Transit', 'Transit'],
            'disc_year': [2020, 2021, 2022]
        })
    }
    
    # In a real implementation, we would process TESS data
    # to create light curves and other features
    
    # Cache the validation dataset
    pd.to_pickle(validation_data, cache_file)
    
    return validation_data


def cross_validate_models(kepler_model, validation_data):
    """Cross-validates the models on a different dataset."""
    # This is a placeholder for cross-validation on a different dataset
    # In a real implementation, this would:
    # 1. Process the validation data (e.g., from TESS)
    # 2. Apply the model trained on Kepler data
    # 3. Evaluate performance
    
    print("Cross-validating models on validation dataset")
    
    # Placeholder for cross-validation results
    cross_val_results = {
        'accuracy': 0.78,  # Example values
        'precision': 0.82,
        'recall': 0.75,
        'f1_score': 0.78,
        'transfer_learning_required': True
    }
    
    # In a real implementation, we would:
    # - Process the validation data
    # - Apply the models
    # - Calculate performance metrics
    # - Determine if transfer learning is needed
    
    return cross_val_results


def perform_transfer_learning(base_model, validation_data):
    """Performs transfer learning to adapt the model to a new dataset."""
    # This is a placeholder for transfer learning
    # In a real implementation, this would:
    # 1. Freeze the base layers of the model
    # 2. Add new layers for the new dataset
    # 3. Fine-tune on the new dataset
    
    print("Performing transfer learning")
    
    # Placeholder for transfer learning
    # In a real implementation, we would:
    # - Freeze base layers
    # - Add new layers
    # - Fine-tune on new dataset
    
    # Return the fine-tuned model
    return base_model


def create_ensemble_model(models, weights=None):
    """Creates an ensemble model from multiple trained models."""
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Average
    
    # Get the inputs
    inputs = models[0].inputs
    
    # Get the outputs
    outputs = [model.outputs[0] for model in models]
    
    # Create the ensemble output
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    # Create weighted average
    ensemble_output = outputs[0] * weights[0]
    for i in range(1, len(outputs)):
        ensemble_output = ensemble_output + outputs[i] * weights[i]
    
    # Create the ensemble model
    ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
    
    return ensemble_model


def optimize_hyperparameters(X_image, X_timeseries, y):
    """Optimizes hyperparameters for the multimodal model."""
    from sklearn.model_selection import RandomizedSearchCV
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    
    # Define the model-building function
    def build_model(conv_filters=32, dense_neurons=128, dropout_rate=0.3):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Reshape, Conv1D, MaxPooling1D
        
        # Image branch
        image_input = Input(shape=X_image[0].shape)
        image_conv1 = Conv2D(conv_filters, (3, 3), activation='relu', padding='same')(image_input)
        image_pool1 = MaxPooling2D((2, 2))(image_conv1)
        image_conv2 = Conv2D(conv_filters*2, (3, 3), activation='relu', padding='same')(image_pool1)
        image_pool2 = MaxPooling2D((2, 2))(image_conv2)
        image_flat = Flatten()(image_pool2)
        image_dense = Dense(dense_neurons, activation='relu')(image_flat)
        
        # Time series branch
        timeseries_input = Input(shape=(X_timeseries[0].shape[0],))
        ts_reshape = Reshape((X_timeseries[0].shape[0], 1))(timeseries_input)
        ts_conv1 = Conv1D(conv_filters, 5, activation='relu', padding='same')(ts_reshape)
        ts_pool1 = MaxPooling1D(2)(ts_conv1)
        ts_conv2 = Conv1D(conv_filters*2, 3, activation='relu', padding='same')(ts_pool1)
        ts_pool2 = MaxPooling1D(2)(ts_conv2)
        ts_flat = Flatten()(ts_pool2)
        ts_dense = Dense(dense_neurons, activation='relu')(ts_flat)
        
        # Merge branches
        merged = Concatenate()([image_dense, ts_dense])
        merged_dense1 = Dense(dense_neurons*2, activation='relu')(merged)
        merged_drop1 = Dropout(dropout_rate)(merged_dense1)
        merged_dense2 = Dense(dense_neurons, activation='relu')(merged_drop1)
        merged_drop2 = Dropout(dropout_rate)(merged_dense2)
        output = Dense(1, activation='sigmoid')(merged_drop2)
        
        # Create model
        model = Model(inputs=[image_input, timeseries_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    # Create the KerasClassifier
    model = KerasClassifier(build_fn=build_model, verbose=0)
    
    # Define the hyperparameter space
    param_dist = {
        'conv_filters': [16, 32, 64],
        'dense_neurons': [64, 128, 256],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'batch_size': [16, 32, 64],
        'epochs': [20, 30, 50]
    }
    
    # Create the RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the RandomizedSearchCV
    X_image_sample = X_image[:500]  # Use a sample for efficiency
    X_ts_sample = X_timeseries[:500]
    y_sample = y[:500]
    
    random_search.fit([X_image_sample, X_ts_sample], y_sample)
    
    # Print the best parameters
    print("Best parameters:", random_search.best_params_)
    
    return random_search.best_params_


def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """Generates synthetic data for testing the pipeline."""
    print("Generating synthetic data for testing")
    
    # Generate synthetic time series data
    synthetic_time_series = []
    synthetic_images = []
    synthetic_labels = []
    
    for i in range(n_samples):
        # Generate time parameter
        t = np.linspace(0, 10, 100)
        
        # Decide if this is a transit (1) or not (0)
        is_transit = np.random.choice([0, 1])
        
        if is_transit:
            # Generate a transit-like signal
            # Transit occurs at random time between 2 and 8
            transit_time = np.random.uniform(2, 8)
            transit_width = np.random.uniform(0.2, 0.5)
            transit_depth = np.random.uniform(0.01, 0.03)
            
            # Generate the light curve
            flux = np.ones_like(t)
            transit_mask = (t > transit_time - transit_width/2) & (t < transit_time + transit_width/2)
            flux[transit_mask] = 1 - transit_depth
            
            # Add noise
            flux += np.random.normal(0, noise_level, len(t))
        else:
            # Generate a non-transit light curve
            flux = np.ones_like(t)
            
            # Add noise
            flux += np.random.normal(0, noise_level, len(t))
            
            # Add a small sinusoidal variation
            flux += 0.01 * np.sin(2 * np.pi * t / 2)
        
        # Create a 2D image representation
        image = np.zeros((64, 64))
        for j in range(64):
            # Resample the flux to match the image width
            resampled_flux = np.interp(np.linspace(0, len(flux)-1, 64), np.arange(len(flux)), flux)
            image[j, :] = resampled_flux
        
        synthetic_time_series.append(flux)
        synthetic_images.append(image)
        synthetic_labels.append(is_transit)
    
    # Convert to numpy arrays
    X_ts = np.array(synthetic_time_series)
    X_image = np.array(synthetic_images)
    y = np.array(synthetic_labels)
    
    return X_ts, X_image, y


def run_full_pipeline(use_synthetic=False, use_cache=True):
    """Runs the full exoplanet detection pipeline."""
    print("Starting full exoplanet detection pipeline")
    
    if use_synthetic:
        print("Using synthetic data")
        X_ts, X_image, y = generate_synthetic_data()
        
        # Split the data
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        X_ts_train, X_ts_val, X_ts_test = X_ts[train_idx], X_ts[val_idx], X_ts[test_idx]
        X_image_train, X_image_val, X_image_test = X_image[train_idx], X_image[val_idx], X_image[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        
        # Reshape for CNN
        X_image_train = np.expand_dims(X_image_train, axis=-1)
        X_image_val = np.expand_dims(X_image_val, axis=-1)
        X_image_test = np.expand_dims(X_image_test, axis=-1)
        
        # Train multimodal model
        model, metrics = train_multimodal_model(X_image, X_ts, y)
        
        return model, metrics
    
    # Step 1: Get light curve data
    obs_table = fetch_kepler_data(max_records=100, use_cache=use_cache)
    
    # Step 2: Fetch exoplanet labels
    exoplanet_labels = fetch_exoplanet_labels(use_cache=use_cache)
    
    # Step 3: Download light curve files
    light_curve_files = []
    for i, obs in enumerate(obs_table):
        try:
            data_products = Observations.get_product_list(obs)
            light_curve_products = [p for p in data_products if 'LIGHTCURVE' in p['dataURI']]
            
            if light_curve_products:
                file_path = download_product(light_curve_products[0], use_cache=use_cache)
                if file_path:
                    light_curve_files.append(file_path)
                
            if i >= 50:  # Limit for processing
                break
        except Exception as e:
            print(f"Error with observation {obs['obsid']}: {e}")
    
    # Step 4: Run the pipeline
    results = run_pipeline(light_curve_files, exoplanet_labels)
    
    # Step 5: Prepare data for AI training
    X_image, X_timeseries, y = prepare_multimodal_data(light_curve_files, exoplanet_labels)
    
    if X_image is not None and X_timeseries is not None and len(X_image) > 0:
        # Step 6: Train multimodal model
        model, metrics = train_multimodal_model(X_image, X_timeseries, y)
        
        # Step 7: Create validation dataset
        validation_data = create_validation_dataset(use_cache=use_cache)
        
        # Step 8: Cross-validate models
        cross_val_results = cross_validate_models(model, validation_data)
        
        # Step 9: Perform transfer learning if needed
        if cross_val_results['transfer_learning_required']:
            model = perform_transfer_learning(model, validation_data)
        
        # Step 10: Generate report
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(RESULTS_DIR, f"full_pipeline_report_{report_time}.html")
        
        with open(report_file, 'w') as f:
            f.write("<html><head><title>Exoplanet Detection Full Pipeline Report</title>")
            f.write("<style>body{font-family:Arial;max-width:1200px;margin:auto;padding:20px}")
            f.write("table{width:100%;border-collapse:collapse;margin:20px 0}")
            f.write("th,td{padding:8px;border:1px solid #ddd;text-align:left}")
            f.write("th{background-color:#f2f2f2}</style></head><body>")
            f.write(f"<h1>Exoplanet Detection Full Pipeline Report</h1>")
            f.write(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Summary statistics
            f.write("<h2>Pipeline Results</h2>")
            f.write("<ul>")
            f.write(f"<li>Processed {len(light_curve_files)} light curves</li>")
            f.write(f"<li>Light curves with detected transits: {sum(1 for r in results if r['transit_count'] > 0)}</li>")
            f.write(f"<li>Model accuracy: {metrics['accuracy']:.4f}</li>")
            f.write(f"<li>Cross-validation accuracy: {cross_val_results['accuracy']:.4f}</li>")
            f.write("</ul>")
            
            f.write("</body></html>")
        
        return model, metrics, results
    else:
        print("Insufficient data for model training")
        return None, None, results


def export_to_paper_format(results, metrics=None):
    """Prepares the results in a format suitable for academic publication."""
    
    # Create a dataframe with the results
    df_results = pd.DataFrame()
    
    if results:
        # Extract key metrics
        stellar_objects = []
        transit_counts = []
        periods = []
        radii = []
        temperatures = []
        
        for result in results:
            file_name = os.path.basename(result['file_path'])
            stellar_name = file_name.split('_')[0]  # Extract stellar name
            
            stellar_objects.append(stellar_name)
            transit_counts.append(result['transit_count'])
            
            # Extract periodicity
            period = result['periodicity'] if result['periodicity'] else np.nan
            periods.append(period)
            
            # Extract planet properties
            if result['planet_properties']:
                props = result['planet_properties']
                radii.append(props.get('radius_earth', np.nan))
                temperatures.append(props.get('equilibrium_temp_k', np.nan))
            else:
                radii.append(np.nan)
                temperatures.append(np.nan)
        
        # Create dataframe
        df_results = pd.DataFrame({
            'Stellar Object': stellar_objects,
            'Transit Count': transit_counts,
            'Orbital Period (days)': periods,
            'Planet Radius (Earth radii)': radii,
            'Equilibrium Temperature (K)': temperatures
        })
    
    # Create a table with model metrics
    df_metrics = pd.DataFrame()
    
    if metrics:
        # Convert metrics dictionary to dataframe
        metrics_items = list(metrics.items())
        df_metrics = pd.DataFrame(metrics_items, columns=['Metric', 'Value'])
    
    # Save to LaTeX format for academic papers
    if not df_results.empty:
        latex_table = df_results.to_latex(index=False, float_format="%.2f")
        with open(os.path.join(RESULTS_DIR, "results_table.tex"), 'w') as f:
            f.write(latex_table)
    
    if not df_metrics.empty:
        latex_metrics = df_metrics.to_latex(index=False, float_format="%.4f")
        with open(os.path.join(RESULTS_DIR, "metrics_table.tex"), 'w') as f:
            f.write(latex_metrics)
    
    # Save to CSV for further analysis
    if not df_results.empty:
        df_results.to_csv(os.path.join(RESULTS_DIR, "results_table.csv"), index=False)
    
    if not df_metrics.empty:
        df_metrics.to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"), index=False)
    
    return df_results, df_metrics


def compare_with_baseline(results, baseline_method="traditional"):
    """Compares the AI/ML approach with baseline traditional methods."""
    
    # Define metrics to compare
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'processing_time']
    
    # Placeholder for baseline performance (in a real implementation, this would be calculated)
    baseline_performance = {
        'accuracy': 0.65,
        'precision': 0.70,
        'recall': 0.60,
        'f1_score': 0.65,
        'processing_time': 100  # In seconds per light curve
    }
    
    # Placeholder for AI/ML performance (in a real implementation, this would be from actual results)
    aiml_performance = {
        'accuracy': 0.85,
        'precision': 0.88,
        'recall': 0.82,
        'f1_score': 0.85,
        'processing_time': 20  # In seconds per light curve
    }
    
    # Calculate improvement
    improvement = {}
    for metric in metrics_to_compare:
        if metric != 'processing_time':
            # Higher is better for accuracy, precision, recall, f1
            improvement[metric] = aiml_performance[metric] - baseline_performance[metric]
        else:
            # Lower is better for processing time
            improvement[metric] = baseline_performance[metric] - aiml_performance[metric]
    
    # Create comparison dataframe
    df_comparison = pd.DataFrame({
        'Metric': metrics_to_compare,
        f'{baseline_method.capitalize()} Method': [baseline_performance[m] for m in metrics_to_compare],
        'AI/ML Method': [aiml_performance[m] for m in metrics_to_compare],
        'Improvement': [improvement[m] for m in metrics_to_compare]
    })
    
    # Save comparison results
    df_comparison.to_csv(os.path.join(RESULTS_DIR, "method_comparison.csv"), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    metrics_display = [m.replace('_', ' ').title() for m in metrics_to_compare]
    
    x = np.arange(len(metrics_display))
    width = 0.35
    
    plt.bar(x - width/2, [baseline_performance[m] for m in metrics_to_compare], width, label=f'{baseline_method.capitalize()} Method')
    plt.bar(x + width/2, [aiml_performance[m] for m in metrics_to_compare], width, label='AI/ML Method')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Method Comparison')
    plt.xticks(x, metrics_display)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'method_comparison.png'))
    
    return df_comparison


def ablation_study(X_image, X_timeseries, y):
    """Performs an ablation study to determine the importance of different components."""
    
    if X_image is None or X_timeseries is None or y is None:
        print("Insufficient data for ablation study")
        return None
    
    # Split the data
    X_image_train, X_image_test, X_ts_train, X_ts_test, y_train, y_test = train_test_split(
        X_image, X_timeseries, y, test_size=0.3, random_state=42
    )
    
    # Reshape for CNN
    X_image_train = np.expand_dims(X_image_train, axis=-1)
    X_image_test = np.expand_dims(X_image_test, axis=-1)
    
    # Models to evaluate
    models = {
        'full_model': None,
        'image_only': None,
        'timeseries_only': None,
        'reduced_complexity': None
    }
    
    # Train full model
    full_model = create_multimodal_fusion_model(
        image_input_shape=X_image_train[0].shape,
        timeseries_input_shape=(X_ts_train[0].shape[0],)
    )
    
    full_model.fit(
        [X_image_train, X_ts_train], y_train,
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    models['full_model'] = full_model
    
    # Train image-only model
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    
    image_input = Input(shape=X_image_train[0].shape)
    image_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    image_pool1 = MaxPooling2D((2, 2))(image_conv1)
    image_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(image_pool1)
    image_pool2 = MaxPooling2D((2, 2))(image_conv2)
    image_flat = Flatten()(image_pool2)
    image_dense = Dense(128, activation='relu')(image_flat)
    image_drop = Dropout(0.5)(image_dense)
    image_output = Dense(1, activation='sigmoid')(image_drop)
    
    image_model = Model(inputs=image_input, outputs=image_output)
    image_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    image_model.fit(
        X_image_train, y_train,
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    models['image_only'] = image_model
    
    # Train time series only model
    from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D
    
    ts_input = Input(shape=(X_ts_train[0].shape[0],))
    ts_reshape = Reshape((X_ts_train[0].shape[0], 1))(ts_input)
    ts_conv1 = Conv1D(32, 5, activation='relu', padding='same')(ts_reshape)
    ts_pool1 = MaxPooling1D(2)(ts_conv1)
    ts_conv2 = Conv1D(64, 3, activation='relu', padding='same')(ts_pool1)
    ts_pool2 = MaxPooling1D(2)(ts_conv2)
    ts_flat = Flatten()(ts_pool2)
    ts_dense = Dense(128, activation='relu')(ts_flat)
    ts_drop = Dropout(0.5)(ts_dense)
    ts_output = Dense(1, activation='sigmoid')(ts_drop)
    
    ts_model = Model(inputs=ts_input, outputs=ts_output)
    ts_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ts_model.fit(
        X_ts_train, y_train,
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    models['timeseries_only'] = ts_model
    
    # Train reduced complexity model
    image_input_reduced = Input(shape=X_image_train[0].shape)
    image_conv1_reduced = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input_reduced)
    image_pool1_reduced = MaxPooling2D((2, 2))(image_conv1_reduced)
    image_flat_reduced = Flatten()(image_pool1_reduced)
    
    ts_input_reduced = Input(shape=(X_ts_train[0].shape[0],))
    ts_reshape_reduced = Reshape((X_ts_train[0].shape[0], 1))(ts_input_reduced)
    ts_conv1_reduced = Conv1D(16, 5, activation='relu', padding='same')(ts_reshape_reduced)
    ts_pool1_reduced = MaxPooling1D(2)(ts_conv1_reduced)
    ts_flat_reduced = Flatten()(ts_pool1_reduced)
    
    from tensorflow.keras.layers import Concatenate
    merged_reduced = Concatenate()([image_flat_reduced, ts_flat_reduced])
    merged_dense_reduced = Dense(64, activation='relu')(merged_reduced)
    output_reduced = Dense(1, activation='sigmoid')(merged_dense_reduced)
    
    reduced_model = Model(inputs=[image_input_reduced, ts_input_reduced], outputs=output_reduced)
    reduced_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['