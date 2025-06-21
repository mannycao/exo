"""
Functions for processing light curve data, including preprocessing,
transit detection, and feature extraction for machine learning models.
"""
import logging
import numpy as np
import skimage
import pandas as pd # For DataFrame creation if needed
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel # convolve and Box1DKernel are not used in current version, can be removed if not planned
from scipy.signal import find_peaks, savgol_filter
from skimage.transform import resize # For create_image_representations
from pathlib import Path # Ensure Path is imported if used for file_path manipulation

# Attempt to import config from parent directory
try:
    from .. import config 
except ImportError:
    try:
        import config
    except ImportError:
        class DummyConfig: 
            TRANSIT_SENSITIVITY = 3.0
            MIN_TRANSIT_DURATION = 0.05 
            MAX_TRANSIT_DURATION = 0.5  
            WINDOW_SIZE = 0.5 
            IMAGE_SIZE = (64,64)
        config = DummyConfig()
        logging.error("Could not import config.py in light_curve_processor.py. Using dummy config.")


logger = logging.getLogger(__name__)

def preprocess_light_curve(file_path):
    """
    Load and preprocess light curve data from a FITS file.
    - Reads time and flux.
    - Removes NaNs.
    - Performs basic normalization/detrending.
    - Extracts relevant metadata.
    """
    logger.debug(f"Starting preprocess_light_curve for: {file_path}")
    metadata = {'file_path_original': str(file_path)} # Store original path as string
    time_array = None
    flux_array = None

    try:
        with fits.open(file_path) as hdul:
            logger.debug(f"  Successfully opened FITS file: {file_path}")
            if len(hdul) < 2 or not hasattr(hdul[1], 'data') or hdul[1].data is None:
                logger.error(f"  FITS file {file_path} does not have required HDU1 data structure.")
                metadata['error'] = 'Invalid FITS structure (no HDU1 data)'
                return None, None, metadata
            
            data = hdul[1].data 
            header = hdul[0].header 
            
            logger.debug(f"  HDU1 Columns: {data.columns.names}")

            if 'TIME' in data.columns.names:
                time_array = data.field('TIME')
                logger.debug(f"  Read 'TIME' column, initial shape: {time_array.shape if hasattr(time_array, 'shape') else 'N/A'}")
            else:
                logger.error(f"  'TIME' column not found in FITS file: {file_path}")
                metadata['error'] = "'TIME' column not found"
                return None, None, metadata

            flux_col_name = None
            if 'PDCSAP_FLUX' in data.columns.names:
                flux_col_name = 'PDCSAP_FLUX'
            elif 'SAP_FLUX' in data.columns.names:
                flux_col_name = 'SAP_FLUX'
            
            if flux_col_name:
                flux_array = data.field(flux_col_name)
                logger.debug(f"  Read '{flux_col_name}' column, initial shape: {flux_array.shape if hasattr(flux_array, 'shape') else 'N/A'}")
            else:
                logger.error(f"  No PDCSAP_FLUX or SAP_FLUX column in FITS file: {file_path}")
                metadata['error'] = 'Flux column (PDCSAP_FLUX or SAP_FLUX) not found'
                return None, None, metadata

            # Extract metadata
            metadata.update({
                'object_id': header.get('OBJECT', header.get('KEPLERID', Path(file_path).stem)),
                'telescope': header.get('TELESCOP', 'Unknown'),
                'ra': header.get('RA_OBJ', None),
                'dec': header.get('DEC_OBJ', None),
                'kepler_mag': header.get('KEPLRMAG', None),
                'st_rad': float(header.get('ST_RAD', header.get('RADIUS', 1.0))), # Ensure float
                'st_mass': float(header.get('ST_MASS', header.get('MASS', 1.0))), # Ensure float
                'st_teff': float(header.get('ST_TEFF', header.get('TEFF', 5778)))  # Ensure float
            })
            
            # Cadence calculation from header (best effort) or from time diff later
            # Common TESS cadence keyword is 'TIMEDEL', Kepler is 'EXPOSURE' or 'INT_TIME'
            cadence_s_header = header.get('TIMEDEL', header.get('EXPOSURE', header.get('INT_TIME'))) 
            if cadence_s_header is not None:
                try:
                    metadata['cadence_days'] = float(cadence_s_header) / (24.0 * 60.0 * 60.0)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse cadence '{cadence_s_header}' from header of {file_path}. Will estimate from time array.")
                    metadata['cadence_days'] = None # Mark for later calculation
            else:
                logger.debug(f"No explicit cadence keyword found in header of {file_path}. Will estimate from time array.")
                metadata['cadence_days'] = None

            # Handle BJD offsets if present
            bjdrefi = data.field('BJDREFI') if 'BJDREFI' in data.columns.names else 0.0
            bjdreff = data.field('BJDREFF') if 'BJDREFF' in data.columns.names else 0.0
            if not isinstance(bjdrefi, (int, float, np.number)): bjdrefi = 0.0 # Ensure numeric
            if not isinstance(bjdreff, (int, float, np.number)): bjdreff = 0.0 # Ensure numeric

            time_array = np.asarray(time_array, dtype=float) + bjdrefi + bjdreff
            flux_array = np.asarray(flux_array, dtype=float)
        
        logger.debug(f"  Data read from FITS. Time points: {len(time_array)}, Flux points: {len(flux_array)}")

        # Remove NaNs and infinite values
        valid_indices = np.isfinite(time_array) & np.isfinite(flux_array)
        time_array = time_array[valid_indices]
        flux_array = flux_array[valid_indices]
        logger.debug(f"  After NaN/inf removal: Time points: {len(time_array)}, Flux points: {len(flux_array)}")

        if len(time_array) == 0 or len(flux_array) == 0:
            logger.warning(f"No valid data after NaN/inf removal for {file_path}")
            metadata['error'] = 'No valid data after NaN/inf removal'
            return None, None, metadata
        if len(time_array) < 2 : # Need at least 2 points for diff to calculate cadence
            logger.warning(f"Too few time points ({len(time_array)}) after NaN/inf removal for {file_path} to robustly process.")
            metadata['error'] = 'Too few data points after NaN/inf removal'
            return None, None, metadata


        # If cadence wasn't found in header or was invalid, calculate from median time diff
        if metadata.get('cadence_days') is None or metadata.get('cadence_days', 0) <= 0:
            median_diff_time = np.median(np.diff(time_array))
            if median_diff_time > 0 and not np.isnan(median_diff_time):
                metadata['cadence_days'] = median_diff_time
                logger.debug(f"  Calculated median cadence from time array: {metadata['cadence_days']:.6f} days for {file_path}")
            else:
                default_cad_days = (2.0 / (24.0 * 60.0)) # Default 2-min
                logger.warning(f"  Could not calculate valid median cadence for {file_path}. Using default: {default_cad_days:.6f} days.")
                metadata['cadence_days'] = default_cad_days


        # Normalize flux (simple median normalization)
        median_flux = np.median(flux_array)
        if median_flux != 0 and not np.isnan(median_flux):
            flux_normalized = flux_array / median_flux
            logger.debug(f"  Flux normalized by median: {median_flux:.4e}")
        else:
            logger.warning(f"Median flux is zero or NaN for {file_path}. Using raw flux for detrending (if any).")
            flux_normalized = flux_array 

        # Basic detrending (e.g., Savitzky-Golay filter)
        # Ensure window_length is odd and smaller than the data length.
        # A common choice for Kepler SC is around 2 days window ~ 1440 points / (2 day / 2 min_cadence)
        # Let's use a window relative to expected signal duration, e.g., 5-10 times MAX_TRANSIT_DURATION
        detrend_window_days = getattr(config, 'DETRENDING_WINDOW_DAYS', 5 * config.MAX_TRANSIT_DURATION) # Example
        detrend_window_cadences = int(detrend_window_days / metadata['cadence_days']) if metadata['cadence_days'] > 0 else 101
        
        # Ensure window is odd and at least 5, and less than data length
        detrend_window_cadences = max(5, detrend_window_cadences if detrend_window_cadences % 2 != 0 else detrend_window_cadences + 1)
        
        polyorder = 2 # Must be less than window_length
        flux_detrended = flux_normalized # Default if detrending fails or is skipped

        if len(flux_normalized) > detrend_window_cadences and detrend_window_cadences > polyorder:
            try:
                trend = savgol_filter(flux_normalized, detrend_window_cadences, polyorder)
                flux_detrended = flux_normalized / trend # Divide by trend, then re-center around 1
                flux_detrended = flux_detrended / np.median(flux_detrended) # Re-normalize after division
                logger.debug(f"  Applied Savitzky-Golay detrending with window {detrend_window_cadences} cadences.")
            except ValueError as sve: 
                logger.warning(f"Savitzky-Golay filter failed for {file_path} (window={detrend_window_cadences}, N={len(flux_normalized)}): {sve}. Using normalized flux without this detrending step.")
                # flux_detrended remains flux_normalized
        else:
            logger.debug(f"Skipping Savitzky-Golay for {file_path} due to insufficient data points relative to window/polyorder (N={len(flux_normalized)}, W={detrend_window_cadences}, P={polyorder}).")

        logger.info(f"Preprocessing successful for {file_path}. Final time points: {len(time_array)}, Final flux points: {len(flux_detrended)}")
        return time_array, flux_detrended, metadata

    except FileNotFoundError:
        logger.error(f"FITS file not found: {file_path}", exc_info=True)
        metadata['error'] = 'File not found'
        return None, None, metadata
    except Exception as e:
        logger.error(f"Unhandled error during preprocessing of {file_path}: {e}", exc_info=True)
        metadata['error'] = f"General preprocessing error: {str(e)}"
        return None, None, metadata


def detect_transits(time, flux, sensitivity=3.0, 
                    min_duration_cadences=3, max_duration_cadences=30,
                    prominence_factor=0.5):
    """
    Detects transit-like dips in a light curve.
    Now accepts min_duration_cadences and max_duration_cadences.
    """
    # time_array_valid and flux_array_valid are defined in pipeline_runner.py
    # For this module to be self-contained for testing, they should be defined here too or imported.
    # Let's define simplified versions here if not available via import
    def _is_valid_array(arr):
        return arr is not None and isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) > 0

    if not _is_valid_array(time) or not _is_valid_array(flux) or len(time) != len(flux):
        logger.error("Invalid time or flux arrays provided to detect_transits.")
        return None
    
    logger.debug(f"Detecting transits with sensitivity={sensitivity}, min_dur_cad={min_duration_cadences}, max_dur_cad={max_duration_cadences}, prom_factor={prominence_factor}")

    try:
        # Invert flux to find peaks (dips become peaks)
        # Adding a small constant if flux is all zeros or very flat, before inversion.
        if np.all(flux == flux[0]): # Flat light curve
            inverted_flux = np.zeros_like(flux) # No peaks to find
            logger.debug("  Flux is flat, no peaks will be found in inverted_flux.")
        else:
            inverted_flux = -flux + (np.max(flux) + 1e-6) # Ensure positive values for find_peaks

        flux_std = np.std(flux)
        if flux_std == 0: # Handle flat or near-flat light curves where std might be zero
            flux_std = 1e-6 # A very small non-zero std to prevent division by zero or zero height/prominence
            logger.debug("  Flux standard deviation is zero, using small epsilon.")
        
        # Height: minimum peak height in inverted_flux. This corresponds to dip depth.
        # Dips must be deeper than sensitivity * flux_std from the mean (or from a local baseline).
        # For find_peaks on inverted_flux, height is absolute.
        # Let's set height relative to the minimum of inverted_flux (which is max of original flux)
        min_peak_height = sensitivity * flux_std
        
        # Prominence: how much a peak stands out from its surroundings.
        min_prominence = prominence_factor * flux_std
        
        # Ensure min_duration and max_duration are sensible
        min_dur = max(1, int(min_duration_cadences))
        max_dur = max(min_dur + 1, int(max_duration_cadences)) # Ensure max is greater than min
        
        logger.debug(f"  find_peaks params: height={min_peak_height:.4e}, prominence={min_prominence:.4e}, width=({min_dur}, {max_dur})")

        peaks_indices, properties = find_peaks(
            inverted_flux, 
            height=min_peak_height, 
            prominence=min_prominence, 
            width=(min_dur, max_dur) 
        )

        if len(peaks_indices) == 0:
            logger.debug("  No peaks found by find_peaks matching criteria in detect_transits.")
            return None

        transit_times = time[peaks_indices]
        # Depths are trickier with find_peaks on inverted flux.
        # 'peak_heights' from properties are absolute heights in inverted_flux.
        # 'prominences' are often a better measure of relative depth.
        # Let's use prominences as the primary depth measure here.
        transit_depths = properties.get("prominences", np.zeros_like(peaks_indices, dtype=float))
        
        # Durations (widths from find_peaks are often at half-prominence)
        # Use left_ips and right_ips for a more robust width at base or near base.
        durations_cadences = []
        if 'left_ips' in properties and 'right_ips' in properties:
            durations_cadences = properties['right_ips'] - properties['left_ips']
        elif 'widths' in properties: 
            durations_cadences = properties['widths'] # Width at half-prominence
        else: 
            # Fallback using the min_duration if specific widths not available
            # This is not ideal, means find_peaks didn't return detailed width info
            durations_cadences = np.full_like(peaks_indices, min_dur, dtype=float)
            logger.warning("  Width information ('left_ips'/'right_ips' or 'widths') not found in find_peaks properties. Using min_duration as fallback duration.")


        transit_info_dict = {
            'times': transit_times.tolist(),
            'depths': transit_depths.tolist(), 
            'durations_cadences': durations_cadences.tolist(),
            'peak_indices': peaks_indices.tolist(),
            'properties': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in properties.items()}
        }
        logger.debug(f"  Detected {len(peaks_indices)} transits by find_peaks. Sample times: {transit_times.tolist()[:3]}")
        return transit_info_dict

    except Exception as e:
        logger.error(f"Error in detect_transits: {e}", exc_info=True)
        return None


def extract_transit_features(time, flux, transit_info, window_size_cadences=200):
    """
    Extracts segments of the light curve around detected transits.
    `window_size_cadences` is the total number of data points for the segment.
    """
    if transit_info is None or not transit_info.get('peak_indices'):
        logger.debug("extract_transit_features: No transit_info or peak_indices provided.")
        return []

    peak_indices = transit_info['peak_indices']
    segments = []
    
    if window_size_cadences <= 0 :
        logger.error(f"Invalid window_size_cadences: {window_size_cadences}. Must be positive.")
        return []
    half_window = window_size_cadences // 2

    logger.debug(f"Extracting features with window_size_cadences: {window_size_cadences}")

    for peak_idx in peak_indices:
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(flux), peak_idx + half_window + (window_size_cadences % 2)) # Ensure full window if possible
        
        segment = flux[start_idx:end_idx]
        
        if len(segment) < window_size_cadences:
            padding_needed = window_size_cadences - len(segment)
            pad_value = np.median(segment) if len(segment) > 0 else np.median(flux) # Use overall median if segment is empty
            if np.isnan(pad_value) or np.isinf(pad_value): pad_value = 1.0 

            if peak_idx - half_window < 0: 
                padding = np.full(padding_needed, pad_value)
                segment = np.concatenate([padding, segment])
            else: 
                padding = np.full(padding_needed, pad_value)
                segment = np.concatenate([segment, padding])
        
        if len(segment) > window_size_cadences:
            segment = segment[:window_size_cadences]
        elif len(segment) < window_size_cadences: 
             logger.warning(f"Segment for peak {peak_idx} still too short ({len(segment)}) after padding. Required {window_size_cadences}. Skipping this feature.")
             continue

        seg_median = np.median(segment)
        if seg_median != 0 and not np.isnan(seg_median) and not np.isinf(seg_median):
            normalized_segment = segment / seg_median 
        else:
            logger.warning(f"Segment median is zero, NaN, or Inf for peak {peak_idx}. Using segment as is for ML feature (may cause issues).")
            normalized_segment = segment.copy() # Avoid modifying original segment if it's a view
        
        segments.append(normalized_segment)
        
    logger.debug(f"Extracted {len(segments)} transit segments for ML.")
    return segments


def create_image_representations(segments, image_size=(64, 64)):
    """
    Converts 1D light curve segments into 2D image representations using skimage.transform.resize.
    """
    if not segments or len(segments) == 0:
        logger.debug("create_image_representations: No segments provided.")
        return []
        
    images = []
    target_h, target_w = image_size

    for i, segment in enumerate(segments):
        if segment is None or not isinstance(segment, np.ndarray) or segment.ndim != 1 or len(segment) == 0:
            logger.warning(f"Segment {i} is None, not a 1D numpy array, or empty. Skipping image creation for this segment.")
            continue
        try:
            # Skimage resize expects image of at least 2 dimensions for image_output
            # Reshape 1D segment to (1, N) or (N, 1) before resizing to (H, W)
            # For simple resizing, treating it as a (1, N) image often works.
            # Or, one could make it roughly square first if desired.
            # Let's try to make it a 1-row image
            segment_as_row = segment.reshape(1, -1)
            
            # Resize to target image_size. preserve_range=True is important if data isn't [0,1]
            # anti_aliasing=True is generally good.
            image = resize(segment_as_row, (target_h, target_w), anti_aliasing=True, mode='reflect', preserve_range=True)
            
            # Resize output is (H, W). If a channel dimension is needed for CNN, add it.
            # image = image[..., np.newaxis] # -> (H, W, 1) - This is usually done in reshape_data_for_cnn
            images.append(image)
        except ImportError: # Should have been caught earlier, but good to have specific error here
            logger.critical("scikit-image is not installed, cannot create image representations. Please run 'pip install scikit-image'.")
            raise # Re-raise the import error if it somehow got here
        except Exception as e:
            logger.error(f"Error creating image from segment {i}: {e}. Segment shape: {segment.shape}", exc_info=True)
            # Optionally append a placeholder or skip
            # images.append(np.zeros(image_size)) 
            continue
            
    logger.debug(f"Created {len(images)} image representations of size {image_size}.")
    return images


def reshape_data_for_cnn(X_image_data, image_size_config=None): # image_size_config for clarity
    """
    Reshapes image data to be suitable for CNN input (e.g., add channel dimension).
    """
    if X_image_data is None: # Check for None
        logger.warning("X_image_data is None in reshape_data_for_cnn. Returning empty array.")
        return np.array([])
    if not isinstance(X_image_data, np.ndarray): # Ensure it's an array
        try:
            X_image_data = np.asarray(X_image_data)
        except Exception as e:
            logger.error(f"Could not convert X_image_data to NumPy array: {e}. Returning empty array.")
            return np.array([])
    if X_image_data.size == 0: # Check for empty array after conversion
        logger.debug("X_image_data is empty in reshape_data_for_cnn.")
        return X_image_data # Return the empty array as is

    final_shape = None
    if X_image_data.ndim == 3: # (num_samples, height, width)
        final_shape = X_image_data.shape + (1,)  # Add channel dimension
    elif X_image_data.ndim == 2: 
        logger.warning(f"Input image data has 2 dimensions {X_image_data.shape}. Attempting reshape based on config.IMAGE_SIZE.")
        img_h, img_w = image_size_config if image_size_config else getattr(config, 'IMAGE_SIZE', (64,64))
        if X_image_data.shape[1] == img_h * img_w: # Assume it's flattened (num_samples, H*W)
            final_shape = (-1, img_h, img_w, 1)
        else:
            logger.error(f"Cannot reliably reshape 2D image data of shape {X_image_data.shape} using image_size {img_h}x{img_w}.")
            return X_image_data # Return as is, may cause error later
    elif X_image_data.ndim == 4: # Already in (num_samples, height, width, channels)
        logger.debug("Image data already has 4 dimensions, assuming correct for CNN.")
        return X_image_data # Already in correct shape
    else:
        logger.error(f"Unexpected image data dimension: {X_image_data.ndim}. Cannot reshape for CNN.")
        return X_image_data 

    if final_shape:
        try:
            X_image_data = X_image_data.reshape(final_shape)
            logger.debug(f"Reshaped image data for CNN to shape: {X_image_data.shape}")
        except ValueError as e:
            logger.error(f"ValueError during reshape to {final_shape}: {e}. Original shape: {X_image_data.shape}")
            # Return original if reshape fails, to allow further debugging
            return X_image_data 
            
    return X_image_data
