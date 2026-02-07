# data/dataset_generator.py

import logging
import os
import numpy as np
import pandas as pd
import re
from astropy.io import fits, ascii
from skimage.transform import resize
from data.augmentation_utils import advanced_phase_folding, augment_data
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from astropy.timeseries import LombScargle
import config
import time
from tqdm import tqdm
from detection.periodicity_analyzer import analyze_periodicity
from detection.transit_detector import apply_transit_modeling, estimate_planet_properties, find_transits_bls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch # Added for LightCurveContrastiveDataset

# Import the new Provenance object
from data.provenance import DataProvenance

logger = logging.getLogger(__name__)

def process_single_file(file_info_dict, image_size, FIXED_LENGTH):
    """Helper function to process a single light curve file."""
    file_path = file_info_dict['file_path']
    current_label = file_info_dict['type']
    
    try:
        start_file_processing = time.time()

        if file_path.endswith(('.fits', '.npz')):
            with fits.open(file_path, mode='readonly') as hdul:
                primary_header = hdul[0].header
                data_header = hdul[1].header
                data = hdul[1].data
                
                time_lc = data.field('TIME')
                flux_lc = data.field('PDCSAP_FLUX')

                # --- Create DataProvenance Object for FITS ---
                source = primary_header.get('TELESCOP', 'Unknown')
                instrument_id = primary_header.get('KEPLERID') or primary_header.get('TICID', 'UnknownID')
                mission_quarter = primary_header.get('QUARTER')
                mission_sector = primary_header.get('SECTOR')
                injected = 'INJECT' in primary_header and primary_header['INJECT']
                detrending_method = 'PDC-MAP' # As this is PDCSAP_FLUX

        elif file_path.endswith('.tbl'):
            table = ascii.read(file_path)
            time_lc = table['TIME'].value
            flux_lc = table['PDCSAP_FLUX'].value
            logger.debug(f"Successfully read .tbl file: {os.path.basename(file_path)}")

            # --- Create DataProvenance Object for .tbl ---
            source = 'Kepler' # .tbl files are typically Kepler
            
            # Extract KEPLERID from filename for .tbl if not in meta
            match_kepid = re.search(r'kplr(\d+)', os.path.basename(file_path))
            instrument_id = int(match_kepid.group(1)) if match_kepid else 'UnknownID'
            
            # Extract Quarter from filename if not in meta (e.g., 2009166043257 -> 2009 Q1)
            match_quarter = re.search(r'(\d{4})(\d{3})', os.path.basename(file_path))
            if match_quarter:
                year = int(match_quarter.group(1))
                day_of_year = int(match_quarter.group(2))
                # Approximate quarter based on day of year
                if day_of_year <= 90: # ~March 31
                    mission_quarter = f"{year}-Q1"
                elif day_of_year <= 181: # ~June 30
                    mission_quarter = f"{year}-Q2"
                elif day_of_year <= 273: # ~September 31
                    mission_quarter = f"{year}-Q3"
                else: # ~December 31
                    mission_quarter = f"{year}-Q4"
            else:
                mission_quarter = None # Or 'UnknownQuarter'
            
            mission_sector = None # Kepler .tbl usually doesn't have a 'sector' concept directly
            detrending_method = 'PDC-MAP' # Assuming PDCSAP_FLUX for .tbl too
            injected = False # Assuming no 'INJECT' flag in .tbl files
        else:
            logger.warning(f"Skipping {os.path.basename(file_path)}: Unsupported file type. Only .fits, .npz, and .tbl are supported.")
            return None

        # Calculate cadence from time array (common to both)
        cadence = np.nanmedian(np.diff(time_lc)) * 24 * 3600 if len(time_lc) > 1 else -1.0

        provenance = DataProvenance(
            source=source,
            instrument_id=str(instrument_id),
            cadence_s=float(cadence),
            mission_quarter=mission_quarter,
            mission_sector=mission_sector,
            detrending_method=detrending_method,
            injected=injected,
        )
        # ---

        finite_mask = np.isfinite(flux_lc) & np.isfinite(time_lc)
        time_lc = time_lc[finite_mask]
        flux_lc = flux_lc[finite_mask]
        logger.debug(f"After finite mask for {os.path.basename(file_path)}: {len(flux_lc)} data points remaining.")

        if len(flux_lc) < 100:
            logger.warning(f"Skipping {os.path.basename(file_path)}: Not enough finite data points ({len(flux_lc)} < 100).")
            return None

        logger.debug(f"Calling find_transits_bls for {os.path.basename(file_path)} with {len(time_lc)} data points.")
        # --- Transit Detection and Periodicity Analysis ---
        transit_info, periodicity_data = find_transits_bls(time_lc, flux_lc)
        
        if transit_info is None or periodicity_data is None:
            logger.warning(f"Skipping {os.path.basename(file_path)}: find_transits_bls returned None. Transit_info: {transit_info}, Periodicity_data: {periodicity_data}")
            return None

        # Use the detected period for phase folding
        period = periodicity_data.get('median_period', np.random.uniform(1, 100))

        # Estimate planet properties
        planet_properties = estimate_planet_properties(transit_info, periodicity_data)

        flux_norm = (flux_lc - np.median(flux_lc)) / np.std(flux_lc)
        start = max(0, len(flux_norm) // 2 - FIXED_LENGTH // 2)
        segment = flux_norm[start : start + FIXED_LENGTH]
        if len(segment) < FIXED_LENGTH:
            segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
        
        # Feature Engineering
        start_feature_engineering = time.time()
        # Limit frequency range to Nyquist frequency for efficiency
        if len(time_lc) > 1:
            nyquist_frequency = 0.5 / (time_lc[1] - time_lc[0])
            frequency, power = LombScargle(time_lc, flux_lc).autopower(nyquist_factor=1, maximum_frequency=nyquist_frequency)
        else:
            frequency, power = LombScargle(time_lc, flux_lc).autopower()
        autocorr = np.correlate(flux_norm, flux_norm, mode='full')[len(flux_norm)-1:]

        # Resize features to a fixed length
        power = resize(power, (config.FEATURE_VECTOR_LENGTH,), preserve_range=True, anti_aliasing=False)
        # autocorr = resize(autocorr, (config.FEATURE_VECTOR_LENGTH,), preserve_range=True, anti_aliasing=False) # Not using autocorr for now

        feature_vector = power
        feature_engineering_time = time.time() - start_feature_engineering

        # Image Generation
        start_image_generation = time.time()
        if len(time_lc) != len(flux_lc):
            logger.warning(f"Time and flux length mismatch for {os.path.basename(file_path)}. Skipping phase folding for 2D image.")
            img_1d = segment[:image_size[0] * image_size[1]]
            if len(img_1d) < image_size[0] * image_size[1]:
                img_1d = np.pad(img_1d, (0, image_size[0] * image_size[1] - len(img_1d)), 'constant', constant_values=0)
            img_2d = img_1d.reshape(image_size)
        else:
            binned_flux = advanced_phase_folding(time_lc, flux_lc, period, n_bins=image_size[0] * image_size[1])
            img_2d = binned_flux.reshape(image_size)

        img_norm = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-8)
        image_generation_time = time.time() - start_image_generation
        
        end_file_processing = time.time()
        logger.debug(f"Processed {os.path.basename(file_path)} in {end_file_processing - start_file_processing:.4f}s (Features: {feature_engineering_time:.4f}s, Image: {image_generation_time:.4f}s)")
        
        # --- Generate and save light curve plot ---
        plot_dir = os.path.join(os.path.dirname(file_path), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        # Remove common extensions for plotting
        if base_name.endswith('.fits'):
            base_name = base_name.replace('.fits', '')
        elif base_name.endswith('.tbl'):
            base_name = base_name.replace('.tbl', '')
        elif base_name.endswith('.npz'):
            base_name = base_name.replace('.npz', '')
        plot_path = os.path.join(plot_dir, f"{base_name}_light_curve.png")

        plt.figure(figsize=(10, 4))
        plt.plot(time_lc, flux_lc, '.-', markersize=2, alpha=0.7)
        if transit_info and transit_info.get('times') is not None and len(transit_info.get('times')) > 0:
            plt.plot(transit_info['times'], flux_lc[transit_info['peak_indices']], 'ro', markersize=5, label='Detected Transits')
        plt.title(f"Light Curve for {os.path.basename(file_path)}")
        plt.xlabel("Time")
        plt.ylabel("Normalized Flux")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return segment, img_norm, feature_vector, (1 if current_label == 'confirmed_planet' else 0), transit_info, periodicity_data, planet_properties, plot_path, provenance

    except Exception as e:
        logger.error(f"FAILED to process {os.path.basename(file_path)}. Error: {e}. Skipping.", exc_info=True)
        return None

def create_dataset(file_paths, output_dir, image_size=(64, 64), max_workers=os.cpu_count()):
    """
    Creates a multimodal dataset (1D time-series, 2D image, and engineered features) from FITS files.
    """
    logger.info(f"Starting multimodal dataset creation with {len(file_paths)} files using {max_workers} workers.")
    
    start_time = time.time()

    all_timeseries = []
    all_images = []
    all_features = []
    all_labels = [] # We still need to collect labels for the final output
    all_provenance = [] # To store provenance objects
    all_pipeline_results_raw = [] # Renamed to avoid confusion with augmented version
    
    FIXED_LENGTH = 2048 # Define a fixed length for time-series segments

    # file_paths is already a list of dictionaries with 'file_path' and 'type' (label)
    file_info_dicts = file_paths 

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_single_file, image_size=image_size, FIXED_LENGTH=FIXED_LENGTH)
        
        results = list(tqdm(executor.map(process_func, file_info_dicts), total=len(file_info_dicts), desc="Processing Light Curves"))

    file_processing_time = time.time() - start_time
    logger.info(f"File processing completed in {file_processing_time:.2f} seconds.")

    for i, result in enumerate(results):
        if result is not None:
            segment, img_norm, feature_vector, label, transit_info, periodicity_data, planet_properties, plot_path, provenance = result
            all_timeseries.append(segment)
            all_images.append(img_norm)
            all_features.append(feature_vector)
            all_labels.append(label)
            all_provenance.append(provenance)
            all_pipeline_results_raw.append({
                'file_path': file_info_dicts[i]['file_path'], # Original file path
                'success': True,
                'transit_count': len(transit_info.get('times', [])),
                'periodicity': periodicity_data.get('median_period'),
                'bls_max_power': periodicity_data.get('max_power'),
                'planet_properties': planet_properties,
                'light_curve_plot_path': plot_path # Store the full path to the plot
            })

    if not all_timeseries:
        logger.error("CRITICAL: No files were successfully processed.")
        return None, None, None, None, None, None # Return None for all expected outputs

    X_ts = np.array(all_timeseries)
    X_img = np.array(all_images)[..., np.newaxis]
    X_features = np.array(all_features)
    y = np.array(all_labels)

    logger.info(f"Multimodal dataset created successfully with {len(y)} samples (before augmentation).")
    print(f"DEBUG: all_pipeline_results_raw is defined: {'all_pipeline_results_raw' in locals()}") # Add this line
    return X_ts, X_img, X_features, y, all_pipeline_results_raw, all_provenance

def create_contrastive_dataset(file_paths, output_dir, FIXED_LENGTH=config.FIXED_LENGTH, augmentation_factor=config.AUGMENTATION_FACTOR):
    """
    Creates a dataset specifically for contrastive pre-training.
    It processes light curve files, normalizes them, segments them to a fixed length,
    and applies augmentation to create two augmented versions (X_aug1, X_aug2).
    These augmented versions are saved as .npy files in the output_dir.
    """
    logger.info(f"Starting contrastive dataset creation with {len(file_paths)} files.")

    all_segments = []
    
    # Process files to get raw time-series segments
    for file_path in tqdm(file_paths, desc="Processing files for contrastive dataset"):
        try:
            with fits.open(file_path, mode='readonly') as hdul:
                data = hdul[1].data
                time_lc = data.field('TIME')
                flux_lc = data.field('PDCSAP_FLUX')

                finite_mask = np.isfinite(flux_lc) & np.isfinite(time_lc)
                time_lc = time_lc[finite_mask]
                flux_lc = flux_lc[finite_mask]

                if len(flux_lc) < 100:
                    logger.warning(f"Skipping {os.path.basename(file_path)}: Not enough finite data points for contrastive dataset.")
                    continue

                flux_norm = (flux_lc - np.median(flux_lc)) / np.std(flux_lc)
                start = max(0, len(flux_norm) // 2 - FIXED_LENGTH // 2)
                segment = flux_norm[start : start + FIXED_LENGTH]
                if len(segment) < FIXED_LENGTH:
                    segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)), 'constant', constant_values=0)
                
                all_segments.append(segment)

        except Exception as e:
            logger.error(f"FAILED to process {os.path.basename(file_path)} for contrastive dataset. Error: {e}. Skipping.", exc_info=True)
            continue

    if not all_segments:
        logger.error("CRITICAL: No segments were successfully processed for contrastive dataset.")
        return

    X_raw = np.array(all_segments)
    logger.info(f"Raw segments for contrastive dataset: {X_raw.shape}")

    # Apply augmentation twice to create two augmented views
    # The augment_data function expects a list of arrays for X, and y. Here y is dummy.
    # We need to adapt it for single time-series augmentation.
    # Let's create a simplified augmentation for contrastive learning if augment_data is too complex.
    
    # For simplicity, let's use a basic augmentation strategy for contrastive learning:
    # 1. Random noise
    # 2. Random scaling
    
    X_aug1 = []
    X_aug2 = []

    for segment in tqdm(X_raw, desc="Augmenting data for contrastive dataset"):
        # Augmentation 1: Add random noise
        noise1 = np.random.normal(0, 0.01, segment.shape)
        aug_segment1 = segment + noise1
        X_aug1.append(aug_segment1)

        # Augmentation 2: Random scaling
        scale_factor = np.random.uniform(0.9, 1.1)
        aug_segment2 = segment * scale_factor
        X_aug2.append(aug_segment2)

    X_aug1 = np.array(X_aug1)
    X_aug2 = np.array(X_aug2)

    np.save(os.path.join(output_dir, 'X_aug1.npy'), X_aug1)
    np.save(os.path.join(output_dir, 'X_aug2.npy'), X_aug2)
    logger.info(f"Contrastive dataset created and saved to {output_dir}.")


class LightCurveContrastiveDataset(torch.utils.data.Dataset):
    """
    Dataset for CACL pre-training, loading two augmented views of light curves.
    """
    def __init__(self, data_dir):
        self.X_aug1 = np.load(os.path.join(data_dir, 'X_aug1.npy'))
        self.X_aug2 = np.load(os.path.join(data_dir, 'X_aug2.npy'))
        assert len(self.X_aug1) == len(self.X_aug2)

    def __len__(self):
        return len(self.X_aug1)

    def __getitem__(self, idx):
        return torch.tensor(self.X_aug1[idx], dtype=torch.float32), \
               torch.tensor(self.X_aug2[idx], dtype=torch.float32)