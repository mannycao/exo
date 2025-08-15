# data/fusion_dataset_generator.py

import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- Add project root for imports ---
try:
    project_root = Path(__file__).resolve().parents[1]
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except IndexError:
    project_root = Path(__file__).resolve().parent
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from data.real_data_fetcher import smart_data_fetcher
from data.multi_method_fetcher import fetch_all_data_for_target

from legacy_pipeline import preprocess_light_curve
from data.feature_extractor import extract_ttv_features

logger = logging.getLogger(__name__)

def standardize(arr):
    arr = arr.astype(np.float32)
    mean = arr.mean(axis=tuple(range(1, arr.ndim)), keepdims=True)
    std = arr.std(axis=tuple(range(1, arr.ndim)), keepdims=True)
    return (arr - mean) / (std + 1e-7)

def create_fusion_dataset(planets_dir, false_positives_dir, max_files=None, ttv_dim=7, use_rv=True, use_imaging=True):
    """Creates a full multi-method dataset for the Bayesian Fusion Model.
    Args:
        planets_dir: Directory with planet FITS files.
        false_positives_dir: Directory with false positive FITS files.
        max_files: If set, limit the number of files used (for testing).
    """
    logger.info("--- Starting Multi-Method Fusion Dataset Creation ---")

    all_fits_files = smart_data_fetcher(planets_dir, false_positives_dir)
    if not all_fits_files:
        logger.error("No FITS files found. Aborting.")
        return None

    # --- Log class balance for debugging imbalance ---
    from collections import Counter
    class_counts = Counter([1 if 'confirmed_planets' in info['file_path'] else 0 for info in all_fits_files])
    logger.info(f"Class balance in all_fits_files: {dict(class_counts)}")

    if max_files is not None and len(all_fits_files) > max_files:
        # --- Enhanced stratified sampling: ensure minimum multi-method targets per class ---
        from collections import defaultdict
        import random
        import pickle, os
        class_to_files = defaultdict(list)
        for info in all_fits_files:
            label = 1 if 'confirmed_planets' in info['file_path'] else 0
            class_to_files[label].append(info)

        # Load or build multi-method cache (rv/imaging mask > 0)
        mm_cache_path = os.path.join("results", "rv_imaging_cache.pkl")
        if os.path.exists(mm_cache_path):
            logger.info(f"Loading cached RV/Imaging data from {mm_cache_path}")
            with open(mm_cache_path, "rb") as f:
                mm_data = pickle.load(f)
            # Map file_path to multi-method status
            mm_status = {d['transit_file_path']: (d['rv_mask'] > 0 or d['imaging_mask'] > 0) for d in mm_data if 'transit_file_path' in d}
        else:
            mm_status = {}

        n_classes = len(class_to_files)
        n_per_class = max_files // n_classes
        min_mm_per_class = min(50, n_per_class // 2)  # Try to get at least 50 multi-method per class if possible
        sampled = []
        for label, files in class_to_files.items():
            # Split into multi-method and not
            mm_files = [f for f in files if mm_status.get(f['file_path'], False)]
            non_mm_files = [f for f in files if not mm_status.get(f['file_path'], False)]
            n_mm = min(len(mm_files), min_mm_per_class)
            n_non_mm = n_per_class - n_mm
            sampled.extend(random.sample(mm_files, n_mm))
            if len(non_mm_files) > n_non_mm:
                sampled.extend(random.sample(non_mm_files, n_non_mm))
            else:
                sampled.extend(non_mm_files)
        # If not enough files due to class imbalance, fill up to max_files with remaining files
        if len(sampled) < max_files:
            leftovers = [info for info in all_fits_files if info not in sampled]
            sampled.extend(random.sample(leftovers, min(len(leftovers), max_files - len(sampled))))
        random.shuffle(sampled)
        logger.warning(f"Enhanced stratified sampling: limiting dataset to {len(sampled)} files (max_files={max_files}), with at least {min_mm_per_class} multi-method per class if available.")
        all_fits_files = sampled
    file_paths = [info['file_path'] for info in all_fits_files]

    import os, pickle
    cache_path = os.path.join("results", "rv_imaging_cache.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached RV/Imaging data from {cache_path}")
        with open(cache_path, "rb") as f:
            multi_method_data = pickle.load(f)
    else:
        logger.info(f"Fetching RV/Imaging data for {len(file_paths)} targets...")
        multi_method_data = []
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(fetch_all_data_for_target, file_paths), total=len(file_paths), desc="Fetching Follow-up Data"))
            for res in results:
                if res is not None:
                    multi_method_data.append(res)
        with open(cache_path, "wb") as f:
            pickle.dump(multi_method_data, f)

    # --- FIX: Add explicit logging for data fetching success rate ---
    rv_found_count = sum(1 for d in multi_method_data if d['rv_mask'] > 0)
    imaging_found_count = sum(1 for d in multi_method_data if d['imaging_mask'] > 0)
    logger.info(f"Successfully found RV data for {rv_found_count}/{len(multi_method_data)} targets.")
    logger.info(f"Successfully found Imaging data for {imaging_found_count}/{len(multi_method_data)} targets.")
    # Log multi-method counts per class
    mm_mask = [(d['rv_mask'] > 0 or d['imaging_mask'] > 0) for d in multi_method_data]
    y_labels_tmp = [1 if 'confirmed_planets' in d['transit_file_path'] else 0 for d in multi_method_data]
    mm_counts = {0: 0, 1: 0}
    for i, is_mm in enumerate(mm_mask):
        if is_mm:
            mm_counts[y_labels_tmp[i]] += 1
    logger.info(f"Multi-method samples per class in dataset: {mm_counts}")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    logger.info(f"Processing {len(multi_method_data)} transit light curves...")
    transit_paths_to_process = [data['transit_file_path'] for data in multi_method_data]
    
    processed_transits_cache_path = os.path.join("results", "processed_transits_cache.pkl")
    if os.path.exists(processed_transits_cache_path):
        logger.info(f"Loading cached processed transit data from {processed_transits_cache_path}")
        with open(processed_transits_cache_path, "rb") as f:
            processed_transits = pickle.load(f)
    else:
        processed_transits = {}
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(preprocess_light_curve, transit_paths_to_process), total=len(transit_paths_to_process), desc="Processing Transit Data"))
            for path, result in zip(transit_paths_to_process, results):
                if result is not None and result[0] is not None:
                    processed_transits[path] = result
        with open(processed_transits_cache_path, "wb") as f:
            pickle.dump(processed_transits, f)

    logger.info("Assembling final dataset arrays...")
    y_labels, X_transit_img, X_transit_ts, X_rv, X_rv_mask, X_img, X_img_mask, X_ttv = [], [], [], [], [], [], [], []

    for data in multi_method_data:
        path = data['transit_file_path']
        if path in processed_transits:
            transit_img, transit_ts = processed_transits[path]
            X_transit_img.append(transit_img)
            X_transit_ts.append(transit_ts)
            # --- RV ---
            if use_rv:
                rv_data = data.get('rv_data')
                if rv_data is None or not hasattr(rv_data, 'shape') or rv_data.shape != (50, 1):
                    rv_data = np.zeros((50, 1), dtype=np.float32)
                    rv_mask = 0.0
                else:
                    rv_mask = data.get('rv_mask', 0.0)
                X_rv.append(rv_data)
                X_rv_mask.append(rv_mask)
            # --- Imaging ---
            if use_imaging:
                imaging_data = data.get('imaging_data')
                if imaging_data is None or not hasattr(imaging_data, 'shape') or imaging_data.shape != (2,):
                    imaging_data = np.zeros(2, dtype=np.float32)
                    imaging_mask = 0.0
                else:
                    imaging_mask = data.get('imaging_mask', 0.0)
                X_img.append(imaging_data)
                X_img_mask.append(imaging_mask)
            # --- TTV Feature Extraction ---
            if hasattr(transit_ts, 'shape') and transit_ts.shape[0] > 1:
                N = transit_ts.shape[0]
                time = np.linspace(0, 1, N)
                flux = transit_ts.squeeze()
                lc = np.stack([time, flux])  # shape (2, N)
                lc = lc.reshape(1, 2, N)
                ttv_feat = extract_ttv_features(lc, ttv_dim)
                X_ttv.append(ttv_feat[0])
            else:
                X_ttv.append(np.zeros(ttv_dim, dtype=np.float32))
            y_labels.append(1 if 'confirmed_planets' in path else 0)

    if not y_labels:
        logger.error("Final dataset is empty after processing. Aborting.")
        return None

    # Debug: Print and log all unique shapes and their counts
    from collections import Counter
    def log_shape_counts(name, arr):
        shapes = [np.shape(x) for x in arr]
        shape_counts = Counter(shapes)
        logger.info(f"{name} unique shapes and counts: {dict(shape_counts)}")
        return shapes, shape_counts

    img_shapes, img_shape_counts = log_shape_counts("X_transit_image", X_transit_img)
    ts_shapes, ts_shape_counts = log_shape_counts("X_transit_timeseries", X_transit_ts)


    # Pad/crop and reshape to match model config
    TS_TARGET_LEN = 200
    IMG_TARGET_LEN = 64 * 64  # 4096
    IMG_TARGET_SHAPE = (64, 64, 1)
    TS_TARGET_SHAPE = (200, 1)
    RV_TARGET_LEN = 50
    RV_TARGET_SHAPE = (50, 1)


    def pad_or_crop(arr, target_len):
        arr = np.asarray(arr)
        if arr.shape[0] == target_len:
            return arr
        elif arr.shape[0] > target_len:
            return arr[:target_len]
        else:
            pad_width = target_len - arr.shape[0]
            return np.pad(arr, (0, pad_width), mode='constant')


    # Process transit time series
    X_transit_ts = [pad_or_crop(a, TS_TARGET_LEN).reshape(TS_TARGET_SHAPE) for a in X_transit_ts]
    # Process transit images: pad/crop to 4096, reshape to (64,64,1)
    X_transit_img = [pad_or_crop(a, IMG_TARGET_LEN).reshape(64, 64, 1) for a in X_transit_img]
    # Process RV data: pad/crop to 50, reshape to (50,1)
    X_rv = [pad_or_crop(a, RV_TARGET_LEN).reshape(RV_TARGET_SHAPE) for a in X_rv]

    logger.info(f"All transit time series padded/cropped to {TS_TARGET_SHAPE}, images to {IMG_TARGET_SHAPE}.")
    logger.info("--- Fusion Dataset Creation Finished ---")
    # Ensure all arrays are np.array with correct dtype and shape
    X_transit_img = np.stack([np.asarray(x, dtype=np.float32) for x in X_transit_img]) if X_transit_img else np.empty((0, *IMG_TARGET_SHAPE), dtype=np.float32)
    X_transit_ts = np.stack([np.asarray(x, dtype=np.float32) for x in X_transit_ts]) if X_transit_ts else np.empty((0, *TS_TARGET_SHAPE), dtype=np.float32)
    X_rv = np.stack([np.asarray(x, dtype=np.float32) for x in X_rv]) if X_rv else np.empty((0, *RV_TARGET_SHAPE), dtype=np.float32)
    X_img = np.stack([np.asarray(x, dtype=np.float32) for x in X_img]) if X_img else np.empty((0, 2), dtype=np.float32)
    y_labels = np.asarray(y_labels, dtype=np.int32) if y_labels else np.empty((0,), dtype=np.int32)
    X_rv_mask = np.asarray(X_rv_mask, dtype=np.float32) if X_rv_mask else np.empty((0,), dtype=np.float32)
    X_img_mask = np.asarray(X_img_mask, dtype=np.float32) if X_img_mask else np.empty((0,), dtype=np.float32)

    # Standardize the data
    X_transit_img = standardize(X_transit_img)
    X_transit_ts = standardize(X_transit_ts)
    X_rv = standardize(X_rv)
    X_img = standardize(X_img)

    out = {
        "X_transit_image": X_transit_img,
        "X_transit_timeseries": X_transit_ts,
        "X_ttv": X_ttv,
        "y": y_labels,
    }
    if use_rv:
        out["X_rv"] = X_rv
        out["rv_mask"] = X_rv_mask
    if use_imaging:
        out["X_imaging"] = X_img
        out["imaging_mask"] = X_img_mask
    return out
