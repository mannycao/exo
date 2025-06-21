# FILE: pipeline/enhanced_pipeline_runner.py (Refactored)

import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count

import config
from data.dataset_generator import balance_dataset, augment_dataset
from models.model_trainer import train_enhanced_model
# --- THIS IS THE KEY FIX ---
# Import from the new core file, not from itself or the other runner
from .pipeline_core import (
    process_light_curve,
    prepare_datasets,
    prepare_train_test_split
)
# --- END OF FIX ---
from .report_generator import generate_report
from utils.visualization import visualize_detection_results
from data.light_curve_processor import reshape_data_for_cnn
from models.cnn_model import build_transit_detection_model

logger = logging.getLogger(__name__)

# ... (The rest of the file remains the same as the last version I provided)
# The `run_enhanced_pipeline` and `train_ai_models_enhanced` functions
# from "runner_final_fix" are correct. Just ensure the imports at the top
# of the file match what is shown here.
def run_enhanced_pipeline(light_curve_files_with_types, exoplanet_labels, **kwargs):
    """
    Main orchestrator for the enhanced pipeline, now using a more robust
    multiprocessing pool to prevent hangs.
    """
    output_dir_base = kwargs.get('output_dir_base')
    max_workers = kwargs.get('max_workers', cpu_count())
    
    logger.info(f"Running enhanced pipeline. Results will be saved to: {output_dir_base}")
    processed_files_artifacts_dir = output_dir_base / "processed_artifacts"
    processed_files_artifacts_dir.mkdir(parents=True, exist_ok=True)

    process_args = []
    for file_path, file_type in light_curve_files_with_types:
        file_specific_result_dir = processed_files_artifacts_dir / Path(file_path).stem
        process_args.append((file_path, str(file_specific_result_dir), file_type))

    processed_results = []
    logger.info(f"Processing {len(process_args)} files using a pool of {max_workers} workers...")
    
    with Pool(processes=max_workers) as pool:
        results_from_pool = pool.starmap(process_light_curve, process_args)

    for i, res in enumerate(results_from_pool):
        if res and res.get('success'):
            res['file_type_source'] = light_curve_files_with_types[i][1]
            processed_results.append(res)
        else:
            logger.warning(f"File processing failed or returned no result for: {light_curve_files_with_types[i][0]}")

    if not processed_results:
        logger.error("No files were successfully processed. Halting pipeline.")
        return {}

    X_image, X_timeseries, y_labels = prepare_datasets(processed_results, exoplanet_labels)
    X_bal, X_ts_bal, y_bal = balance_dataset(X_image, X_timeseries, y_labels)
    X_aug, X_ts_aug, y_aug = augment_dataset(X_bal, X_ts_bal, y_bal, augmentation_factor=kwargs.get('augmentation_factor', 1))

    model_training_results = {}
    if X_aug is not None and len(X_aug) > 0:
        # ... The rest of the model training and reporting logic ...
        pass
    
    logger.info("Enhanced pipeline execution completed.")
    return {"successfully_processed_count": len(processed_results)} # simplified return
