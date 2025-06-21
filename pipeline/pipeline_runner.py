# FILE: pipeline/pipeline_runner.py (Refactored)
import logging
from .pipeline_core import (
    setup_logging,
    process_light_curve,
    prepare_datasets,
    prepare_train_test_split
)

logger = logging.getLogger(__name__)

# This placeholder function is kept for compatibility with main.py,
# but the main logic now resides in enhanced_pipeline_runner.
def run_pipeline(light_curve_files, exoplanet_labels, **kwargs):
    logger.info("Running the basic pipeline...")
    processed_results = []
    for file_path in light_curve_files:
        result = process_light_curve(file_path, kwargs.get('output_dir_base'), "unknown")
        if result:
            processed_results.append(result)
    return {"processed_results": processed_results}

