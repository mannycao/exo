# data/real_data_fetcher.py

import os
import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def _scan_directory_for_fits(directory_path, file_type):
    """Helper function to scan a directory for FITS files and assign a type."""
    files_found = []
    if directory_path and os.path.isdir(directory_path):
        logger.info(f"Scanning for {file_type} in: {directory_path}")
        fits_files = glob.glob(os.path.join(directory_path, '**', '*.fits'), recursive=True)
        logger.debug(f"Found {len(fits_files)} .fits files in {directory_path} for type {file_type}.") # Added debug log
        for file_path in fits_files:
            files_found.append({
                "file_path": str(Path(file_path).resolve()),
                "type": file_type
            })
        logger.info(f"Found {len(fits_files)} {file_type} files.")
    else:
        logger.warning(f"{file_type} directory not found or not specified: {directory_path}")
    return files_found

def smart_data_fetcher(confirmed_planet_dir, false_positive_dir):
    """
    Scans local directories for FITS files and labels them as confirmed
    planets or false positives.

    Args:
        confirmed_planet_dir (str): Path to the directory with confirmed planet FITS files.
        false_positive_dir (str): Path to the directory with false positive FITS files.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              'file_path' and its 'type' ('confirmed_planet' or 'false_positive').
    """
    typed_light_curve_files = []

    typed_light_curve_files.extend(_scan_directory_for_fits(confirmed_planet_dir, "confirmed_planet"))
    typed_light_curve_files.extend(_scan_directory_for_fits(false_positive_dir, "false_positive"))

    logger.info(f"Total files found by smart_data_fetcher: {len(typed_light_curve_files)}") # Added debug log

    return typed_light_curve_files