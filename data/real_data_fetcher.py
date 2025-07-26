# data/real_data_fetcher.py

import os
import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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

    # Process Confirmed Planets
    if confirmed_planet_dir and os.path.isdir(confirmed_planet_dir):
        logger.info(f"Scanning for confirmed planets in: {confirmed_planet_dir}")
        # Use glob to find all .fits files, including in subdirectories
        confirmed_files = glob.glob(os.path.join(confirmed_planet_dir, '**', '*.fits'), recursive=True)
        for file_path in confirmed_files:
            typed_light_curve_files.append({
                "file_path": str(Path(file_path).resolve()),
                "type": "confirmed_planet"
            })
        logger.info(f"Found {len(confirmed_files)} confirmed planet files.")
    else:
        logger.warning(f"Confirmed planets directory not found or not specified: {confirmed_planet_dir}")

    # Process False Positives
    if false_positive_dir and os.path.isdir(false_positive_dir):
        logger.info(f"Scanning for false positives in: {false_positive_dir}")
        fp_files = glob.glob(os.path.join(false_positive_dir, '**', '*.fits'), recursive=True)
        for file_path in fp_files:
            typed_light_curve_files.append({
                "file_path": str(Path(file_path).resolve()),
                "type": "false_positive"
            })
        logger.info(f"Found {len(fp_files)} false positive files.")
    else:
        logger.warning(f"False positives directory not found or not specified: {false_positive_dir}")

    return typed_light_curve_files