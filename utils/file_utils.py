"""
File utility functions for the exoplanet detection pipeline.
"""

import os
import logging
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


def ensure_directory(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def generate_file_hash(params):
    """
    Generate a consistent hash for query parameters to use as cache filename.
    
    Args:
        params: Dictionary of parameters to hash
    
    Returns:
        str: MD5 hash of the parameters
    """
    # Convert parameters to a sorted JSON string
    param_str = json.dumps(params, sort_keys=True)
    # Generate MD5 hash
    return hashlib.md5(param_str.encode()).hexdigest()


def save_pickle(data, file_path):
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        file_path: Path to save the pickle file
    """
    ensure_directory(os.path.dirname(file_path))
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Data saved to pickle file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {e}", exc_info=True)


def load_pickle(file_path):
    """
    Load data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        object: Loaded data or None if file not found or error
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Data loaded from pickle file: {file_path}")
            return data
        else:
            logger.warning(f"Pickle file not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}", exc_info=True)
        return None


def save_json(data, file_path, indent=2):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: Indentation level for JSON formatting
    """
    ensure_directory(os.path.dirname(file_path))
    try:
        # Handle non-serializable numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, cls=NumpyEncoder)
        logger.debug(f"Data saved to JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}", exc_info=True)


def load_json(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        dict: Loaded data or None if file not found or error
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Data loaded from JSON file: {file_path}")
            return data
        else:
            logger.warning(f"JSON file not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}", exc_info=True)
        return None


def load_fits_file(file_path):
    """
    Load data from a FITS file.
    
    Args:
        file_path: Path to the FITS file
    
    Returns:
        tuple: (time, flux, metadata) or (None, None, None) if error
    """
    try:
        with fits.open(file_path) as hdul:
            # Extract header information
            header = hdul[0].header
            
            # Extract data from the first extension (typically contains light curve)
            data = hdul[1].data
            
            # Extract time and flux arrays
            time = data['TIME']
            flux = data['PDCSAP_FLUX']
            
            # Filter out invalid values
            mask = np.isfinite(time) & np.isfinite(flux)
            time_filtered = time[mask]
            flux_filtered = flux[mask]
            
            # Create metadata dictionary
            metadata = {
                'file_path': file_path,
                'target_id': header.get('KEPLERID', header.get('TICID', 'Unknown')),
                'ra': header.get('RA', 0),
                'dec': header.get('DEC', 0),
                'n_points': len(time_filtered),
                'date_obs': header.get('DATE-OBS', 'Unknown'),
                'exptime': header.get('EXPTIME', 0)
            }
            
            return time_filtered, flux_filtered, metadata
    except Exception as e:
        logger.error(f"Error loading FITS file {file_path}: {e}", exc_info=True)
        return None, None, None


def save_dataframe(df, file_path, format='csv'):
    """
    Save a pandas DataFrame to file.
    
    Args:
        df: pandas DataFrame
        file_path: Path to save the file
        format: File format ('csv', 'xlsx', 'pkl')
    """
    ensure_directory(os.path.dirname(file_path))
    try:
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'xlsx':
            df.to_excel(file_path, index=False)
        elif format.lower() == 'pkl':
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"DataFrame saved to {format} file: {file_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {file_path}: {e}", exc_info=True)


def load_dataframe(file_path, format=None):
    """
    Load a pandas DataFrame from file.
    
    Args:
        file_path: Path to the file
        format: File format (if None, inferred from extension)
    
    Returns:
        pandas.DataFrame: Loaded DataFrame or None if error
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        # Infer format from file extension if not specified
        if format is None:
            _, ext = os.path.splitext(file_path)
            format = ext.lstrip('.').lower()
        
        if format == 'csv':
            df = pd.read_csv(file_path)
        elif format in ('xlsx', 'xls'):
            df = pd.read_excel(file_path)
        elif format == 'pkl':
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"DataFrame loaded from {format} file: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {file_path}: {e}", exc_info=True)
        return None


def save_model_weights(model, file_name, output_dir=None):
    if output_dir is None:
        output_dir = config.MODEL_DIR
    
    file_path = os.path.join(output_dir, file_name)
    
    try:
        ensure_directory(os.path.dirname(file_path))
        model.save_weights(file_path)
        logger.info(f"Model weights saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving model weights to {file_path}: {e}", exc_info=True)


def load_model_weights(model, file_path):
    """
    Load weights into a Keras model.
    
    Args:
        model: Keras model
        file_path: Path to the weights file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if os.path.exists(file_path):
            model.load_weights(file_path)
            logger.info(f"Model weights loaded from: {file_path}")
            return True
        else:
            logger.warning(f"Weights file not found: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error loading model weights from {file_path}: {e}", exc_info=True)
        return False


def list_files_with_extension(directory, extension):
    """
    List all files in a directory with the specified extension.
    
    Args:
        directory: Directory to search
        extension: File extension (e.g., '.fits', '.csv')
    
    Returns:
        list: List of file paths
    """
    extension = extension.lower()
    if not extension.startswith('.'):
        extension = '.' + extension
    
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return []
        
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extension):
                    file_list.append(os.path.join(root, file))
        
        logger.debug(f"Found {len(file_list)} {extension} files in {directory}")
        return file_list
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}", exc_info=True)
        return []
