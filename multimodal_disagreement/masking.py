import numpy as np

def mask_temporal(X_multimodal):
    """
    Applies structured masking to isolate the spatial modality
    by zeroing out the temporal (1D) input branch.
    
    Args:
        X_multimodal: List of [X_spatial, X_temporal, X_features]
    Returns:
        List with X_temporal zeroed out.
    """
    X_spatial, X_temporal, X_features = X_multimodal
    return [X_spatial, np.zeros_like(X_temporal), np.zeros_like(X_features)]

def mask_spatial(X_multimodal):
    """
    Applies structured masking to isolate the temporal modality
    by zeroing out the spatial (2D) input branch.
    
    Args:
        X_multimodal: List of [X_spatial, X_temporal, X_features]
    Returns:
        List with X_spatial zeroed out.
    """
    X_spatial, X_temporal, X_features = X_multimodal
    return [np.zeros_like(X_spatial), X_temporal, np.zeros_like(X_features)]
