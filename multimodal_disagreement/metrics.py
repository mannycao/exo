import numpy as np

def disagreement(p_temporal, p_spatial):
    """
    Computes the inter-modal disagreement metric delta.
    delta = |P_temporal - P_spatial|
    """
    return abs(p_temporal - p_spatial)

def apply_veto(delta, threshold=0.3):
    """
    Applies the Disagreement Veto threshold to filter out false-positive signals.
    """
    return delta > threshold
