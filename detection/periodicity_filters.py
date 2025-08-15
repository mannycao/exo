"""
Filtering and validation functions for periodicity analysis results.
"""

import logging
import numpy as np
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

def validate_period_detection(
    periodogram_data: Dict,
    min_confidence: float = 0.5,
    min_power_snr: float = 2.0,
    max_fap: float = 0.01,
    data_span: Optional[float] = None
) -> Tuple[bool, str, float]:
    """
    Validate a period detection using multiple criteria.
    
    Args:
        periodogram_data: Dictionary containing periodicity analysis results
        min_confidence: Minimum required confidence score (0-1)
        min_power_snr: Minimum required power SNR
        max_fap: Maximum allowed false alarm probability
        data_span: Total time span of the data (optional)
    
    Returns:
        Tuple of (is_valid: bool, reason: str, adjusted_confidence: float)
    """
    if not periodogram_data or 'best_period' not in periodogram_data:
        return False, "Missing periodogram data", 0.0
    
    period = periodogram_data['best_period']
    confidence = periodogram_data['confidence_score']
    fap = periodogram_data['fap']
    power_snr = periodogram_data['power_snr']
    
    # Initialize with base confidence
    adjusted_confidence = confidence
    
    # Check basic confidence threshold
    if confidence < min_confidence:
        return False, f"Low confidence: {confidence:.2f} < {min_confidence}", confidence
    
    # Check false alarm probability
    if fap > max_fap:
        adjusted_confidence *= 0.8  # Reduce confidence
        if fap > max_fap * 2:
            return False, f"High false alarm probability: {fap:.3f} > {max_fap}", adjusted_confidence
    
    # Check power SNR
    if power_snr < min_power_snr:
        adjusted_confidence *= 0.9  # Reduce confidence
        if power_snr < min_power_snr / 2:
            return False, f"Low power SNR: {power_snr:.1f} < {min_power_snr}", adjusted_confidence
    
    # Check data span if provided
    if data_span is not None:
        if period > data_span / 2:
            adjusted_confidence *= 0.7  # Significantly reduce confidence
            return False, f"Period ({period:.1f} days) exceeds half the data span ({data_span/2:.1f} days)", adjusted_confidence
    
    return True, "Valid period detection", adjusted_confidence


def filter_periodicity_results(
    results: List[Dict],
    min_confidence: float = 0.5,
    min_power_snr: float = 2.0,
    max_fap: float = 0.01,
    require_multiple_transits: bool = True
) -> List[Dict]:
    """
    Filter a list of periodicity results based on quality criteria.
    
    Args:
        results: List of dictionaries containing periodicity analysis results
        min_confidence: Minimum required confidence score
        min_power_snr: Minimum required power SNR
        max_fap: Maximum allowed false alarm probability
        require_multiple_transits: If True, require at least 3 transits
    
    Returns:
        List of filtered results with added validation information
    """
    filtered_results = []
    
    for result in results:
        if not result or 'periodogram' not in result:
            continue
        
        periodogram_data = result['periodogram']
        
        # Check number of transits if required
        if require_multiple_transits and len(result.get('transit_times', [])) < 3:
            continue
        
        # Get data span if available
        data_span = None
        transit_times = result.get('transit_times')
        if transit_times and len(transit_times) > 1:
            data_span = max(transit_times) - min(transit_times)
        
        # Validate the period detection
        is_valid, reason, adjusted_confidence = validate_period_detection(
            periodogram_data,
            min_confidence=min_confidence,
            min_power_snr=min_power_snr,
            max_fap=max_fap,
            data_span=data_span
        )
        
        # Add validation results to the result dictionary
        result['period_validation'] = {
            'is_valid': is_valid,
            'reason': reason,
            'adjusted_confidence': adjusted_confidence,
            'original_confidence': periodogram_data['confidence_score']
        }
        
        if is_valid:
            filtered_results.append(result)
        else:
            logger.debug(f"Filtered out result: {reason}")
    
    return filtered_results


def get_period_quality_metrics(periodogram_data: Dict) -> Dict:
    """
    Calculate additional quality metrics for a period detection.
    
    Args:
        periodogram_data: Dictionary containing periodicity analysis results
    
    Returns:
        Dictionary of quality metrics
    """
    if not periodogram_data:
        return {}
    
    metrics = {
        'confidence_score': periodogram_data.get('confidence_score', 0.0),
        'fap': periodogram_data.get('fap', 1.0),
        'power_snr': periodogram_data.get('power_snr', 0.0),
        'peak_snr': periodogram_data.get('peak_snr', 0.0),
    }
    
    # Add derived metrics
    metrics['fap_confidence'] = 1.0 - metrics['fap']
    metrics['normalized_power_snr'] = min(1.0, metrics['power_snr'] / 10.0)
    metrics['normalized_peak_snr'] = min(1.0, metrics['peak_snr'] / 5.0)
    
    # Calculate overall quality score (weighted average)
    weights = {
        'fap_confidence': 0.4,
        'normalized_power_snr': 0.3,
        'normalized_peak_snr': 0.3
    }
    
    metrics['quality_score'] = sum(
        metrics[key] * weight 
        for key, weight in weights.items()
    )
    
    return metrics
