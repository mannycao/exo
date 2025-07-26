# data/feature_extractor.py

import logging
import numpy as np
from astropy.timeseries import BoxLeastSquares

logger = logging.getLogger(__name__)

def detect_and_extract_features(time, flux, config):
    """
    Final version of the feature extractor using a highly sensitive BLS implementation.
    It extracts features for the most significant periodic signal found in the light curve.
    """
    if len(flux) < 100:
        return []

    time_span = time[-1] - time[0]
    durations = np.linspace(0.05, 0.3, 10)
    min_period = max(durations.max() * 2, 0.5)
    max_period = time_span / 2

    if min_period >= max_period:
        return []

    bls = BoxLeastSquares(t=time, y=flux)
    periods = np.logspace(np.log10(min_period), np.log10(max_period), 10000)
    
    try:
        power_results = bls.power(periods, durations)
    except Exception:
        return []

    if power_results.power.max() < config.BLS_POWER_THRESHOLD:
        return []

    index = np.argmax(power_results.power)
    period = power_results.period[index]
    t0 = power_results.transit_time[index]
    duration = power_results.duration[index]
    depth = power_results.depth[index]
    
    transit_times = t0 + np.arange(0, time.max(), period)
    noise = np.std(flux)
    snr = depth / noise if noise > 0 else 0
    
    depths = []
    for transit_time in transit_times:
        idx = np.argmin(np.abs(time - transit_time))
        window = int(duration / (np.median(np.diff(time)) * 24)) * 2
        start, end = max(0, idx - window), min(len(flux), idx + window)
        if end > start:
            depths.append(flux[start:end].min())
    
    asymmetry = np.std(depths) / np.mean(depths) if len(depths) > 1 and np.mean(depths) != 0 else 0

    return [{
        'depth': depth,
        'duration_hours': duration * 24,
        'snr': snr,
        'period_days': period,
        'asymmetry': asymmetry,
        'num_transits': len(transit_times)
    }]