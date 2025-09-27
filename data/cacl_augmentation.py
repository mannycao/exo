import numpy as np

def add_noise(light_curve, noise_level=0.01):
    """Adds Gaussian noise to a light curve."""
    return light_curve + np.random.normal(0, noise_level, light_curve.shape)

def time_jitter(light_curve, max_jitter=5):
    """
    Applies time jitter by shifting the light curve and interpolating.
    Assumes light_curve is a 1D array representing flux over time.
    """
    if light_curve.ndim != 1:
        raise ValueError("Light curve must be a 1D array for time jitter.")

    length = len(light_curve)
    shift = np.random.randint(-max_jitter, max_jitter + 1)
    
    if shift == 0:
        return light_curve

    augmented_light_curve = np.zeros_like(light_curve, dtype=float)
    
    # Shift and fill with original values or a sensible default (e.g., mean)
    if shift > 0: # Shift right
        augmented_light_curve[shift:] = light_curve[:-shift]
        augmented_light_curve[:shift] = np.mean(light_curve) # Fill beginning with mean
    else: # Shift left
        augmented_light_curve[:shift] = light_curve[-shift:]
        augmented_light_curve[shift:] = np.mean(light_curve) # Fill end with mean
        
    return augmented_light_curve

def scale_light_curve(light_curve, min_scale=0.9, max_scale=1.1):
    """Scales the flux values of a light curve randomly."""
    scale_factor = np.random.uniform(min_scale, max_scale)
    return light_curve * scale_factor

def generate_contrastive_pair(light_curve, noise_level=0.01, max_jitter=5, min_scale=0.9, max_scale=1.1):
    """Generates two distinct augmented versions of a single light curve for contrastive learning."""
    # Ensure light_curve is a numpy array
    light_curve = np.asarray(light_curve)

    # Augmentation 1: Noise + Scaling
    aug1 = add_noise(light_curve, noise_level=noise_level)
    aug1 = scale_light_curve(aug1, min_scale=min_scale, max_scale=max_scale)

    # Augmentation 2: Time Jitter + Noise
    aug2 = time_jitter(light_curve, max_jitter=max_jitter)
    aug2 = add_noise(aug2, noise_level=noise_level)
    
    return aug1, aug2