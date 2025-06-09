"""
Functions for data balancing and augmentation to improve model training.
Uses the 'imbalanced-learn' and 'scikit-image' libraries.
"""
import logging
import skimage
import numpy as np

# Use a try-except block for imblearn import as it might not be installed
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logging.error(
        "imbalanced-learn library not found. Data balancing will be skipped. "
        "Please install it (`pip install imbalanced-learn`)."
    )

# Use a try-except block for scikit-image import
try:
    from skimage.transform import shift
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.error(
        "scikit-image library not found. Image augmentation will be skipped. "
        "Please install it (`pip install scikit-image`)."
    )


logger = logging.getLogger(__name__)

def balance_dataset(X_image, X_timeseries, y, method='oversample'):
    """
    Balances the dataset using specified method from the imbalanced-learn library.
    """
    if not IMBLEARN_AVAILABLE:
        logger.warning("imbalanced-learn not available. Skipping dataset balancing.")
        return X_image, X_timeseries, y

    logger.info(f"Attempting to balance dataset using '{method}' method.")
    
    n_samples, height, width = X_image.shape
    X_image_reshaped = X_image.reshape(n_samples, -1)
    
    if X_timeseries is not None and len(X_timeseries) > 0:
        X_timeseries_reshaped = X_timeseries.reshape(n_samples, -1) if X_timeseries.ndim > 1 else X_timeseries.reshape(-1, 1)
        X_combined = np.concatenate([X_image_reshaped, X_timeseries_reshaped], axis=1)
    else:
        X_combined = X_image_reshaped

    sampler = None
    if method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        logger.warning(f"Unknown balancing method: '{method}'. Returning original dataset.")
        return X_image, X_timeseries, y

    try:
        X_resampled, y_resampled = sampler.fit_resample(X_combined, y)
    except ValueError as e:
        logger.error(f"Error during resampling with method '{method}': {e}. Returning original dataset.")
        return X_image, X_timeseries, y
        
    image_feature_len = height * width
    X_image_resampled_flat = X_resampled[:, :image_feature_len]
    X_image_resampled = X_image_resampled_flat.reshape(-1, height, width)
    
    X_timeseries_resampled = None
    if X_timeseries is not None and len(X_timeseries) > 0:
        X_timeseries_resampled = X_resampled[:, image_feature_len:]
    
    class_dist = dict(zip(*np.unique(y_resampled, return_counts=True)))
    logger.info(f"Dataset after '{method}' balancing: {len(y_resampled)} samples. Class distribution: {class_dist}")
    
    return X_image_resampled, X_timeseries_resampled, y_resampled


def _augment_single_timeseries(segment):
    """Applies simple augmentations to a single timeseries segment."""
    noise = np.random.normal(0, 0.005 * np.std(segment), segment.shape) 
    augmented_segment = segment + noise
    augmented_segment *= np.random.uniform(0.98, 1.02)
    return augmented_segment

def _augment_single_image(image):
    """Applies simple augmentations to a single image."""
    if not SKIMAGE_AVAILABLE: return image 
    noise = np.random.normal(0, 0.005 * np.std(image), image.shape)
    augmented_image = image + noise
    h_shift, w_shift = np.random.uniform(-1.5, 1.5, 2)
    augmented_image = shift(augmented_image, (h_shift, w_shift), mode='reflect')
    return augmented_image


def augment_dataset(X_image, X_timeseries, y, augmentation_factor=2, only_positive_class=False):
    """
    Augments the dataset by creating modified copies of samples.
    """
    if not SKIMAGE_AVAILABLE:
        logger.warning("scikit-image not available. Skipping dataset augmentation.")
        return X_image, X_timeseries, y

    if augmentation_factor <= 1:
        logger.info("Augmentation factor is 1 or less, no augmentation performed.")
        return X_image, X_timeseries, y
        
    logger.info(f"Augmenting dataset. Each selected sample will have a total of {augmentation_factor} versions.")

    augmented_images = list(X_image)
    augmented_timeseries = list(X_timeseries) if X_timeseries is not None and len(X_timeseries) > 0 else []
    augmented_labels = list(y)

    indices_to_augment = np.arange(len(y))
    if only_positive_class:
        indices_to_augment = np.where(y == 1)[0]
    
    logger.info(f"Creating {augmentation_factor - 1} new versions for {len(indices_to_augment)} samples.")

    for i in indices_to_augment:
        for _ in range(augmentation_factor - 1): 
            aug_img = _augment_single_image(X_image[i])
            augmented_images.append(aug_img)

            if X_timeseries is not None and len(X_timeseries) > 0:
                aug_ts = _augment_single_timeseries(X_timeseries[i])
                augmented_timeseries.append(aug_ts)
            
            augmented_labels.append(y[i])

    final_X_image = np.array(augmented_images)
    final_y = np.array(augmented_labels)
    final_X_timeseries = np.array(augmented_timeseries) if X_timeseries is not None and len(X_timeseries) > 0 else None

    class_dist = dict(zip(*np.unique(final_y, return_counts=True)))
    logger.info(f"Dataset after augmentation: {len(final_y)} examples. Class distribution: {class_dist}")

    return final_X_image, final_X_timeseries, final_y
