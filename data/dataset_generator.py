# data/dataset_generator.py (New Refactored Version)

import logging
import numpy as np
import pandas as pd  # <--- ADD THIS LINE
from imblearn.over_sampling import RandomOverSampler
from scipy import ndimage

logger = logging.getLogger(__name__)

def _augment_single_image_scipy(image):
    """
    Applies a random set of augmentations to a single 2D image using scipy.
    This function no longer depends on scikit-image.
    """
    if image is None:
        return None

    # Squeeze to 2D for scipy operations, but keep original shape info
    original_shape = image.shape
    image_2d = image.squeeze()
    
    augmented_image = image_2d.copy()

    # 1. Random Rotation
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        augmented_image = ndimage.rotate(augmented_image, angle, reshape=False, mode='nearest')

    # 2. Random Shear (Affine Transformation)
    if np.random.rand() > 0.5:
        shear_factor = np.random.uniform(-0.15, 0.15)
        transform_matrix = np.array([[1, shear_factor], [0, 1]])
        h, w = augmented_image.shape
        center_offset = 0.5 * np.array([h, w]) - (0.5 * np.array([h, w])).dot(transform_matrix)
        augmented_image = ndimage.affine_transform(
            augmented_image, transform_matrix.T, offset=center_offset, mode='nearest'
        )

    # 3. Random Noise Injection
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.03, augmented_image.shape)
        augmented_image += noise

    # 4. Random Horizontal Flip
    if np.random.rand() > 0.5:
        augmented_image = np.fliplr(augmented_image)

    return augmented_image.reshape(original_shape)

def augment_dataset(features, timeseries, labels, augmentation_factor=2, only_positive_class=True):
    """
    Augments a dataset of image features using the scipy-based augmentation.
    """
    if features is None or labels is None or len(features) == 0 or augmentation_factor <= 1:
        return features, timeseries, labels

    target_indices = np.where(labels == 1)[0] if only_positive_class else np.arange(len(features))
    if len(target_indices) == 0:
        logger.warning("No examples found for augmentation.")
        return features, timeseries, labels

    logger.info(f"Augmenting {len(target_indices)} examples with factor {augmentation_factor}.")
    
    new_features, new_timeseries, new_labels = [], [], []

    # Create n-1 new augmented copies for each target image
    for _ in range(augmentation_factor - 1):
        for idx in target_indices:
            augmented_image = _augment_single_image_scipy(features[idx])
            new_features.append(augmented_image)
            new_timeseries.append(timeseries[idx]) # Timeseries is not augmented in this version
            new_labels.append(labels[idx])

    if not new_features:
        return features, timeseries, labels

    # Combine original data with the new augmented data
    final_features = np.concatenate([features, np.array(new_features)], axis=0)
    final_timeseries = np.concatenate([timeseries, np.array(new_timeseries)], axis=0)
    final_labels = np.concatenate([labels, np.array(new_labels)], axis=0)

    # Shuffle the combined dataset
    shuffle_indices = np.random.permutation(len(final_features))
    logger.info(f"Dataset size after augmentation: {len(final_features)} samples.")
    
    return final_features[shuffle_indices], final_timeseries[shuffle_indices], final_labels[shuffle_indices]

def balance_dataset(features, timeseries, labels, method='oversample'):
    """
    Balances the dataset using the specified method from imblearn.
    """
    if len(np.unique(labels)) < 2:
        logger.warning("Dataset contains only one class. Skipping balancing.")
        return features, timeseries, labels

    logger.info(f"Attempting to balance dataset using '{method}' method.")
    original_shape = features.shape
    
    # Reshape image data for imblearn (2D)
    features_reshaped = features.reshape(len(features), -1)

    if method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    else:
        # Placeholder for other methods like SMOTE, RandomUnderSampler, etc.
        logger.warning(f"Balancing method '{method}' not fully implemented, using oversample.")
        sampler = RandomOverSampler(random_state=42)

    # We can't easily resample the timeseries data in the same way,
    # so we will apply the sampling based on the image features and apply
    # the same indices to the other arrays.
    _, y_resampled = sampler.fit_resample(features_reshaped, labels)
    
    # The `sample_indices_` attribute holds the indices of the samples selected
    indices = sampler.sample_indices_

    X_image_resampled = features[indices]
    X_timeseries_resampled = timeseries[indices]
    y_resampled_final = labels[indices]

    logger.info(f"Dataset after '{method}' balancing: {len(y_resampled_final)} samples. "
                f"Class distribution: {dict(pd.Series(y_resampled_final).value_counts())}")

    return X_image_resampled, X_timeseries_resampled, y_resampled_final