# /data/enhanced_augmentation.py (New Refactored Version)

import logging
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)

def _augment_single_image(image):
    """
    Applies a random set of augmentations to a single 2D image using scipy.

    Args:
        image (np.ndarray): A single 2D image (e.g., 64x64).

    Returns:
        np.ndarray: The augmented 2D image.
    """
    # Ensure image is 2D before processing
    if image.ndim > 2:
        image = image.squeeze()

    augmented_image = image.copy()

    # 1. Random Rotation
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        augmented_image = ndimage.rotate(augmented_image, angle, reshape=False, mode='nearest')

    # 2. Random Shear (Affine Transformation)
    if np.random.rand() > 0.5:
        shear_factor = np.random.uniform(-0.15, 0.15)
        # Create an affine transformation matrix for shearing along one axis
        transform_matrix = np.array([[1, shear_factor], [0, 1]])
        
        # Apply the transformation
        h, w = augmented_image.shape
        center_offset = 0.5 * np.array([h, w]) - (0.5 * np.array([h, w])).dot(transform_matrix)
        
        augmented_image = ndimage.affine_transform(
            augmented_image,
            transform_matrix.T, # Scipy expects the inverse transformation matrix
            offset=center_offset,
            mode='nearest'
        )

    # 3. Random Noise Injection
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.03, augmented_image.shape)
        augmented_image += noise

    # 4. Random Horizontal Flip
    if np.random.rand() > 0.5:
        augmented_image = np.fliplr(augmented_image)

    return augmented_image


def augment_dataset(features, labels, augmentation_factor=2, only_positive_class=True):
    """
    Augments a dataset of image features.

    Args:
        features (np.ndarray): The feature data (images).
        labels (np.ndarray): The corresponding labels.
        augmentation_factor (int): The number of augmented versions to generate for each eligible image.
                                   A factor of 2 means one new augmented image is created per original.
        only_positive_class (bool): If True, only augment the minority class (assumed to be label 1).

    Returns:
        tuple[np.ndarray, np.ndarray]: The augmented features and labels.
    """
    if features is None or labels is None or len(features) == 0:
        logger.warning("Cannot augment an empty or invalid dataset.")
        return features, labels

    if augmentation_factor <= 1:
        logger.info("Augmentation factor is 1 or less, returning original dataset.")
        return features, labels

    # If we are only augmenting the positive class, find those indices
    if only_positive_class:
        target_indices = np.where(labels == 1)[0]
        if len(target_indices) == 0:
            logger.warning("No positive examples (label=1) found to augment.")
            return features, labels
        logger.info(f"Augmenting {len(target_indices)} positive examples with factor {augmentation_factor}.")
    else:
        # Augment all examples
        target_indices = np.arange(len(features))
        logger.info(f"Augmenting all {len(target_indices)} examples with factor {augmentation_factor}.")

    augmented_features_list = []
    augmented_labels_list = []

    # Loop to create augmented copies
    for _ in range(augmentation_factor - 1): # Loop n-1 times to create n-1 new copies
        for idx in target_indices:
            original_image = features[idx]
            original_label = labels[idx]

            augmented_image = _augment_single_image(original_image)
            
            # Ensure the output shape matches the input (e.g., with channel dimension)
            if original_image.ndim == 3 and augmented_image.ndim == 2:
                 augmented_image = np.expand_dims(augmented_image, axis=-1)

            augmented_features_list.append(augmented_image)
            augmented_labels_list.append(original_label)

    if not augmented_features_list:
        logger.warning("No new images were generated during augmentation.")
        return features, labels

    # Combine original data with the new augmented data
    final_features = np.concatenate([features, np.array(augmented_features_list)], axis=0)
    final_labels = np.concatenate([labels, np.array(augmented_labels_list)], axis=0)

    # Shuffle the combined dataset to mix original and augmented data
    shuffle_indices = np.random.permutation(len(final_features))
    
    logger.info(f"Dataset size after augmentation: {len(final_features)} samples.")
    return final_features[shuffle_indices], final_labels[shuffle_indices]