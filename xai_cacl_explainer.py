import numpy as np
import torch
import tensorflow as tf # Keep for potential future Keras integration
from tensorflow import keras # Keep for potential future Keras integration
import os
from typing import Dict, List, Tuple, Optional
import logging
import math
import torch.nn as nn
import torch.nn.functional as F
import config # Import config

# Import necessary functions from cacl_utils.py
from cacl_utils import (
    create_feature_partitions,
    build_context_groups,
    _upper_triangle_argmax, # Not directly used here, but good to have for context
    compute_dependency_matrix,
    compute_context_embeddings,
    explain_record,
    MultiViewModel as PyTorchMultiViewModel # Alias to avoid conflict
)
# from convert_cacl_to_keras import KerasTransformerEncoder # Not using Keras CACL directly for now

logger = logging.getLogger(__name__)

class CACLFeatureExtractor:
    def __init__(self,
                 model_path: str,
                 partitions: List[List[int]],
                 input_dim: int, # Total input dimension for the CACL model (e.g., light curve length)
                 output_dim: int = 64, # Output dimension of each encoder
                 proj_dim: int = 32, # Projection dimension
                 nhead: int = 2,
                 num_layers: int = 1):
        self.partitions = partitions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj_dim = proj_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        logger.info(f"CACLFeatureExtractor initialized. Using device: {self.device}")

    def _load_model(self, model_path: str):
        """
        Loads the pre-trained PyTorch MultiViewModel.
        """
        if not os.path.exists(model_path):
            logger.error(f"CACL model not found at {model_path}. Please pre-train the CACL model first.")
            raise FileNotFoundError(f"CACL model not found at {model_path}")

        model = PyTorchMultiViewModel(
            partitions=self.partitions,
            encoder_type="transformer", # Assuming transformer encoder from cacl_pretrain.py
            output_dim=self.output_dim,
            proj_dim=self.proj_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=config.CACL_DROPOUT # Added dropout parameter
        )
        logger.debug(f"PyTorchMultiViewModel constructed with num_layers: {self.num_layers}")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval() # Set to evaluation mode
        logger.info(f"CACL PyTorch MultiViewModel loaded from {model_path}")
        return model

    def get_partition_embeddings(self, data_sample: np.ndarray) -> List[np.ndarray]:
        """
        Generates CACL partition embeddings for a single data sample (e.g., time-series).
        The data_sample is split into views based on self.partitions, and each view
        is passed through the corresponding encoder and projector.
        """
        if len(data_sample) != self.input_dim:
            raise ValueError(f"Data sample dimension ({len(data_sample)}) does not match "
                             f"expected input_dim ({self.input_dim}) for CACL model.")

        views = []
        for part in self.partitions:
            # Ensure data_sample is 1D and slice correctly
            view_data = data_sample[part]
            views.append(torch.tensor(view_data, dtype=torch.float32).unsqueeze(0).to(self.device))
        
        with torch.no_grad():
            embeddings = self.model(views) # List of (1, proj_dim) tensors
        
        return [e.squeeze(0).cpu().numpy() for e in embeddings] # List of (proj_dim,) numpy arrays

def explain_with_cacl(
    data_sample: np.ndarray, # Assumed to be time-series data for now
    cacl_explainer: CACLFeatureExtractor,
    dependency_matrix: np.ndarray,
    context_avg_embeddings: Optional[np.ndarray] = None,
    context_indices: Optional[List[int]] = None,
    eta: float = 0.2,
    context_threshold_quantile: Optional[float] = None,
    ref_ctx_sim_distribution: Optional[np.ndarray] = None,
) -> Dict:
    """
    Generates a CACL-based explanation for a single data sample.
    
    Args:
        data_sample: A 1D numpy array representing the time-series data for a single sample.
        cacl_explainer: An initialized CACLFeatureExtractor instance.
        dependency_matrix: The pre-computed dependency matrix from CACL training data.
        context_avg_embeddings: Average embeddings of other records for context alignment.
        context_indices: Indices of context neighbors for the current record.
        eta: Threshold for semantic agreement.
        context_threshold_quantile: Quantile threshold for context similarity.
        ref_ctx_sim_distribution: Distribution of context similarity scores from inliers.

    Returns:
        A dictionary containing the CACL explanation.
    """
    record_embeddings = cacl_explainer.get_partition_embeddings(data_sample)
    
    explanation = explain_record(
        record_embeddings=record_embeddings,
        context_avg_embeddings=context_avg_embeddings,
        context_indices=context_indices,
        eta=eta,
        dependency_matrix=dependency_matrix,
        context_threshold_quantile=context_threshold_quantile,
        ref_ctx_sim_distribution=ref_ctx_sim_distribution,
    )
    return explanation
