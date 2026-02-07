import torch
import torch.nn as nn
from cacl_utils import MultiViewModel, _to1d, _upper_triangle_argmax, _avg_partition_embeddings, _get_partition_embeddings, explain_record, create_feature_partitions
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class CACLFeatureExtractor(nn.Module):
    def __init__(self, model_path: str, partitions: List[List[int]], 
                 input_dim: int, output_dim: int, proj_dim: int, 
                 nhead: int, num_layers: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.partitions = partitions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate MultiViewModel with dummy partitions (will be replaced by loaded state_dict)
        # The partitions are mainly used to define the structure of the encoders and projectors
        # when creating the MultiViewModel. When loading a state_dict, the structure must match.
        # However, for pure inference (feature extraction), we only care about the encoders.
        
        # A more robust way would be to pass the actual partitions used during pre-training
        # or infer them if possible. For now, we assume the provided partitions match.
        
        self.cacl_model = MultiViewModel(
            partitions=partitions,
            encoder_type="transformer",
            output_dim=output_dim,
            proj_dim=proj_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        # Load the pre-trained state dictionary
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # The state_dict from cacl_pretrain saves the entire model (encoders + projectors)
            # If we only want the encoders, we need to filter the state_dict or load into a specific part.
            # For now, let's load the full state_dict and then use model.encoders for feature extraction.
            self.cacl_model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded CACL model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load CACL model from {model_path}: {e}")
            raise

        self.cacl_model.eval() # Set to evaluation mode

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extracts embeddings for each partition.
        Input x should be a single light curve (1, input_dim).
        """
        # Split the input light curve into views based on partitions
        views = []
        for p in self.partitions:
            views.append(x[:, p].to(self.device))
        
        with torch.no_grad():
            # The MultiViewModel's forward returns a list of projected embeddings
            # For feature extraction, we want the output of the encoders.
            # We need to adapt this, or consider the projected embeddings as features.
            # Let's return the projected embeddings for now, as they are normalized and suitable.
            
            # If we wanted raw encoder outputs:
            # encoded_outputs = []
            # for i, view in enumerate(views):
            #     encoded_outputs.append(self.cacl_model.encoders[i](view))
            # return encoded_outputs
            
            return self.cacl_model(views) # This returns projected embeddings

def explain_with_cacl(
    data_sample: np.ndarray,
    cacl_explainer: CACLFeatureExtractor,
    dependency_matrix: Optional[np.ndarray] = None,
    context_avg_embeddings: Optional[np.ndarray] = None,
    context_indices: Optional[List[int]] = None,
    eta: float = 0.2,
    context_threshold_quantile: Optional[float] = None,
    ref_ctx_sim_distribution: Optional[np.ndarray] = None,
) -> Dict:
    """
    Computes CACL-based explanation for a single data sample.
    """
    # Ensure data_sample is 2D (batch_size, input_dim) for model input
    if data_sample.ndim == 1:
        data_sample = np.expand_dims(data_sample, axis=0)
    
    # Get projected embeddings for the data sample
    # The cacl_explainer's forward method returns a list of projected embeddings (one for each partition)
    record_projected_embeddings = cacl_explainer(torch.from_numpy(data_sample).float())
    
    # Convert list of torch tensors to list of numpy arrays for explain_record
    record_projected_embeddings_np = [emb.cpu().numpy().flatten() for emb in record_projected_embeddings]
    
    return explain_record(
        record_embeddings=record_projected_embeddings_np,
        context_avg_embeddings=context_avg_embeddings,
        context_indices=context_indices,
        eta=eta,
        dependency_matrix=dependency_matrix,
        context_threshold_quantile=context_threshold_quantile,
        ref_ctx_sim_distribution=ref_ctx_sim_distribution,
    )

