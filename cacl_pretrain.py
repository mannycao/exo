import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import logging

from cacl_utils import MultiViewModel, contrastive_loss_ctx, create_feature_partitions, build_context_groups
from data.dataset_generator import create_contrastive_dataset, LightCurveContrastiveDataset
import glob

# from data.data_fetcher import LightCurveFetcher # Removed as LightCurveFetcher is not defined

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cacl_pretrain(
    data_dir: str = "./data",
    output_model_path: str = "./models/cacl_transformer_feature_extractor.pt",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4, # Added for L2 regularization
    dropout: float = 0.1, # Added for regularization
    K_partitions: int = 4, # Number of feature partitions
    k_context: int = 5,    # Number of context neighbors
    data_type: str = "temporal", # For build_context_groups
    output_dim: int = 64,  # Output dimension of the encoder
    proj_dim: int = 32,    # Projection dimension
    nhead: int = 2,        # Transformer heads
    num_layers: int = 1,   # Transformer layers
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Data Preparation
    logger.info("Fetching light curve paths...")
    unlabeled_file_paths = glob.glob(os.path.join(data_dir, "*.fits"))
    if not unlabeled_file_paths:
        logger.warning(f"No FITS files found in {data_dir}. Generating sample light curves for demonstration.")
        from data.data_fetcher import generate_sample_light_curves
        generate_sample_light_curves(count=100, output_dir=data_dir)
        unlabeled_file_paths = glob.glob(os.path.join(data_dir, "*.fits"))
        if not unlabeled_file_paths:
            logger.error("Failed to generate sample light curves or find them after generation. Exiting.")
            return

    # If actual data fetching is used, ensure create_contrastive_dataset is run first
    # For now, we assume X_aug1.npy and X_aug2.npy exist in data_dir
    if not os.path.exists(os.path.join(data_dir, 'X_aug1.npy')) or \
       not os.path.exists(os.path.join(data_dir, 'X_aug2.npy')):
        logger.info("Running create_contrastive_dataset to generate augmented data...")
        # This call needs actual file paths if not using dummy data
        create_contrastive_dataset(unlabeled_file_paths, data_dir)

    dataset = LightCurveContrastiveDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Loaded contrastive dataset with {len(dataset)} samples.")

    # Determine input_dim for partitions (assuming all light curves have the same length)
    sample_light_curve_length = dataset.X_aug1.shape[1]
    # For light curves, we might treat the entire light curve as a single feature for partitioning
    # Or, we can create partitions based on segments of the light curve.
    # For simplicity, let's assume the entire light curve is the input to the TransformerEncoder
    # and create dummy partitions for MultiViewModel initialization.
    # A more sophisticated approach would involve actual feature partitioning of the light curve.
    
    # For light curves, the 'features' are the time steps. We can partition these.
    # Let's create simple contiguous partitions for now.
    # Example: if length is 2048 and K_partitions is 4, each partition is 512 long.
    partition_size = sample_light_curve_length // K_partitions
    partitions = [[j for j in range(i * partition_size, (i + 1) * partition_size)] for i in range(K_partitions - 1)]
    # Add remaining elements to the last partition
    partitions.append([j for j in range((K_partitions - 1) * partition_size, sample_light_curve_length)])
    
    # Filter out empty partitions if any due to division
    partitions = [p for p in partitions if p]
    logger.info(f"Created {len(partitions)} feature partitions.")

    # 2. Model, Optimizer, and Loss
    model = MultiViewModel(
        partitions=partitions,
        encoder_type="transformer",
        output_dim=output_dim,
        proj_dim=proj_dim,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    logger.info("Starting CACL pre-training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for aug1_batch, aug2_batch in dataloader:
            # For contrastive learning, we treat each augmented version as a 'view'
            # The MultiViewModel expects a list of views, where each view corresponds to a partition.
            # Here, we have two augmented versions of the *entire* light curve.
            # We need to adapt this to the MultiViewModel's expectation of K partitions per sample.
            
            # For light curves, each 'view' is a segment of the light curve.
            # So, for each augmented light curve, we need to split it into K partitions.
            
            # Reshape aug1_batch and aug2_batch to (batch_size, K_partitions, partition_size)
            # and then pass each partition as a separate view to the MultiViewModel.
            
            # This part needs careful handling to align with MultiViewModel's input expectation
            # MultiViewModel expects a list of K tensors, where each tensor is (batch_size, len(partition_k))
            
            # Let's create views for each augmented batch
            views1 = []
            views2 = []
            for p_idx, part in enumerate(partitions):
                views1.append(aug1_batch[:, part].to(device))
                views2.append(aug2_batch[:, part].to(device))

            # Concatenate views from both augmented batches to form a single batch for MultiViewModel
            # The MultiViewModel expects a list of K tensors, where each tensor is (2*batch_size, len(partition_k))
            combined_views = []
            for p_idx in range(len(partitions)):
                combined_views.append(torch.cat((views1[p_idx], views2[p_idx]), dim=0))
            
            # Generate embeddings for the combined views
            embeddings = model(combined_views)

            # Create batch_idx for contrastive_loss_ctx
            # Each original sample (and its two augmented versions) should have the same batch_idx
            # So, if batch_size is B, the first B samples in combined_views are from aug1, next B from aug2
            # The batch_idx should reflect the original sample index.
            original_batch_indices = torch.arange(aug1_batch.shape[0], device=device)
            batch_idx = torch.cat((original_batch_indices, original_batch_indices), dim=0).tolist()

            # For CACL, we need context groups. These are typically built on the original (unaugmented) data.
            # Since we are pre-training on unlabeled data, we can build context groups dynamically per batch
            # or pre-compute them if the dataset is small enough. For large datasets, dynamic is better.
            # However, build_context_groups expects numpy array, not torch tensor.
            # For simplicity in pre-training, we might initially skip context-aware loss or simplify context generation.
            # Let's simplify context generation for now, assuming a simple kNN on the batch itself.
            
            # For now, let's use a simplified context_idx or even just contrastive_loss if context is too complex for dynamic batching
            # The original CACL notebook builds context groups on the full dataset (X_train).
            # For pre-training, we can either: 
            # 1. Build context groups on the entire unlabeled dataset once (if feasible).
            # 2. Build context groups on the current batch (less ideal for true context, but simpler).
            # 3. Use contrastive_loss (InfoNCE) without context-awareness for pre-training.
            
            # Let's use contrastive_loss for simplicity in this pre-training phase.
            loss = contrastive_loss_ctx(embeddings, ctx_idx=None, batch_idx=batch_idx) # Simplified: no context for now

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 3. Save the pre-trained feature extractor (encoders)
    # We only save the encoders, not the projection heads, as they are discarded for fine-tuning.
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"Pre-trained Transformer feature extractor saved to {output_model_path}")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have some light curve data in your data_dir
    # and that data_fetcher.py has a get_light_curve_paths function
    cacl_pretrain()