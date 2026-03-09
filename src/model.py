import torch
import torch.nn as nn
import torchvision.models as models

class DualBranchIntegrityNet(nn.Module):
    def __init__(self):
        super(DualBranchIntegrityNet, self).__init__()
        
        # --- Branch A: Electro-Optical (RGB) ---
        # Standard ResNet18
        # Note: weights='DEFAULT' loads the best available pre-trained weights
        self.eo_branch = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.eo_branch.fc = nn.Linear(512, 128) # Feature Bottleneck (512 -> 128)
        
        # --- Branch B: SAR (Radar) ---
        # ResNet18 adapted for 1-channel input
        self.sar_branch = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # KEY ARCHITECTURAL CHANGE: Modify first layer for 1-channel input (Grayscale SAR)
        # Original was Conv2d(3, 64, ...). We change to Conv2d(1, 64, ...).
        # We sum the weights of the original RGB channels to keep the pre-trained initialization magnitude.
        original_weights = self.sar_branch.conv1.weight.data
        new_weights = original_weights.sum(dim=1, keepdim=True)
        
        self.sar_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sar_branch.conv1.weight.data = new_weights
        
        self.sar_branch.fc = nn.Linear(512, 128) # Feature Bottleneck
        
        # --- Consensus/Fusion Layer ---
        # Takes concatenated features (128 + 128 = 256)
        self.fusion = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout is critical for "Systems Engineering" robustness claims
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # --- Independent Heads (The "Sensors") ---
        # These allow us to calculate Disagreement (delta = |p_eo - p_sar|)
        self.eo_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.sar_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x_eo, x_sar):
        # 1. Feature Extraction
        f_eo = self.eo_branch(x_eo)   # Shape: [Batch, 128]
        f_sar = self.sar_branch(x_sar) # Shape: [Batch, 128]
        
        # 2. Independent Predictions (Internal System State)
        p_eo = self.eo_head(f_eo)
        p_sar = self.sar_head(f_sar)
        
        # 3. Fusion (External System Output)
        combined = torch.cat((f_eo, f_sar), dim=1) # Shape: [Batch, 256]
        p_fused = self.fusion(combined)
        
        return p_fused, p_eo, p_sar