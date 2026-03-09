import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DualBranchIntegrityNet
from dataset import SEN12Dataset

# Config
DATA_DIR = "processed_sen12_data/processed"
MODEL_PATH = "nominal_model.pth"

def inject_speckle(img_tensor, intensity):
    """
    Simulates SAR Speckle Noise (Multiplicative).
    Intensity: Variance of the noise.
    """
    # Speckle is multiplicative: Signal = Signal * (1 + Noise)
    noise = torch.randn_like(img_tensor) * intensity
    return img_tensor * (1 + noise)

def evaluate():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"⚠️ Starting SAR Stress Test (Experiment 3) on {device}...")
    
    model = DualBranchIntegrityNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    val_ds = SEN12Dataset(DATA_DIR, split='val', transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    results = []
    
    # Sweep Noise Intensity
    for intensity in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        deltas = []
        accuracies = []
        
        with torch.no_grad():
            for x_eo, x_sar, y in val_loader:
                x_eo, x_sar, y = x_eo.to(device), x_sar.to(device), y.float().to(device).unsqueeze(1)
                
                # INJECT FAULT (SAR Only)
                x_sar_deg = inject_speckle(x_sar, intensity)
                
                # Run System
                p_fused, p_eo, p_sar = model(x_eo, x_sar_deg)
                
                delta = torch.abs(p_eo - p_sar)
                deltas.extend(delta.cpu().numpy().flatten())
                
                preds = (p_fused > 0.5).float()
                acc = (preds == y).float().mean().item()
                accuracies.append(acc)
        
        mean_delta = np.mean(deltas)
        mean_acc = np.mean(accuracies)
        print(f"Intensity {intensity:.1f} -> Accuracy: {mean_acc:.2f} | Disagreement (δ): {mean_delta:.3f}")
        
        results.append({
            'Intensity': intensity,
            'Mean_Disagreement': mean_delta,
            'Fused_Accuracy': mean_acc,
            'Experiment': 'Asymmetric_SAR_Speckle'
        })
        
    pd.DataFrame(results).to_csv("sar_noise_results.csv", index=False)
    print("✅ Results saved to 'sar_noise_results.csv'")

if __name__ == "__main__":
    evaluate()