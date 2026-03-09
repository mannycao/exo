import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DualBranchIntegrityNet
from dataset import SEN12Dataset

# Config (Ensure this matches your training paths)
DATA_DIR = "processed_sen12_data/processed"
MODEL_PATH = "nominal_model.pth"

def symmetric_blur(img_tensor, k_size):
    """
    Simulates Symmetric Degradation (e.g., Optics Blur + Radar Defocusing).
    """
    if k_size == 0:
        return img_tensor
    # Kernel size must be odd
    k = 2 * k_size + 1
    transform = transforms.GaussianBlur(kernel_size=k, sigma=(k_size * 0.5))
    return transform(img_tensor)

def evaluate():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"⚠️ Starting Symmetric Stress Test (The Blind Spot) on {device}...")
    
    # Load Model
    model = DualBranchIntegrityNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("❌ Model file not found. Please finish full training first.")
        return

    model.eval()
    
    # Data
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # Use 'val' or 'test' split
    val_ds = SEN12Dataset(DATA_DIR, split='val', transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    results = []
    
    # Sweep Blur Radius (0 = Clear, 10 = Heavy Blur)
    for severity in [0, 2, 4, 6, 8, 10]:
        deltas = []
        accuracies = []
        
        with torch.no_grad():
            for x_eo, x_sar, y in val_loader:
                x_eo, x_sar, y = x_eo.to(device), x_sar.to(device), y.float().to(device).unsqueeze(1)
                
                # INJECT SYMMETRIC FAULT (Blur BOTH)
                x_eo_deg = symmetric_blur(x_eo, severity)
                x_sar_deg = symmetric_blur(x_sar, severity)
                
                # Run System
                p_fused, p_eo, p_sar = model(x_eo_deg, x_sar_deg)
                
                # Integrity Metric
                delta = torch.abs(p_eo - p_sar)
                deltas.extend(delta.cpu().numpy().flatten())
                
                # Performance Metric
                preds = (p_fused > 0.5).float()
                acc = (preds == y).float().mean().item()
                accuracies.append(acc)
        
        mean_delta = np.mean(deltas)
        mean_acc = np.mean(accuracies)
        print(f"Severity {severity} -> Accuracy: {mean_acc:.2f} | Disagreement (δ): {mean_delta:.3f}")
        
        results.append({
            'Severity': severity,
            'Mean_Disagreement': mean_delta,
            'Fused_Accuracy': mean_acc,
            'Experiment': 'Symmetric_Blur'
        })
        
    # Save
    pd.DataFrame(results).to_csv("symmetric_results.csv", index=False)
    print("✅ Results saved to 'symmetric_results.csv'")

if __name__ == "__main__":
    evaluate()