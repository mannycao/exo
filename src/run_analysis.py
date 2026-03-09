import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DualBranchIntegrityNet
from dataset import SEN12Dataset

# --- CONFIGURATION ---
DATA_DIR = "processed_sen12_data/processed"
MODEL_PATH = "nominal_model.pth"
VIOLATION_OPACITY = 0.8  # We define "80% Cloud" as the Hazard we must detect

def inject_clouds(img_tensor, opacity):
    noise = torch.randn_like(img_tensor)
    return (1 - opacity) * img_tensor + opacity * noise

def run_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Threshold Analysis (Experiment 5) on {device}...")

    # 1. Load Model
    model = DualBranchIntegrityNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
         print("❌ Model file not found. Please finish full training first.")
         return
    model.eval()

    # 2. Data Loader
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    val_ds = SEN12Dataset(DATA_DIR, split='val', transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    nominal_scores = []
    violation_scores = []

    print("📊 Collecting instance-level Disagreement scores...")
    with torch.no_grad():
        for x_eo, x_sar, _ in val_loader:
            x_eo, x_sar = x_eo.to(device), x_sar.to(device)

            # --- PASS 1: NOMINAL (Clear) ---
            # This generates the "Normal" distribution for False Veto Rate
            _, p_eo_nom, p_sar_nom = model(x_eo, x_sar)
            delta_nom = torch.abs(p_eo_nom - p_sar_nom).cpu().numpy().flatten()
            nominal_scores.extend(delta_nom)

            # --- PASS 2: VIOLATION (Cloudy) ---
            # This generates the "Hazard" distribution for Detection Rate
            x_eo_deg = inject_clouds(x_eo, VIOLATION_OPACITY)
            _, p_eo_viol, p_sar_viol = model(x_eo_deg, x_sar)
            delta_viol = torch.abs(p_eo_viol - p_sar_viol).cpu().numpy().flatten()
            violation_scores.extend(delta_viol)

    # 3. Calculate ROC & Statistics
    y_true = [0] * len(nominal_scores) + [1] * len(violation_scores)
    y_scores = nominal_scores + violation_scores
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n✅ Analysis Complete:")
    print(f"   AUROC: {roc_auc:.4f}")
    print(f"   Optimal Threshold (τ): {best_threshold:.4f}")
    print(f"   False Veto Rate (FVR) at τ: {fpr[best_idx]:.4f}")
    print(f"   Missed Violation Rate (MVR) at τ: {1 - tpr[best_idx]:.4f}")

    # --- PLOTTING (Figure 3) ---
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Distribution (The Integrity Band)
    sns.kdeplot(nominal_scores, fill=True, color='green', label='Nominal (Clear)', ax=axes[0])
    sns.kdeplot(violation_scores, fill=True, color='red', label=f'Violation (Cloud {VIOLATION_OPACITY})', ax=axes[0])
    axes[0].axvline(best_threshold, color='black', linestyle='--', label=f'Optimal τ={best_threshold:.2f}')
    axes[0].set_title('Disagreement (δ) Distribution')
    axes[0].set_xlabel('Disagreement Score')
    axes[0].legend()

    # Subplot 2: ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].scatter(fpr[best_idx], tpr[best_idx], marker='o', color='black', zorder=5, label=f'Optimal τ')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Veto Rate (FPR)')
    axes[1].set_ylabel('True Detection Rate (TPR)')
    axes[1].set_title('ROC Analysis')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('Figure3_Threshold_Analysis.png', dpi=300)
    print("✅ Generated 'Figure3_Threshold_Analysis.png'")
    
    # Save statistics for the paper text
    with open("threshold_stats.txt", "w") as f:
        f.write(f"AUROC: {roc_auc}\n")
        f.write(f"Optimal Threshold: {best_threshold}\n")
        f.write(f"FVR: {fpr[best_idx]}\n")
        f.write(f"MVR: {1-tpr[best_idx]}\n")

if __name__ == "__main__":
    run_analysis()