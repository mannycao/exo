# src/train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DualBranchIntegrityNet
from dataset import SEN12Dataset

# Config
DATA_DIR = "processed_sen12_data/processed"
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🚀 Training Nominal Baseline on {device}...")
    
    # Transforms (Normalize to ResNet standards)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Data
    train_ds = SEN12Dataset(DATA_DIR, split='train', transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = DualBranchIntegrityNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    # Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_delta = 0
        
        for i, (x_eo, x_sar, y) in enumerate(train_loader):
            x_eo, x_sar, y = x_eo.to(device), x_sar.to(device), y.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            p_fused, p_eo, p_sar = model(x_eo, x_sar)
            
            # Loss: Train Fusion AND Consistency
            # We enforce p_eo ~ p_sar implicitly by training both to match ground truth
            loss = criterion(p_fused, y) + 0.5*criterion(p_eo, y) + 0.5*criterion(p_sar, y)
            
            loss.backward()
            optimizer.step()
            
            # Track Integrity Metric
            delta = torch.abs(p_eo - p_sar).mean().item()
            total_loss += loss.item()
            total_delta += delta
            
            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
        avg_delta = total_delta / len(train_loader)
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f} | Mean Disagreement (δ): {avg_delta:.4f}")

    # Save
    torch.save(model.state_dict(), "nominal_model.pth")
    print("✅ Baseline Model Saved.")

if __name__ == "__main__":
    train()
