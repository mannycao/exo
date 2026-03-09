import torch
from torchvision import transforms
from PIL import Image
from model import DualBranchIntegrityNet
from dataset import SEN12Dataset # Used for data loading example, assume some dummy data

# --- Configuration (Ensure these paths are correct) ---
MODEL_PATH = "nominal_model.pth"
# Assuming DATA_DIR points to the processed SEN12 dataset
DATA_DIR = "processed_sen12_data/processed" 

def run_sample_inference():
    # Set device for inference
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model = DualBranchIntegrityNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode

    # Prepare dummy data for inference
    # In a real scenario, you would load an actual image
    # For demonstration, let's grab a sample from the validation dataset
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    sample_dataset = SEN12Dataset(DATA_DIR, split='val', transform=tfm)
    
    if len(sample_dataset) == 0:
        print("Error: No samples found in the dataset. Cannot perform inference.")
        return

    # Take the first sample
    x_eo_sample, x_sar_sample, _ = sample_dataset[0] 
    
    # Add a batch dimension (model expects batches)
    x_eo_sample = x_eo_sample.unsqueeze(0).to(device)
    x_sar_sample = x_sar_sample.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        p_fused, p_eo, p_sar = model(x_eo_sample, x_sar_sample)

    print("
--- Sample Inference Results ---")
    print(f"Fused Prediction: {p_fused.item():.4f}")
    print(f"EO Branch Prediction: {p_eo.item():.4f}")
    print(f"SAR Branch Prediction: {p_sar.item():.4f}")
    print(f"Disagreement (delta): {torch.abs(p_eo - p_sar).item():.4f}")

if __name__ == "__main__":
    run_sample_inference()
