# Multimodal-Integrity-Monitor

This repository presents a systems-engineering approach to enhance the robustness and reliability of AI-driven perception systems in safety-critical applications. Developed as a JOSS (Journal of Open Source Software) project, it introduces a novel architectural separation between sensor fusion and continuous integrity monitoring. Our focus is on detecting "modal masking" – a condition where a system's output remains stable despite significant degradation in one or more sensor modalities – using Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery.

## Installation

To set up your environment and install the necessary dependencies, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Multimodal-Integrity-Monitor.git
    cd Multimodal-Integrity-Monitor
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    All required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For M-series Apple Silicon Macs, PyTorch will automatically utilize the Metal Performance Shaders (MPS) backend if available, ensuring GPU acceleration.*

## Quick Start: Running a Sample Inference

This section demonstrates how to load the pre-trained `DualBranchIntegrityNet` model and perform a sample inference.

First, ensure you have a trained model file, `nominal_model.pth`, in your project root. If you haven't trained one yet, follow the steps in the "Reproducing Experiments" section.

```python
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

    print("\n--- Sample Inference Results ---")
    print(f"Fused Prediction: {p_fused.item():.4f}")
    print(f"EO Branch Prediction: {p_eo.item():.4f}")
    print(f"SAR Branch Prediction: {p_sar.item():.4f}")
    print(f"Disagreement (delta): {torch.abs(p_eo - p_sar).item():.4f}")

if __name__ == "__main__":
    run_sample_inference()
```
*Save the above code as `src/quick_start.py` and run it from your project root:*
```bash
python src/quick_start.py
```

## Reproducing Experiments

This project includes several scripts to reproduce the experiments discussed in our paper, validating the integrity monitor across various degradation scenarios.

1.  **Train the Baseline Model:**
    The core `DualBranchIntegrityNet` is trained using `src/train_baseline.py`.
    ```bash
    python src/train_baseline.py
    ```
    *Note: Training on the full dataset requires a substantial amount of time and computational resources. This script is configured to leverage GPUs (CUDA or MPS) if available.*

2.  **Evaluate Asymmetric Optical Degradation ("The Cloud Cliff"):**
    This experiment assesses the system's response to cloud occlusion, simulating a failure mode specific to the optical sensor.
    ```bash
    python src/evaluate_cliff.py
    ```
    *Results are saved to `cloud_cliff_results.csv`.*

3.  **Evaluate Asymmetric SAR Degradation:**
    This experiment investigates the impact of SAR speckle noise on the system's performance and integrity.
    ```bash
    python src/evaluate_sar.py
    ```
    *Results are saved to `sar_noise_results.csv`.*

4.  **Evaluate Symmetric Degradation ("The Blind Spot"):**
    This experiment examines scenarios where both EO and SAR modalities are equally degraded (e.g., by blur).
    ```bash
    python src/evaluate_symmetric.py
    ```
    *Results are saved to `symmetric_results.csv`.*

5.  **Perform Threshold Analysis:**
    This script analyzes the distributions of nominal versus degraded disagreement scores and calculates the optimal integrity threshold ($\tau$) using ROC analysis.
    ```bash
    python src/run_analysis.py
    ```
    *Results include `Figure3_Threshold_Analysis.png` and `threshold_stats.txt`.*

6.  **Plot Comprehensive Validation Figure:**
    Generates a combined plot showcasing the integrity monitor's behavior across all tested failure modes.
    ```bash
    python src/plot_comprehensive_results.py
    ```
    *Generates `Figure2_Comprehensive_Validation.png`.*

7.  **Plot Recovery Analysis Figure:**
    Demonstrates the resilience gain achieved by the integrity-managed fusion strategy.
    ```bash
    python src/plot_recovery_analysis.py
    ```
    *Generates `Figure6_Recovery_Analysis.png` and `recovery_results.csv`.*

## Key Systems Engineering Notes

Our integrity monitor is designed around the principle of architectural separation, allowing continuous validation of sensor fusion processes. The optimal integrity threshold, $\tau=0.23$ (as determined by the Youden Index in `run_analysis.py`), is specifically chosen to minimize overall risk in safety-critical contexts by balancing false vetoes against missed anomaly detections. This deliberate design ensures that the system can operate robustly even when individual sensor modalities are compromised.

For a full systems-theoretic formulation, in-depth discussion of modal masking, and Lipschitz continuity proofs underpinning this work, please refer to our accompanying paper in `paper.md`.
