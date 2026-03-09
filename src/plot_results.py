import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_final_results():
    # Load your uploaded data
    df_cloud = pd.read_csv("cloud_cliff_results.csv")
    df_sym = pd.read_csv("symmetric_results.csv")

    # Setup Systems Engineering Plot Style
    plt.style.use('seaborn-v0_8-paper')
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- AXIS 1: INTEGRITY SIGNAL (The Veto) ---
    color_delta = '#D62728' # Brick Red
    ax1.set_xlabel('Degradation Severity (Normalized)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Integrity Monitor (δ)', color=color_delta, fontsize=14, fontweight='bold')
    
    # Normalize X-axis for comparison
    # Cloud Opacity is already 0-1. Symmetric Severity is 0-10, so we divide by 10.
    x_cloud = df_cloud['Opacity']
    x_sym = df_sym['Severity'] / 10.0
    
    # Plot The "Cliff" (Asymmetric)
    l1, = ax1.plot(x_cloud, df_cloud['Mean_Disagreement'], color=color_delta, 
             marker='o', markersize=8, linewidth=3, label='δ (Asymmetric - Cloud)')
    
    # Plot The "Blind Spot" (Symmetric)
    l2, = ax1.plot(x_sym, df_sym['Mean_Disagreement'], color=color_delta, 
             linestyle='--', linewidth=2, alpha=0.6, label='δ (Symmetric - Blur)')
    
    ax1.tick_params(axis='y', labelcolor=color_delta, labelsize=12)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add Safety Threshold Zone
    ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5)
    ax1.text(0.02, 0.32, 'Safety Veto Threshold (τ=0.3)', color='gray', fontsize=10, style='italic')

    # --- AXIS 2: SYSTEM PERFORMANCE (The Masking) ---
    ax2 = ax1.twinx()
    color_acc = '#1F77B4' # Steel Blue
    ax2.set_ylabel('System Confidence / Accuracy', color=color_acc, fontsize=14, fontweight='bold')
    
    # Plot Accuracy
    l3, = ax2.plot(x_cloud, df_cloud['Fused_Accuracy'], color=color_acc, 
             marker='s', markersize=8, linewidth=3, label='Fused Accuracy (Asymmetric)')
    l4, = ax2.plot(x_sym, df_sym['Fused_Accuracy'], color=color_acc, 
             linestyle='--', linewidth=2, alpha=0.6, label='Fused Accuracy (Symmetric)')
    
    ax2.tick_params(axis='y', labelcolor=color_acc, labelsize=12)
    ax2.set_ylim(0.8, 1.05) # Zoom in to show high accuracy
    
    # Legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    plt.title('Validation of Modal Masking: Integrity Signal vs. System Performance', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('Figure2_Modal_Masking.png', dpi=300)
    print("✅ Figure generated: Figure2_Modal_Masking.png")

if __name__ == "__main__":
    plot_final_results()