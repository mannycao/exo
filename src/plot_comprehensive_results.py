import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_final_results():
    # Load all three result files
    files = ["cloud_cliff_results.csv", "symmetric_results.csv", "sar_noise_results.csv"]
    for f in files:
        if not os.path.exists(f):
            print(f"❌ Missing {f}. Run that experiment first!")
            return

    df_cloud = pd.read_csv("cloud_cliff_results.csv")
    df_sym = pd.read_csv("symmetric_results.csv")
    df_sar = pd.read_csv("sar_noise_results.csv")

    # Setup Plot
    plt.style.use('seaborn-v0_8-paper')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- AXIS 1: INTEGRITY (Disagreement) ---
    ax1.set_xlabel('Degradation Severity (Normalized)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Integrity Monitor (δ)', fontsize=14, fontweight='bold')
    
    # Normalize X-axes to 0-1 scale for comparison
    x_cloud = df_cloud['Opacity']
    x_sar = df_sar['Intensity'] 
    x_sym = df_sym['Severity'] / 10.0
    
    # 1. Optical Failure (Red - The Cliff)
    l1, = ax1.plot(x_cloud, df_cloud['Mean_Disagreement'], color='#D62728', 
             marker='o', linewidth=3, label='δ (Optical Failure)')
    
    # 2. Radar Failure (Green - The Degradation)
    l2, = ax1.plot(x_sar, df_sar['Mean_Disagreement'], color='#2CA02C', 
             marker='^', linewidth=3, label='δ (Radar Failure)')

    # 3. Symmetric Failure (Gray Dotted - The Blind Spot)
    l3, = ax1.plot(x_sym, df_sym['Mean_Disagreement'], color='gray', 
             linestyle='--', linewidth=2, alpha=0.8, label='δ (Symmetric Blind Spot)')
    
    ax1.set_ylim(-0.05, 1.05)
    
    # Add Safety Threshold
    ax1.axhline(y=0.32, color='black', linestyle=':', label='Safety Threshold (τ=0.32)')

    # --- AXIS 2: PERFORMANCE (Dotted Lines) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('System Accuracy', color='gray', fontsize=14, fontweight='bold')
    
    # Plot Accuracies (Thinner lines)
    l4, = ax2.plot(x_cloud, df_cloud['Fused_Accuracy'], color='#D62728', linestyle=':', alpha=0.5)
    l5, = ax2.plot(x_sar, df_sar['Fused_Accuracy'], color='#2CA02C', linestyle=':', alpha=0.5)
    l6, = ax2.plot(x_sym, df_sym['Fused_Accuracy'], color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_ylim(0.5, 1.05)
    
    # Legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    plt.title('Validation of Integrity Monitor Across All Failure Modes', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('Figure2_Comprehensive_Validation.png', dpi=300)
    print("✅ Figure generated: Figure2_Comprehensive_Validation.png")

if __name__ == "__main__":
    plot_final_results()