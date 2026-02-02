import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_governance_layers():
    """
    Generates a figure visualizing the pipeline as a series of governance layers.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    
    layers = [
        ("Layer 5: Discovery Governance & Reporting", "#fff2cc"),
        ("Layer 4: Physical & System Constraints (Oracles)", "#f4cccc"),
        ("Layer 3: Self-Consistency & Adversarial Validation", "#d9ead3"),
        ("Layer 2: Probabilistic Inference (Bayesian)", "#c9daf8"),
        ("Layer 1: Hypothesis Generation (ML)", "#d9d2e9"),
        ("Layer 0: Data & Signal Formation", "#e0e0e0"),
    ]

    ax.set_ylim(0, len(layers) * 1.5)
    ax.set_xlim(0, 10)
    ax.axis('off')

    for i, (name, color) in enumerate(layers):
        y_pos = i * 1.5 + 0.75
        
        # Layer Box
        rect = patches.FancyBboxPatch((1, y_pos - 0.6), 8, 1.2, boxstyle="round,pad=0.02",
                                      linewidth=2, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Layer Text
        ax.text(5, y_pos, name, ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows indicating data flow
    for i in range(len(layers) - 1):
        y_start = i * 1.5 + 1.35
        y_end = (i + 1) * 1.5 + 0.15
        ax.annotate("", xy=(5, y_end), xytext=(5, y_start), 
                    arrowprops=dict(arrowstyle="->", lw=2.5, color='gray'))

    plt.title("The Governed Discovery Pipeline", fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig('governance_layers.png', dpi=300, bbox_inches='tight')
    print("Generated: governance_layers.png")


if __name__ == "__main__":
    draw_governance_layers()
