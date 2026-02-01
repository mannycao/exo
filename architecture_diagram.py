import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_elegant():
    fig, ax = plt.subplots(figsize=(24, 12)) # Significantly increased figure size
    ax.set_xlim(0, 24) # Adjusted x-limit
    ax.set_ylim(0, 12) # Adjusted y-limit
    ax.axis('off')

    # Define common box style for process steps
    process_box_props = dict(boxstyle='round,pad=1.5', facecolor='#f0f0f0', edgecolor='gray', linewidth=1.5, alpha=0.9) # Increased pad and linewidth
    # Define input box style
    input_box_props = dict(boxstyle='round,pad=1.5', facecolor='#e0f2f7', edgecolor='#0077b6', linewidth=1.8, alpha=0.9) # Increased pad and linewidth
    # Define model box style
    model_box_props = dict(boxstyle='round,pad=1.5', facecolor='#e6f2ff', edgecolor='#3366cc', linewidth=1.8, alpha=0.9) # Increased pad and linewidth
    # Define score box style (more subtle)
    score_box_props = dict(boxstyle='circle,pad=1.2', facecolor='#fff0e6', edgecolor='#ff9933', linewidth=1.5, alpha=0.9) # Increased pad and linewidth
    # Define oracle box style (distinct, but not jagged)
    oracle_box_props = dict(boxstyle='round,pad=1.5', facecolor='#ffe6e6', edgecolor='#cc0000', linewidth=2.0, alpha=0.9) # Increased pad and linewidth

    # Font sizes - further increased manually
    fs_main = 20
    fs_sub = 18

    # --- Nodes ---
    # Input
    ax.text(2.5, 6, "Raw Data\n(Kepler Light Curves)", ha='center', va='center', bbox=input_box_props, fontsize=fs_main)
    
    # Preprocessing
    # Moved further right to ensure clear separation from Models
    ax.text(9.0, 9, "Transformation A\n(Time Series Features)", ha='center', va='center', bbox=process_box_props, fontsize=fs_sub)
    ax.text(9.0, 3, "Transformation B\n(Phase Fold Image)", ha='center', va='center', bbox=process_box_props, fontsize=fs_sub)
    
    # Models (Moved to the left, but now with a larger gap from transformations)
    ax.text(13.0, 9, "Model 1D\n(CNN on Time Series)", ha='center', va='center', bbox=model_box_props, fontsize=fs_sub) 
    ax.text(13.0, 3, "Model 2D\n(CNN on Image)", ha='center', va='center', bbox=model_box_props, fontsize=fs_sub) 
    
    # Output Scores (Adjusted to keep relative spacing with new model positions)
    ax.text(17.5, 9, "P(1D)", ha='center', va='center', bbox=score_box_props, fontsize=fs_main) 
    ax.text(17.5, 3, "P(2D)", ha='center', va='center', bbox=score_box_props, fontsize=fs_main) 
    
    # Oracle (Adjusted to keep relative spacing)
    ax.text(22.0, 6, "Disagreement Oracle\n(|P(1D) - P(2D)| > tau)", ha='center', va='center', bbox=oracle_box_props, fontsize=fs_main)

    # --- Arrows ---
    arrow_props_common = dict(arrowstyle="->", lw=2.5, color='gray', connectionstyle="arc3,rad=0") # Increased lw
    arrow_props_red = dict(arrowstyle="->", lw=3.0, color='red', connectionstyle="arc3,rad=0") # Increased lw

    # Raw Data to Transformations 
    ax.annotate("", xy=(7.0, 9), xytext=(4.5, 6.5), arrowprops=arrow_props_common)
    ax.annotate("", xy=(7.0, 3), xytext=(4.5, 5.5), arrowprops=arrow_props_common)
    
    # Transformations to Models (Corrected to avoid overlap)
    # xytext (start of arrow) needs to be at the right edge of the Transformation bubble (x-coord of Transformation + half_width)
    # xy (tip of arrow) needs to be at the left edge of the Model bubble (x-coord of Model - half_width)
    # Assuming text box width is ~3.0-3.5 units for the Transformations and Models
    # Transformation A is at x=9.0, Model 1D is at x=13.0
    # Center of Transformation A is 9.0. Right edge is approx 9.0 + 1.75 = 10.75
    # Center of Model 1D is 13.0. Left edge is approx 13.0 - 1.75 = 11.25
    ax.annotate("", xy=(11.5, 9), xytext=(10.5, 9), arrowprops=arrow_props_common) 
    ax.annotate("", xy=(11.5, 3), xytext=(10.5, 3), arrowprops=arrow_props_common) 

    # Models to Scores (Adjusted to keep relative spacing with new model positions)
    ax.annotate("", xy=(16.0, 9), xytext=(14.5, 9), arrowprops=arrow_props_common) 
    ax.annotate("", xy=(16.0, 3), xytext=(14.5, 3), arrowprops=arrow_props_common) 
    
    # Scores to Oracle (Adjusted to keep relative spacing)
    ax.annotate("", xy=(21, 7.5), xytext=(19, 8.5), arrowprops=arrow_props_red)
    ax.annotate("", xy=(21, 4.5), xytext=(19, 3.5), arrowprops=arrow_props_red)

    plt.title("Multimodal Disagreement Oracle Architecture", fontsize=28, pad=30)
    plt.tight_layout()
    plt.savefig('architecture_diagram_elegant.png', dpi=300)
    print("Generated architecture_diagram_elegant.png")

if __name__ == "__main__":
    draw_architecture_elegant()
