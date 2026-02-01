
import matplotlib.pyplot as plt
import numpy as np

# Data based on the user's request
categories = ['High Agreement (1D == 2D)', 'Disagreement (1D != 2D)']
error_rates = [0.021, 0.425]  # 2.1% and 42.5%

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(categories, error_rates, color=['#4CAF50', '#F44336'])

# Add hatch pattern to the 'Disagreement' bar
bars[1].set_hatch('//')

# Add labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.1%}', ha='center', va='bottom')

# Add labels and title
ax.set_ylabel('Error Rate')
ax.set_title('Modality Agreement vs. Error Rate')
ax.set_ylim(0, 0.5)

# Add annotation with an arrow
ax.annotate('Latent Miscalibration Detected',
            xy=(1, 0.425),  # Point to the top of the 'Disagreement' bar
            xytext=(1.2, 0.45),
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='left',
            verticalalignment='top')

# Format the y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()
plt.savefig('Fig5_Consistency.png', dpi=300)

print("Figure saved as Fig5_Consistency.png")
