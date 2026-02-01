
import numpy as np
import matplotlib.pyplot as plt
from data.augmentation_utils import generate_synthetic_transit, advanced_phase_folding

# --- 1. Generate Synthetic Data ---
# Use the provided function to create a synthetic transit light curve
time, flux = generate_synthetic_transit(length=2048, transit_depth=0.1, transit_duration=100, noise_level=0.02)
period = 50 # Assume a period for folding

# --- 2. Create the 3-Panel Figure ---
plt.style.use('grayscale') # Use a black and white style for the paper
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle('Figure 3: Light Curve Data Partitioning', fontsize=16, y=0.95)

# --- Panel 1: Temporal Windowing ---
ax1.plot(time, flux, color='black', lw=0.5)
ax1.set_title('Panel A: Temporal Windowing')
ax1.set_ylabel('Normalized Flux')

# Highlight a specific time window in red
window_start, window_end = 800, 1200
ax1.axvspan(window_start, window_end, color='red', alpha=0.3, label='Temporal Window')
ax1.legend(loc='lower right')


# --- Panel 2: Stochastic Masking ---
# Apply a random mask (20% of points to NaN)
masked_flux = flux.copy()
mask_size = int(0.2 * len(masked_flux))
mask_indices = np.random.choice(len(masked_flux), mask_size, replace=False)
masked_flux[mask_indices] = np.nan

ax2.plot(time, masked_flux, color='black', lw=0.5, drawstyle='steps-post')
ax2.set_title('Panel B: Stochastic Masking')
ax2.set_ylabel('Normalized Flux')


# --- Panel 3: Advanced Phase Folding ---
# Use the advanced_phase_folding function
# We need to remove NaN values for the folding function to work correctly
valid_indices = ~np.isnan(flux)
folded_flux = advanced_phase_folding(time[valid_indices], flux[valid_indices], period=period, n_bins=128)
phase = np.linspace(0, 1, len(folded_flux))

ax3.plot(phase, folded_flux, color='black', lw=1)
ax3.set_title('Panel C: Phase-Folded Curve')
ax3.set_xlabel('Phase')
ax3.set_ylabel('Binned Flux')
ax3.set_xlim(0, 1)


# --- 4. Final Touches and Save ---
plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout to make space for suptitle
plt.savefig('Fig3_Partitioning.png', dpi=300)

print("Figure saved as Fig3_Partitioning.png")
