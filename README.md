# multimodal-disagreement: Exoplanet Classifier Integrity Tools

`multimodal-disagreement` is a Python package for quantifying internal consistency in multimodal deep learning exoplanet classifiers. It enables researchers to compute an inter-modal disagreement metric ($\delta$) between temporal (1D light curve) and spatial (2D centroid) representations, and implement a "Disagreement Veto" to filter out physically implausible or ambiguous transit signals.

## Key Features
- **Inter-modal Disagreement**: Compute $\delta = |P_{temporal} - P_{spatial}|$ to detect model conflict.
- **Disagreement Veto**: Apply a computationally inexpensive filter ($\delta > 0.3$) to improve pipeline reliability.
- **Modal Masking**: Tools for structured branch isolation without requiring independent unimodal retraining.
- **Statistical Validation**: Integrated support for significance testing and bootstrap confidence intervals across astrophysical parameters.

## Installation

### From Source
```bash
git clone https://github.com/<username>/multimodal-disagreement
cd multimodal-disagreement
pip install -e .
```

### Dependencies
The package requires:
- `numpy`, `pandas`, `scipy`
- `tensorflow`
- `statsmodels`
- `scikit-posthocs`

## Usage for New Datasets

You can use the package's core metrics on any multimodal classification output.

```python
from multimodal_disagreement.metrics import disagreement, apply_veto

# 1. Provide predictions from your temporal and spatial branches
p_temp = 0.95  # Confidence from 1D branch
p_spat = 0.15  # Confidence from 2D branch

# 2. Calculate disagreement (delta)
delta = disagreement(p_temp, p_spat)
print(f"Disagreement Score: {delta:.4f}")

# 3. Check if the signal should be vetoed
is_vetoed = apply_veto(delta, threshold=0.3)
if is_vetoed:
    print("VETO: High inter-modal disagreement detected.")
```

## Reproducing Paper Experiments

To reproduce the analysis from the paper "multimodal-disagreement: A Python package for quantifying inter-modal disagreement in deep learning exoplanet classifiers":

1. **Significance Tests**: Verify $p < 10^{-4}$ across regimes.
   ```bash
   python3 paper2_experiments/run_significance_tests.py
   ```

2. **Veto Impact**: Stratify False-Positive Rate by $\delta$.
   ```bash
   python3 paper2_experiments/evaluate_veto_impact.py
   ```

3. **Bootstrap CI**: Validate the "Reliability Cliff" with 5,000 iterations.
   ```bash
   python3 paper2_experiments/generate_bootstrap_ci.py
   ```

## Documentation
For a full description of the methodology and results, see `paper.md`.
