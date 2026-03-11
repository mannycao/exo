---
title: 'multimodal-disagreement: A Python package for quantifying inter-modal disagreement in deep learning exoplanet classifiers'
tags:
  - Python
  - astronomy
  - exoplanets
  - machine learning
  - multimodal fusion
  - neural networks
authors:
  - name: Emmanuel Cao
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Department of Computer Science, University of Exoplanet Research, City, Country
   index: 1
date: 10 March 2026
bibliography: paper.bib
---

# Summary

Multimodal deep learning architectures are increasingly the standard for automated exoplanet detection in large-scale transit surveys, integrating 1D time-series light curves and 2D spatial centroid representations. The `multimodal-disagreement` Python package provides tools for computing inter-modal disagreement metrics in these multimodal exoplanet classification pipelines. By applying structured masking to isolate modality-specific predictions, the software enables researchers to compute an inter-modal disagreement metric ($\delta$) and implement a computationally inexpensive "Disagreement Veto" ($\delta > 0.3$) to filter out false-positive transit signals in automated pipelines.

# Statement of Need

The Kepler space telescope identified thousands of planetary candidates by monitoring the brightness of over 150,000 stars [@Borucki:2010]. As dataset scales increased, machine learning methods—particularly deep neural networks—were introduced to improve transit classification accuracy, establishing new benchmarks for mission data releases [@Shallue:2018; @Dattilo:2019]. 

Existing machine learning pipelines typically provide only fused prediction probabilities and do not expose modality-specific diagnostics. This makes it difficult to evaluate internal model consistency when confronted with sparse or ambiguous transit signals.

This package provides tools for extracting modality-specific predictions and computing disagreement metrics that can be used to analyze model stability. It allows researchers and pipeline engineers to quantify internal predictive consistency across astrophysical parameters without requiring the independent retraining of unimodal networks.

# Software Description

The `multimodal-disagreement` package is implemented in Python and organized into modules for prediction extraction, disagreement analysis, and diagnostic visualization.

Key functionality includes:
- Extraction of modality-specific predictions using structured masking.
- Computation of the disagreement metric $\delta = |P_{temporal} - P_{spatial}|$.
- Utilities for analyzing disagreement across astrophysical parameters (such as orbital period regimes).
- Command-line tools and batch scripts for large-scale survey dataset analysis.

The software can be seamlessly integrated into existing exoplanet classification pipelines or used independently for post-hoc diagnostic analysis.

# Installation

The package can be installed directly from source:

```bash
git clone https://github.com/<username>/multimodal-disagreement
cd multimodal-disagreement
pip install -e .
```

# Usage Example

The package provides utilities for extracting modality-specific predictions from multimodal classifiers and computing the inter-modal disagreement metric. A minimal example demonstrating the calculation of the disagreement score ($\delta$) and the application of the veto is shown below:

```python
from multimodal_disagreement.metrics import disagreement, apply_veto

# Simulated independent branch probabilities
p_temporal = 0.91
p_spatial = 0.12

# Calculate the disagreement metric (delta)
delta = disagreement(p_temporal, p_spatial)

print(f"Inter-modal Disagreement: {delta}")
# Output: Inter-modal Disagreement: 0.79

# Apply the recommended Disagreement Veto threshold
is_vetoed = apply_veto(delta, threshold=0.3)

print(f"Flagged as False Positive: {is_vetoed}")
# Output: Flagged as False Positive: True
```

# Acknowledgements

The author confirms sole responsibility for software architecture, implementation, and documentation. Data supporting the validation of this software can be found in the Kepler Data Release 25 archive at the NASA Exoplanet Archive [@NASAExoplanetArchive].

# References
