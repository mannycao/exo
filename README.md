# Exoplanet Detection Pipeline using Deep Learning

This project implements a sophisticated, end-to-end pipeline to detect exoplanet transits from time-series photometric data, such as that from the Kepler Space Telescope. It leverages a multimodal deep learning model that analyzes both 1D time-series data and its 2D image representation to achieve high-precision classification.

For a comprehensive explanation of the methodology, pipeline stages, and full usage instructions, please see the detailed **[documentation.md](documentation.md)**.

---

## Features

-   **Multimodal Fusion Model**: Combines 1D CNNs for time-series analysis and 2D CNNs for image-based feature extraction.
-   **End-to-End Pipeline**: Handles everything from data fetching and preprocessing to model training, evaluation, and reporting.
-   **Robust Training**: Implements techniques like focal loss and class weighting to handle the severe class imbalance inherent in astronomical data.
-   **Scientific Validation**: Includes scripts for rigorous K-Fold cross-validation to produce a reliable measure of the model's performance.
-   **Hyperparameter Tuning**: Provides a dedicated script to automate the search for optimal model and training parameters.

---

## Quickstart

### 1. Setup

First, create and activate a Python virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
