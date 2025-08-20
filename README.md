# Exoplanet Detection Pipeline

## Project Overview
This project implements an advanced pipeline for detecting exoplanets from light curve data, leveraging multimodal deep learning and Bayesian inference for robust predictions and uncertainty quantification. It integrates data fetching, preprocessing, model training, hyperparameter tuning, and uncertainty analysis into a cohesive workflow.

## Background
The search for exoplanets is a rapidly evolving field in astrophysics. Traditional methods often struggle with noisy data and the inherent uncertainties in astronomical observations. This pipeline aims to address these challenges by:
- Utilizing both time-series and image-based representations of light curves.
- Employing deep learning models capable of learning complex patterns.
- Incorporating Bayesian inference (specifically, Monte Carlo Dropout) to provide not just predictions, but also a measure of confidence (uncertainty) in those predictions, which is crucial for scientific discovery and follow-up observations.

## Features
- **Multimodal Data Processing**: Handles both raw time-series light curves and their image representations (e.g., phase-folded transit images).
- **Deep Learning Models**: Implements Convolutional Neural Networks (CNNs) for image data and 1D CNNs for time-series data, fused into a multimodal architecture.
- **Bayesian Inference**: Quantifies prediction uncertainty using Monte Carlo Dropout.
- **Hyperparameter Tuning**: Supports automated hyperparameter optimization for model training.
- **Data Management**: Includes scripts for fetching and generating synthetic light curve data.
- **Comprehensive Reporting**: Generates detailed reports and visualizations of model performance and uncertainty metrics.

## Installation & Setup

To get this project up and running, follow these steps:

### 1. Clone the Repository
First, clone the project repository to your local machine:
```bash
git clone https://github.com/your-username/exoplanet-detection.git
cd exoplanet-detection
```
*(Replace `https://github.com/your-username/exoplanet-detection.git` with the actual repository URL)*

### 2. Set up a Python Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with your system's Python packages.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
Install all required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Generate Sample Data (Optional, for quick start)
If you don't have real light curve data, you can generate a small sample dataset for testing the pipeline:
```bash
python3 generate_data.py
```
This will create a `sample_data` directory containing synthetic light curves.

## Usage

### 1. Run the Main Pipeline
The `main.py` script orchestrates the data processing, model training, and evaluation.

```bash
python3 main.py --planets_dir <path_to_confirmed_planets_data> \
                --false_positives_dir <path_to_false_positives_data> \
                --split_type [all|50_50]
```
-   `<path_to_confirmed_planets_data>`: Directory containing FITS files for confirmed exoplanets.
-   `<path_to_false_positives_data>`: Directory containing FITS files for false positives.
-   `--split_type`:
    -   `all`: Uses the entire dataset for training and validation.
    -   `50_50`: Uses a 50/50 split of planets and false positives for a balanced dataset.

**Example using generated sample data:**
```bash
python3 main.py --planets_dir sample_data/confirmed_planets \
                --false_positives_dir sample_data/false_positives \
                --split_type all
```
This will output results, including the trained model, to a timestamped directory under `results/`.

### 2. Run Bayesian Inference
After training a model, you can run Bayesian inference to quantify prediction uncertainties.

```bash
python3 run_bayesian_inference.py --model_path <path_to_trained_model.keras> \
                                  --planets_dir <path_to_confirmed_planets_data> \
                                  --false_positives_dir <path_to_false_positives_data> \
                                  --output_dir <output_directory_for_inference_results> \
                                  --n_samples <number_of_monte_carlo_samples>
```
-   `<path_to_trained_model.keras>`: Path to the saved Keras model file (e.g., `results/run_YYYYMMDD-HHMMSS/exo_multimodal_model_best.keras`).
-   `<output_directory_for_inference_results>`: Directory where inference results and plots will be saved.
-   `--n_samples`: Number of Monte Carlo samples to draw for uncertainty estimation (e.g., 100).

### 3. Run Hyperparameter Tuning
To optimize model hyperparameters, use the `run_hp_tuning.py` script.

```bash
python3 run_hp_tuning.py --planets_dir <path_to_confirmed_planets_data> \
                         --false_positives_dir <path_to_false_positives_data> \
                         --n_iter <number_of_tuning_iterations> \
                         --cv <number_of_cross_validation_folds> \
                         --max_workers <number_of_parallel_workers>
```
-   `--n_iter`: Number of hyperparameter combinations to try.
-   `--cv`: Number of cross-validation folds for evaluation.
-   `--max_workers`: Number of parallel processes to use for tuning.

## Project Structure
```
.
├── config.py                 # Global configuration settings
├── main.py                   # Main pipeline execution script (training, evaluation)
├── generate_data.py          # Script to generate synthetic light curve data
├── run_bayesian_inference.py # Script for running Bayesian inference
├── run_hp_tuning.py          # Script for hyperparameter tuning
├── requirements.txt          # Python dependency list
├── system_test.py            # Comprehensive system tests
├── data/
│   ├── data_fetcher.py       # Utilities for fetching and handling data
│   ├── dataset_generator.py  # Creates multimodal datasets
│   └── light_curves/         # Placeholder for raw light curve data
│   └── metadata/             # Exoplanet metadata
├── models/
│   ├── bayesian_predictor.py # Implements Bayesian prediction logic
│   ├── cnn_model.py          # CNN model definitions
│   ├── multimodal_model.py   # Combines CNNs for multimodal input
│   └── model_trainer.py      # Handles model training
├── pipeline/
│   ├── enhanced_pipeline_runner.py # Orchestrates the main pipeline steps
│   └── report_generator.py   # Generates final reports
├── utils/
│   ├── file_utils.py         # File system utilities (e.g., logging setup)
│   ├── metrics.py            # Custom evaluation metrics
│   └── plotting.py           # Plotting and visualization functions
└── results/                  # Directory for all pipeline outputs (models, reports, plots)
```

## Version Control
This project uses Git for version control. The main branch is `main`.
-   **Branching Strategy**: We recommend a feature-branch workflow. Create a new branch for each new feature or bug fix from `main`.
    ```bash
    git checkout main
    git pull origin main
    git checkout -b feature/your-feature-name
    ```
-   **Commit Messages**: Use clear and concise commit messages. A good practice is to start with a type (e.g., `feat:`, `fix:`, `docs:`) followed by a brief description.
-   **Pull Requests**: All changes should be submitted via Pull Requests to the `main` branch for review.

## Contributing
We welcome contributions to this project! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure they adhere to the existing code style.
4.  Write or update tests for your changes.
5.  Ensure all existing tests pass (`python3 system_test.py`).
6.  Commit your changes (`git commit -m "feat: Add new feature"`).
7.  Push to your fork (`git push origin feature/your-feature-name`).
8.  Create a Pull Request to the `main` branch of the upstream repository.

## Next Steps / Future Enhancements
-   **Real Data Integration**: Implement robust data fetching and preprocessing for large-scale real astronomical datasets (e.g., Kepler, TESS).
-   **Advanced Models**: Explore more sophisticated deep learning architectures (e.g., Transformers, LSTMs) or ensemble methods.
-   **Uncertainty Calibration**: Improve the calibration of predicted uncertainties.
-   **Deployment**: Develop a deployment strategy (e.g., Docker containers, cloud functions) for running the pipeline in production.
-   **Web Interface**: Create a simple web-based interface for easier interaction and visualization of results.
-   **Continuous Integration/Deployment (CI/CD)**: Set up automated testing and deployment pipelines.

## License
This project is licensed under the MIT License - see the LICENSE file for details.