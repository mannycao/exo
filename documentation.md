# Exoplanet Detection Pipeline Documentation

## Overview

This system implements a comprehensive pipeline for exoplanet detection using astronomical data from space telescopes like Kepler. It combines traditional transit detection methods with modern AI/ML techniques to efficiently process large volumes of light curve data.

The pipeline is designed to:

1. Download and process light curve data from astronomical repositories
2. Detect potential transit events using signal processing techniques
3. Analyze periodicity to identify orbital patterns
4. Estimate planet properties based on transit characteristics
5. Apply machine learning models to enhance detection capabilities
6. Generate comprehensive reports and visualizations

## Project Structure

The project is organized into several modules with clear responsibilities:

```
exoplanet_detection/
│
├── config.py                    # Configuration settings
├── main.py                      # Main entry point
│
├── data/                        # Data management
│   ├── __init__.py
│   ├── data_fetcher.py          # Data acquisition functions
│   ├── light_curve_processor.py # Light curve processing
│   └── dataset_generator.py     # Dataset creation for model training
│
├── detection/                   # Detection algorithms
│   ├── __init__.py
│   ├── transit_detector.py      # Transit detection algorithms
│   └── periodicity_analyzer.py  # Periodicity analysis functions
│
├── models/                      # ML models
│   ├── __init__.py
│   ├── cnn_model.py             # CNN model for transit detection
│   ├── multimodal_model.py      # Multimodal fusion model
│   └── model_trainer.py         # Model training utilities
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── visualization.py         # Visualization tools
│   ├── metrics.py               # Evaluation metrics
│   └── file_utils.py            # File operations
│
└── pipeline/                    # Pipeline components
    ├── __init__.py
    ├── pipeline_runner.py       # Main pipeline execution
    ├── result_analyzer.py       # Analysis of results
    └── report_generator.py      # Report generation
```

## Key Components and Features

### Data Management

- **Data Fetcher**: Downloads light curve data from astronomical repositories like MAST (Mikulski Archive for Space Telescopes)
- **Light Curve Processor**: Normalizes and preprocesses light curve data for analysis
- **Dataset Generator**: Creates training datasets for machine learning models

### Detection Algorithms

- **Transit Detector**: Implements algorithms to detect transit events in light curve data
- **Periodicity Analyzer**: Uses periodogram analysis to identify orbital periods
- **Transit Modeling**: Applies physical models to characterize transit events

### Machine Learning Models

- **CNN Model**: Convolutional Neural Network for image-based transit detection
- **Multimodal Model**: Fusion model combining image and time series data for improved detection
- **Model Trainer**: Utilities for training, evaluating, and optimizing models

### Pipeline Execution

- **Pipeline Runner**: Orchestrates the end-to-end pipeline execution
- **Result Analyzer**: Analyzes pipeline results for scientific insights
- **Report Generator**: Creates comprehensive HTML and JSON reports

## Key Optimizations

1. **Parallel Processing**: Uses ProcessPoolExecutor for parallel light curve processing
2. **Caching System**: Implements caching for downloaded data to avoid redundant operations
3. **Memory Efficiency**: Processes light curves individually to minimize memory usage
4. **Multimodal Fusion**: Combines different data representations for improved detection accuracy
5. **Resource Management**: Configurable worker count to adapt to available resources

## Usage

### Basic Usage

```bash
python main.py --max-records 100
```

### Using Synthetic Data

```bash
python main.py --synthetic-data
```

### Using Multimodal Model

```bash
python main.py --multimodal
```

### Hyperparameter Optimization

```bash
python main.py --optimize
```

## Development Notes

### Adding New Models

To add a new model:

1. Create a new model in the `models` directory
2. Implement the model building function following the pattern in existing models
3. Update `pipeline_runner.py` to include the new model in the training process

### Adding New Data Sources

To add a new data source:

1. Implement a new data fetcher in `data_fetcher.py`
2. Update the main pipeline to use the new data source

### Extending Visualizations

To add new visualizations:

1. Implement the visualization function in `visualization.py`
2. Update the report generator to include the new visualization

## Performance Considerations

- **Memory Usage**: The pipeline is designed to process one light curve at a time to minimize memory usage
- **Disk Space**: Downloaded data and results are cached, so ensure sufficient disk space
- **CPU Usage**: Parallel processing can utilize multiple CPU cores, configure `max_workers` accordingly
- **GPU Acceleration**: TensorFlow models can utilize GPU acceleration if available

***

### `documentation.md` (New and Comprehensive)

This file provides the detailed explanation of the project's methodology, terminology, and full usage instructions.

```markdown
# Exoplanet Detection Pipeline: Detailed Documentation

This document provides a comprehensive overview of the exoplanet detection pipeline, including its motivation, methodology, usage, and key terminology.

---

## 1. Motivation

The search for exoplanets is one of the most exciting frontiers in modern astronomy. The transit method, which detects the slight dimming of a star as a planet passes in front of it, has been incredibly successful but produces vast amounts of data. The primary challenge is sifting through millions of light curves to distinguish the faint, rare signals of true exoplanets from instrumental noise and astrophysical false positives.

This project was motivated by the need for an automated, reliable, and scalable tool to perform this classification. By leveraging deep learning, specifically a multimodal approach, we aim to create a model that can learn the subtle features of a true planetary transit more effectively than traditional algorithms, thereby accelerating the pace of exoplanet discovery.

---

## 2. Glossary of Terms

-   **Light Curve**: A graph of a star's brightness (flux) over time. This is the primary data source.
-   **Transit**: The event where an exoplanet passes in front of its host star from our point of view, causing a periodic dip in the star's light curve.
-   **False Positive**: A signal in a light curve that mimics a transit but is caused by other phenomena, such as an eclipsing binary star system or instrumental noise.
-   **Multimodal Model**: A neural network that accepts and processes multiple types of data (modalities) simultaneously. In this project, we use both the 1D time-series of the light curve and a 2D image representation of it.
-   **Class Imbalance**: A common problem in this domain where the number of non-planet examples (false positives, noise) vastly outnumbers the true planet examples.
-   **Precision**: A performance metric that answers: "Of all the candidates the model flagged as planets, what percentage were actually planets?" High precision is crucial for avoiding wasted follow-up observations.
-   **Recall**: A performance metric that answers: "Of all the real planets in the dataset, what percentage did the model successfully find?"
-   **Focal Loss**: A specialized loss function designed to handle class imbalance by focusing the model's training on harder-to-classify examples.
-   **Classification Threshold**: The probability value (between 0 and 1) used to convert a model's continuous output into a binary decision (planet vs. not a planet). A lower threshold increases recall, while a higher one increases precision.

---

## 3. Methodology & Pipeline Explanation

The pipeline is designed as a modular, end-to-end workflow. Here are the key stages:

### Stage 1: Data Ingestion
-   **Script**: `data/real_data_fetcher.py`
-   **Process**: The pipeline begins by scanning user-provided local directories for `.fits` files. It identifies files in the `planets_dir` as the positive class (1) and files in the `false_positives_dir` as the negative class (0).

### Stage 2: Data Preparation & Feature Engineering
-   **Script**: `data/dataset_generator.py`
-   **Process**: Each `.fits` file is loaded and preprocessed. A fixed-length segment of the normalized flux is extracted. This 1D segment serves two purposes:
    1.  It is used directly as the input for the 1D time-series branch of the model.
    2.  It is reshaped into a 2D image (e.g., 64x64) to serve as the input for the 2D image branch. This allows the model to learn spatial features from the transit's shape.
-   The processed data is saved as NumPy arrays (`X_images.npy`, `X_timeseries.npy`, `y_labels.npy`).

### Stage 3: Dataset Balancing
-   **Script**: `data/dataset_generator.py`
-   **Process**: Before training, the dataset is balanced using SMOTE (Synthetic Minority Over-sampling Technique). This creates synthetic examples of the minority class (planets) to prevent the model from becoming biased towards the majority class (false positives).

### Stage 4: Model Architecture
-   **Script**: `models/multimodal_model.py`
-   **Process**: The model is a multi-input neural network with two parallel branches:
    1.  **Image Branch**: A 2D Convolutional Neural Network (CNN) that processes the 2D image representation of the light curve to learn spatial features.
    2.  **Time-Series Branch**: A 1D CNN that processes the 1D light curve segment to learn temporal features.
-   The outputs of these two branches are flattened, concatenated, and passed through a series of dense layers to produce a final classification probability.

### Stage 5: Model Training
-   **Script**: `models/model_trainer.py`
-   **Process**: The model is trained using the prepared dataset. Several key techniques are employed:
    -   **Focal Loss**: Used to combat class imbalance.
    -   **Class Weights**: Also used to force the model to pay more attention to the minority (planet) class.
    -   **Adam Optimizer**: An efficient and standard optimizer for deep learning.
    -   **Callbacks**: `EarlyStopping` is used to prevent overfitting by stopping the training when validation performance no longer improves, and `ModelCheckpoint` saves the best version of the model.

### Stage 6: Evaluation & Reporting
-   **Scripts**: `pipeline/report_generator.py`, `utils/metrics.py`
-   **Process**: After training, the model's performance is evaluated on the held-out validation set. A detailed HTML report is generated, including key metrics (precision, recall, accuracy) and learning curve plots.

---

## 4. Full Usage Guide

### Setup
Ensure you have created a virtual environment and installed the dependencies from `requirements.txt`.

### A. Main Training Run
This is the primary workflow for training your best model using your full, curated dataset.

```bash
python main.py \
    --planets_dir /path/to/your/confirmed_planets \
    --false_positives_dir /path/to/your/false_positives
