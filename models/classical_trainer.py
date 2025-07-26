# models/classical_trainer.py

import logging
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

def train_classical_model(feature_csv_path, output_dir):
    """
    Trains a tuned RandomForest Classifier on the engineered feature dataset.
    """
    logger.info(f"Loading feature dataset from {feature_csv_path}")
    df = pd.read_csv(feature_csv_path).dropna()

    if 'label' not in df.columns or len(df) < 10:
        logger.error("Dataset is invalid or too small for training.")
        return None

    features = df.drop(columns=['label', 'source_file'])
    labels = df['label']

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info("Performing hyperparameter tuning for RandomForest Classifier...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1_weighted', verbose=1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    logger.info("Classification Report on Validation Set (Tuned Model):\n" + classification_report(y_val, y_pred))

    model_path = os.path.join(output_dir, "tuned_random_forest_model.joblib")
    joblib.dump(best_model, model_path)
    logger.info(f"Tuned classical model saved to: {model_path}")
    
    return {"model": best_model, "metrics": report}