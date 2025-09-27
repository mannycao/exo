import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Attempt to import config from parent directory if this file is in models/
try:
    from .. import config # Assumes model_trainer.py is in models/ and config.py is in project root
except ImportError:
    # Fallback for direct execution or different structure
    try:
        import config
    except ImportError:
        # Define a dummy config if absolutely necessary, though this indicates a setup issue
        class DummyConfig:
            EPOCHS = 50
            BATCH_SIZE = 32
            MODEL_DIR = Path("./models_trained") # Default if config cannot be loaded
            EARLY_STOPPING_PATIENCE = 10
            LEARNING_RATE = 1e-3
        config = DummyConfig()
        logging.error("Could not import config.py in model_trainer.py. Using dummy config.")

from models.model_trainer_utils import train_enhanced_model, visualize_enhanced_learning_curves, focal_loss

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir

    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None):
        # This method will call the train_enhanced_model function
        trained_model, history = train_enhanced_model(
            self.model,
            X_train, y_train,
            X_val, y_val,
            model_name="base_model", # Can be made configurable
            epochs=epochs,
            batch_size=batch_size,
            output_dir=self.output_dir
        )
        self.model = trained_model
        return history