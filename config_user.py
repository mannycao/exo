# config_user.py
# User config for exoplanet pipeline. Provides get_config for pipeline compatibility.

from types import SimpleNamespace

# Example shapes and parameters. Adjust as needed for your data.
def get_config():
    import json
    import os
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.model.transit_image_shape = (64, 64, 1)
    cfg.model.transit_ts_shape = (200, 1)
    cfg.model.rv_shape = (50, 1)
    cfg.model.imaging_shape = (10,)
    # Default values
    cfg.model.dropout_rate = 0.5
    cfg.model.l2_regularization = 0.1
    cfg.training = SimpleNamespace()
    cfg.training.learning_rate = 0.00001 # Changed learning rate
    cfg.training.batch_size = 64
    cfg.training.epochs = 15
    # Try to load best hyperparameters from results/best_hp_params.json
    best_hp_path = os.path.join(os.path.dirname(__file__), 'results', 'best_hp_params.json')
    if os.path.exists(best_hp_path):
        try:
            with open(best_hp_path, 'r') as f:
                best_hp = json.load(f)
            params = best_hp.get('best_params', {})
            # Override if present
            if 'dropout_rate' in params:
                cfg.model.dropout_rate = params['dropout_rate']
            if 'l2_regularization' in params:
                cfg.model.l2_regularization = params['l2_regularization']
            if 'learning_rate' in params:
                cfg.training.learning_rate = params['learning_rate']
            if 'batch_size' in params:
                cfg.training.batch_size = params['batch_size']
            if 'epochs' in params:
                cfg.training.epochs = params['epochs']
        except Exception as e:
            print(f"[WARN] Could not load best hyperparameters: {e}")
    cfg.data = SimpleNamespace()
    cfg.data.planets_dir = '../exo-main/kepler_local_data/confirmed_planets/'
    cfg.data.false_positives_dir = '../exo-main/kepler_local_data/false_positives/'
    cfg.data.random_seed = 42
    return cfg