# minimal_model_test.py
import tensorflow as tf
import tensorflow_probability as tfp
from types import SimpleNamespace
from models.bayesian_fusion_model import create_bayesian_fusion_model

# Dummy config matching your model config
cfg = SimpleNamespace()
cfg.model = SimpleNamespace()
cfg.model.transit_image_shape = (64, 64, 1)
cfg.model.transit_ts_shape = (200, 1)
cfg.model.rv_shape = (50, 1)
cfg.model.imaging_shape = (10,)
cfg.model.dropout_rate = 0.3

# Create the model
model = create_bayesian_fusion_model(cfg, num_train_samples=200)

# Print model summary
model.summary()

# Print input and output shapes
print('Input shapes:')
for inp in model.inputs:
    print(f'  {inp.name}: {inp.shape}')
print('Output shape:', model.output.shape)

# Try a forward pass with dummy data
import numpy as np
batch_size = 2
x_img = np.random.rand(batch_size, *cfg.model.transit_image_shape).astype(np.float32)
x_ts = np.random.rand(batch_size, *cfg.model.transit_ts_shape).astype(np.float32)
x_rv = np.random.rand(batch_size, *cfg.model.rv_shape).astype(np.float32)
x_img_feat = np.random.rand(batch_size, *cfg.model.imaging_shape).astype(np.float32)
x_rv_mask = np.ones((batch_size, 1), dtype=np.float32)
x_img_mask = np.ones((batch_size, 1), dtype=np.float32)

try:
    y = model([x_img, x_ts, x_rv, x_img_feat, x_rv_mask, x_img_mask])
    print('Forward pass output shape:', y.shape)
except Exception as e:
    print('Error during forward pass:', e)
