# minimal_model_debug.py
import tensorflow as tf
import tensorflow_probability as tfp
from types import SimpleNamespace
from models.bayesian_fusion_model import create_bayesian_fusion_model

def print_type_and_shape(name, tensor):
    print(f"{name}: type={type(tensor)}, shape={getattr(tensor, 'shape', None)}")

cfg = SimpleNamespace()
cfg.model = SimpleNamespace()
cfg.model.transit_image_shape = (64, 64, 1)
cfg.model.transit_ts_shape = (200, 1)
cfg.model.rv_shape = (50, 1)
cfg.model.imaging_shape = (10,)
cfg.model.dropout_rate = 0.3

# Build model up to the fusion layer, printing types and shapes
inputs = [
    tf.keras.Input(shape=cfg.model.transit_image_shape, name='transit_image_input'),
    tf.keras.Input(shape=cfg.model.transit_ts_shape, name='transit_ts_input'),
    tf.keras.Input(shape=cfg.model.rv_shape, name='rv_input'),
    tf.keras.Input(shape=cfg.model.imaging_shape, name='imaging_input'),
    tf.keras.Input(shape=(1,), name='rv_mask_input'),
    tf.keras.Input(shape=(1,), name='imaging_mask_input'),
]
transit_image_input, transit_ts_input, rv_input, imaging_input, rv_mask_input, imaging_mask_input = inputs

x1 = tf.keras.layers.Conv1D(32, 5, activation='relu')(transit_ts_input)
x1 = tf.keras.layers.MaxPooling1D(2)(x1)
x1 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x1)
x1 = tf.keras.layers.Conv1D(64, 5, activation='relu')(x1)
x1 = tf.keras.layers.MaxPooling1D(2)(x1)
x1 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x1)
x1_flat = tf.keras.layers.Flatten()(x1)
print_type_and_shape('x1_flat', x1_flat)

x2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(transit_image_input)
x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
x2 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x2)
x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x2)
x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
x2 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x2)
x2_flat = tf.keras.layers.GlobalAveragePooling2D()(x2)
print_type_and_shape('x2_flat', x2_flat)

transit_features = tf.keras.layers.Concatenate(name='transit_features')([x1_flat, x2_flat])
print_type_and_shape('transit_features', transit_features)

x3 = tf.keras.layers.LSTM(32, return_sequences=True)(rv_input)
x3 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x3)
x3 = tf.keras.layers.LSTM(32)(x3)
x3 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x3)
rv_features = x3
print_type_and_shape('rv_features', rv_features)
rv_features_masked = tf.keras.layers.Multiply()([rv_features, rv_mask_input])
print_type_and_shape('rv_features_masked', rv_features_masked)

x4 = tf.keras.layers.Dense(16, activation='relu')(imaging_input)
x4 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x4)
imaging_features = tf.keras.layers.Dense(8, activation='relu')(x4)
print_type_and_shape('imaging_features', imaging_features)
imaging_features_masked = tf.keras.layers.Multiply()([imaging_features, imaging_mask_input])
print_type_and_shape('imaging_features_masked', imaging_features_masked)

fused = tf.keras.layers.Concatenate(name='fused_features')([transit_features, rv_features_masked, imaging_features_masked])
print_type_and_shape('fused', fused)
