# minimal_model_debug_type.py
import tensorflow as tf
import tensorflow_probability as tfp
from types import SimpleNamespace

tfd = tfp.distributions

cfg = SimpleNamespace()
cfg.model = SimpleNamespace()
cfg.model.transit_image_shape = (64, 64, 1)
cfg.model.transit_ts_shape = (200, 1)
cfg.model.rv_shape = (50, 1)
cfg.model.imaging_shape = (10,)
cfg.model.dropout_rate = 0.3

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

x2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(transit_image_input)
x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
x2 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x2)
x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x2)
x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
x2 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x2)
x2_flat = tf.keras.layers.GlobalAveragePooling2D()(x2)

transit_features = tf.keras.layers.Concatenate(name='transit_features')([x1_flat, x2_flat])

x3 = tf.keras.layers.LSTM(32, return_sequences=True)(rv_input)
x3 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x3)
x3 = tf.keras.layers.LSTM(32)(x3)
x3 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x3)
rv_features = x3
rv_features_masked = tf.keras.layers.Multiply()([rv_features, rv_mask_input])

x4 = tf.keras.layers.Dense(16, activation='relu')(imaging_input)
x4 = tf.keras.layers.Dropout(cfg.model.dropout_rate)(x4)
imaging_features = tf.keras.layers.Dense(8, activation='relu')(x4)
imaging_features_masked = tf.keras.layers.Multiply()([imaging_features, imaging_mask_input])

fused = tf.keras.layers.Concatenate(name='fused_features')([transit_features, rv_features_masked, imaging_features_masked])

print('Type of fused:', type(fused))
print('Shape of fused:', getattr(fused, 'shape', None))

kl_fn = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(200, dtype=tf.float32)
try:
    x = tfp.layers.DenseFlipout(128, kernel_divergence_fn=kl_fn, activation='relu', name='denseflipout_1')(fused)
    print('DenseFlipout output shape:', x.shape)
except Exception as e:
    print('Error in DenseFlipout:', e)
