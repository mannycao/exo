import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dropout,
    Conv2D, MaxPooling2D, Concatenate, LSTM, Multiply, Layer, Dense, Add, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
tfd = tfp.distributions

class DenseFlipoutWrapper(Layer):
    def __init__(self, units, activation, kernel_divergence_fn, bias_divergence_fn, dtype, name):
        super(DenseFlipoutWrapper, self).__init__()
        self.dense_flipout = tfp.layers.DenseFlipout(
            units=units,
            activation=activation,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            dtype=dtype,
            name=name
        )

    def call(self, x):
        return self.dense_flipout(x)

def resnet_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same', activation=None, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    input_channels = int(shortcut.shape[-1])
    if stride != 1 or input_channels != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])
    x = LeakyReLU()(x)
    return x

def create_bayesian_fusion_model(cfg, num_train_samples=1000, ttv_dim=7):
    print("[DEBUG] cfg.model.transit_image_shape:", cfg.model.transit_image_shape, type(cfg.model.transit_image_shape))
    print("[DEBUG] cfg.model.transit_ts_shape:", cfg.model.transit_ts_shape, type(cfg.model.transit_ts_shape))
    print("[DEBUG] cfg.model.rv_shape:", cfg.model.rv_shape, type(cfg.model.rv_shape))
    print("[DEBUG] cfg.model.dropout_rate:", cfg.model.dropout_rate, type(cfg.model.dropout_rate))

    print("[DEBUG] Transit Image Shape:", cfg.model.transit_image_shape)
    print("[DEBUG] Transit TS Shape:", cfg.model.transit_ts_shape)
    print("[DEBUG] RV Shape:", cfg.model.rv_shape)
    print("[DEBUG] Dropout Rate:", cfg.model.dropout_rate)

    transit_image_input = Input(shape=cfg.model.transit_image_shape, name='transit_image_input', dtype=tf.float32)
    transit_ts_input = Input(shape=cfg.model.transit_ts_shape, name='transit_ts_input', dtype=tf.float32)

    x1 = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(transit_ts_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(cfg.model.dropout_rate)(x1)
    x1 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(cfg.model.dropout_rate)(x1)
    x1 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)

    x2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(transit_image_input)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = resnet_block(x2, 64)
    x2 = resnet_block(x2, 64)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = resnet_block(x2, 128, stride=2)
    x2 = resnet_block(x2, 128)
    x2 = GlobalAveragePooling2D()(x2)
    
    transit_features = Concatenate(name='transit_features')([x1, x2])
    transit_features = Dense(64, activation='relu', kernel_regularizer=l2(cfg.model.l2_regularization))(transit_features)
    print("[DEBUG] x1_flat type:", type(x1), "shape:", getattr(x1, 'shape', None))
    print("[DEBUG] x2_flat type:", type(x2), "shape:", getattr(x2, 'shape', None))
    print("[DEBUG] transit_features type:", type(transit_features), "shape:", getattr(transit_features, 'shape', None))

    rv_input = Input(shape=cfg.model.rv_shape, name='rv_input', dtype=tf.float32)
    rv_mask_input = Input(shape=(1,), name='rv_mask_input', dtype=tf.float32)
    
    x3 = LSTM(64, return_sequences=True)(rv_input)
    x3 = Dropout(cfg.model.dropout_rate)(x3)
    x3 = LSTM(64)(x3)
    x3 = Dropout(cfg.model.dropout_rate)(x3)
    rv_features = x3
    rv_features = Dense(64, activation='relu', kernel_regularizer=l2(cfg.model.l2_regularization))(rv_features)
    rv_features_masked = Multiply()([rv_features, rv_mask_input])
    print("[DEBUG] rv_features type:", type(rv_features), "shape:", getattr(rv_features, 'shape', None))
    print("[DEBUG] rv_mask_input type:", type(rv_mask_input), "shape:", getattr(rv_mask_input, 'shape', None))
    print("[DEBUG] rv_features_masked type:", type(rv_features_masked), "shape:", getattr(rv_features_masked, 'shape', None))

    imaging_input = Input(shape=(2,), name='imaging_input', dtype=tf.float32)
    imaging_mask_input = Input(shape=(1,), name='imaging_mask_input', dtype=tf.float32)
    
    x4 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(cfg.model.l2_regularization))(imaging_input)
    x4 = Dropout(cfg.model.dropout_rate)(x4)
    imaging_features = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(cfg.model.l2_regularization))(x4)
    imaging_features_masked = Multiply()([imaging_features, imaging_mask_input])
    print("[DEBUG] imaging_features type:", type(imaging_features), "shape:", getattr(imaging_features, 'shape', None))
    print("[DEBUG] imaging_mask_input type:", type(imaging_mask_input), "shape:", getattr(imaging_mask_input, 'shape', None))
    print("[DEBUG] imaging_features_masked type:", type(imaging_features_masked), "shape:", getattr(imaging_features_masked, 'shape', None))

    ttv_input = Input(shape=(ttv_dim,), name='ttv_input', dtype=tf.float32)
    ttv_features = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(cfg.model.l2_regularization))(ttv_input)
    ttv_features = Dropout(cfg.model.dropout_rate)(ttv_features)

    # Attention-based fusion
    attention_probs = Dense(4, activation='softmax', name='attention_probs')(Concatenate()([transit_features, rv_features_masked, imaging_features_masked, ttv_features]))
    transit_weight = attention_probs[:, 0:1]
    rv_weight = attention_probs[:, 1:2]
    imaging_weight = attention_probs[:, 2:3]
    ttv_weight = attention_probs[:, 3:4]

    fused = Add()([
        Multiply()([transit_features, transit_weight]),
        Multiply()([rv_features_masked, rv_weight]),
        Multiply()([imaging_features_masked, imaging_weight]),
        Multiply()([ttv_features, ttv_weight])
    ])

    print("[DEBUG] fused type:", type(fused), "shape:", getattr(fused, 'shape', None))

    def kl_divergence_fn(q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(num_train_samples, dtype=tf.float32)

    x = DenseFlipoutWrapper(
        units=256,
        activation='relu',
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        dtype=tf.float32,
        name='denseflipout_1'
    )(fused)
    x = Dropout(0.5)(x)
    
    x = DenseFlipoutWrapper(
        units=128,
        activation='relu',
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        dtype=tf.float32,
        name='denseflipout_2'
    )(x)
    x = Dropout(0.5)(x)
    
    logits = DenseFlipoutWrapper(
        units=1,
        activation=None,
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        dtype=tf.float32,
        name='logits'
    )(x)
    
    output = tf.keras.layers.Activation('sigmoid', dtype=tf.float32, name='output')(logits)

    inputs = {
        'transit_image_input': transit_image_input,
        'transit_ts_input': transit_ts_input,
        'rv_input': rv_input,
        'imaging_input': imaging_input,
        'rv_mask_input': rv_mask_input,
        'imaging_mask_input': imaging_mask_input,
        'ttv_input': ttv_input
    }
    
    model = Model(
        inputs=inputs,
        outputs=output,
        name="BayesianFusionModel"
    )

    return model
