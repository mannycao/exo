import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        
    def call(self, inputs):
        attn_output = self.attention(inputs, inputs)
        x = self.add([inputs, attn_output])
        x = self.layernorm(x)
        return x

class CrossModalAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = tf.keras.layers.Attention()
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        
    def call(self, x1, x2):
        attn_output = self.attention([x1, x2])
        x = self.add([x1, attn_output])
        x = self.layernorm(x)
        return x

def build_unified_bnn(cfg, num_train_samples):
    """
    Builds an enhanced multi-modal Bayesian Neural Network with attention mechanisms
    and cross-modal learning.
    """
    # --- Define Inputs for all four data streams ---
    timeseries_input = tf.keras.layers.Input(shape=cfg.LC_TIMESERIES_SHAPE, name='timeseries_input')
    image_input = tf.keras.layers.Input(shape=cfg.LC_IMAGE_SHAPE, name='image_input')
    ttv_input = tf.keras.layers.Input(shape=(cfg.TTV_DIM,), name='ttv_input')
    rv_input = tf.keras.layers.Input(shape=(cfg.RV_DIM,), name='rv_input')

    # --- Timeseries Branch (now fully trainable) ---
    x_ts = tf.keras.layers.Reshape((cfg.LC_TIMESERIES_SHAPE[0], 1), name='reshape')(timeseries_input)
    x_ts = tf.keras.layers.Conv1D(16, 5, activation='relu', name='conv1d')(x_ts)
    x_ts = tf.keras.layers.BatchNormalization(name='batch_normalization_2')(x_ts)
    x_ts = tf.keras.layers.MaxPooling1D(pool_size=2, name='max_pooling1d')(x_ts)
    x_ts = tf.keras.layers.Conv1D(32, 5, activation='relu', name='conv1d_1')(x_ts)
    x_ts = tf.keras.layers.BatchNormalization(name='batch_normalization_3')(x_ts)
    x_ts = tf.keras.layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1')(x_ts)
    x_ts = tf.keras.layers.Flatten(name='flatten_1')(x_ts)
    x_ts = tf.keras.layers.Dense(32, activation='relu', name='dense_1')(x_ts)
    features_ts = tf.keras.layers.Dropout(0.2, name='dropout_1')(x_ts)

    # --- Enhanced Image Branch with Residual Connections ---
    x_img = tf.keras.layers.Conv2D(32, (3,3), activation='relu', name='conv2d')(image_input)
    x_img = tf.keras.layers.BatchNormalization(name='batch_normalization')(x_img)
    x_img_res = x_img  # Save for residual connection
    
    x_img = tf.keras.layers.Conv2D(32, (3,3), activation='relu', name='conv2d_1')(x_img)
    x_img = tf.keras.layers.BatchNormalization(name='batch_normalization_1')(x_img)
    x_img = tf.keras.layers.Add()([x_img, x_img_res])  # Residual connection
    x_img = tf.keras.layers.MaxPooling2D(pool_size=2, name='max_pooling2d')(x_img)
    
    x_img = tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2d_2')(x_img)
    x_img = tf.keras.layers.BatchNormalization(name='batch_normalization_2')(x_img)
    x_img_res2 = x_img
    
    x_img = tf.keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2d_3')(x_img)
    x_img = tf.keras.layers.BatchNormalization(name='batch_normalization_3')(x_img)
    x_img = tf.keras.layers.Add()([x_img, x_img_res2])
    x_img = tf.keras.layers.MaxPooling2D(pool_size=2, name='max_pooling2d_1')(x_img)
    
    x_img = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x_img)
    features_img = tf.keras.layers.Dense(64, activation='relu', name='dense_img')(x_img)
    
    # --- Enhanced TTV Processing with Uncertainty-Aware Learning ---
    ttv_uncertainty = tf.keras.layers.Dense(32, activation='sigmoid', name='ttv_uncertainty')(ttv_input)
    ttv_features = tf.keras.layers.Dense(32, activation='relu', name='ttv_features')(ttv_input)
    ttv_features = tf.keras.layers.Multiply()([ttv_features, ttv_uncertainty])
    features_ttv = tf.keras.layers.Dense(32, activation='relu', name='ttv_output')(ttv_features)
    
    # --- Enhanced RV Processing with Signal Quality Assessment ---
    rv_quality = tf.keras.layers.Dense(32, activation='sigmoid', name='rv_quality')(rv_input)
    rv_features = tf.keras.layers.Dense(32, activation='relu', name='rv_features')(rv_input)
    rv_features = tf.keras.layers.Multiply()([rv_features, rv_quality])
    features_rv = tf.keras.layers.Dense(32, activation='relu', name='rv_output')(rv_features)

    # --- Advanced Cross-Modal Fusion with Attention ---
    # 1. Cross-attention between timeseries and image features
    ts_img_attention = CrossModalAttention(64)([features_ts, features_img])
    
    # 2. Cross-attention between TTV and RV features
    ttv_rv_attention = CrossModalAttention(32)([features_ttv, features_rv])
    
    # 3. Hierarchical fusion with learned weights
    modality_weights = tf.keras.layers.Dense(4, activation='softmax', name='modality_weights')(
        tf.keras.layers.Concatenate()([
            tf.keras.layers.GlobalAveragePooling1D()(ts_img_attention),
            tf.keras.layers.GlobalAveragePooling1D()(ttv_rv_attention)
        ])
    )
    
    # Weighted fusion of all modalities
    weighted_ts_img = tf.keras.layers.Multiply()([ts_img_attention, modality_weights[:, 0:1]])
    weighted_ttv_rv = tf.keras.layers.Multiply()([ttv_rv_attention, modality_weights[:, 1:2]])
    
    # Final fusion with deep dense layers
    concatenated = tf.keras.layers.Concatenate(name='concatenate')(
        [weighted_ts_img, weighted_ttv_rv]
    )
    
    # Enhanced Bayesian inference with mixture density network
    kl_fn = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(num_train_samples, dtype=tf.float32)
    
    # Deeper Bayesian network with skip connections
    x1 = tfp.layers.DenseFlipout(256, kernel_divergence_fn=kl_fn, activation='relu')(concatenated)
    x2 = tfp.layers.DenseFlipout(128, kernel_divergence_fn=kl_fn, activation='relu')(x1)
    x3 = tfp.layers.DenseFlipout(64, kernel_divergence_fn=kl_fn, activation='relu')(x2)
    
    # Skip connections
    x3_with_skip = tf.keras.layers.Concatenate()([x3, x1])
    
    # Mixture density network for better uncertainty estimation
    mixture_weights = tfp.layers.DenseFlipout(3, kernel_divergence_fn=kl_fn, activation='softmax')(x3_with_skip)
    mixture_means = tfp.layers.DenseFlipout(3, kernel_divergence_fn=kl_fn)(x3_with_skip)
    mixture_scales = tfp.layers.DenseFlipout(3, kernel_divergence_fn=kl_fn, activation='softplus')(x3_with_skip)
    
    # Create mixture distribution
    components = [
        tfd.Normal(loc=mixture_means[:, i:i+1], scale=mixture_scales[:, i:i+1])
        for i in range(3)
    ]
    mixture_dist = tfd.Mixture(
        cat=tfd.Categorical(probs=mixture_weights),
        components=components
    )
    
    # Convert to binary prediction
    logits = tfp.layers.DenseFlipout(1, kernel_divergence_fn=kl_fn)(mixture_dist.mean())
    output_distribution = tfp.layers.IndependentBernoulli(1, tfd.Bernoulli, name='output_dist')(logits)
    
    # Create the enhanced model
    model = tf.keras.Model(
        inputs=[timeseries_input, image_input, ttv_input, rv_input],
        outputs=[output_distribution, mixture_dist.mean(), mixture_dist.stddev()],
        name="Enhanced_Multi_Modal_BNN"
    )
    
    print("✅ Unified, end-to-end trainable functional model defined.")
    return model