import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import time
import argparse

# Use TF Probability's alias for distributions
tfd = tfp.distributions

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. DUMMY IMPLEMENTATIONS & CONFIG
# ==============================================================================

class DummyConfig:
    """Simulates your config.py file."""
    LC_IMAGE_SHAPE = (64, 64, 1)
    LC_TIMESERIES_SHAPE = (2048,)
    TTV_DIM = 12
    RV_DIM = 8
    NUM_SAMPLES = 2000 
    TEST_SPLIT_FRACTION = 0.2
    LEARNING_RATE = 0.0001
    EPOCHS = 30 
    BATCH_SIZE = 32
    NUM_MC_SAMPLES = 50
    GRADIENT_CLIP_NORM = 1.0
    KL_WEIGHT = 0.01 

def load_and_preprocess_data(cfg):
    """Simulates data loading."""
    print("Generating dummy data for demonstration...")
    X_lc_timeseries = np.random.randn(cfg.NUM_SAMPLES, cfg.LC_TIMESERIES_SHAPE[0]).astype(np.float32)
    X_lc_image = np.random.randn(cfg.NUM_SAMPLES, *cfg.LC_IMAGE_SHAPE).astype(np.float32)
    X_ttv = np.random.randn(cfg.NUM_SAMPLES, cfg.TTV_DIM).astype(np.float32)
    X_rv = np.random.randn(cfg.NUM_SAMPLES, cfg.RV_DIM).astype(np.float32)
    y = np.random.randint(0, 2, size=cfg.NUM_SAMPLES).astype(np.float32)
    return (X_lc_timeseries, X_lc_image, X_ttv, X_rv), y

# ==============================================================================
# 2. MODEL DEFINITION (NEW UNIFIED, END-TO-END APPROACH)
# ==============================================================================

def negative_log_likelihood(y_true, logits):
    """Calculates NLL from logits and labels."""
    y_true = tf.cast(y_true, logits.dtype)
    dist = tfd.Independent(tfd.Bernoulli(logits=logits), reinterpreted_batch_ndims=1)
    return -tf.reduce_mean(dist.log_prob(y_true))

class UnifiedMultiModalBNN(tf.keras.Model):
    def __init__(self, cfg):
        super(UnifiedMultiModalBNN, self).__init__()
        self.cfg = cfg
        
        # --- Define ALL layers of the model from scratch ---
        
        # Timeseries Branch Layers (now trainable)
        self.reshape_ts = tf.keras.layers.Reshape((cfg.LC_TIMESERIES_SHAPE[0], 1))
        self.conv1d = tf.keras.layers.Conv1D(16, 5, activation='relu')
        self.bn_ts1 = tf.keras.layers.BatchNormalization()
        self.pool_ts1 = tf.keras.layers.MaxPooling1D(2)
        self.conv1d_1 = tf.keras.layers.Conv1D(32, 5, activation='relu')
        self.bn_ts2 = tf.keras.layers.BatchNormalization()
        self.pool_ts2 = tf.keras.layers.MaxPooling1D(2)
        self.flatten_ts = tf.keras.layers.Flatten()
        self.dense_ts = tf.keras.layers.Dense(32, activation='relu')
        self.dropout_ts = tf.keras.layers.Dropout(0.2)

        # Image Branch Layers (now trainable)
        self.conv2d = tf.keras.layers.Conv2D(16, (3,3), activation='relu')
        self.bn_img1 = tf.keras.layers.BatchNormalization()
        self.pool_img1 = tf.keras.layers.MaxPooling2D(2)
        self.conv2d_1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.bn_img2 = tf.keras.layers.BatchNormalization()
        self.pool_img2 = tf.keras.layers.MaxPooling2D(2)
        self.flatten_img = tf.keras.layers.Flatten()
        self.dense_img = tf.keras.layers.Dense(32, activation='relu')
        self.dropout_img = tf.keras.layers.Dropout(0.2)
        
        # TTV and RV MLPs
        self.ttv_mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(cfg.TTV_DIM,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        self.rv_mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(cfg.RV_DIM,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        
        # Fusion and Bayesian Head
        self.concatenate = tf.keras.layers.Concatenate()
        self.bnn_dense1 = tfp.layers.DenseFlipout(128, activation='relu')
        self.bnn_dense2 = tfp.layers.DenseFlipout(64, activation='relu')
        self.bnn_logits = tfp.layers.DenseFlipout(1)
        
        print("✅ Unified, end-to-end trainable model defined.")

    def call(self, inputs):
        """Defines the forward pass."""
        input_timeseries, input_image, input_ttv, input_rv = inputs
        
        x_ts = self.reshape_ts(input_timeseries)
        x_ts = self.conv1d(x_ts)
        x_ts = self.bn_ts1(x_ts)
        x_ts = self.pool_ts1(x_ts)
        x_ts = self.conv1d_1(x_ts)
        x_ts = self.bn_ts2(x_ts)
        x_ts = self.pool_ts2(x_ts)
        x_ts = self.flatten_ts(x_ts)
        x_ts = self.dense_ts(x_ts)
        features_ts = self.dropout_ts(x_ts)
        
        x_img = self.conv2d(input_image)
        x_img = self.bn_img1(x_img)
        x_img = self.pool_img1(x_img)
        x_img = self.conv2d_1(x_img)
        x_img = self.bn_img2(x_img)
        x_img = self.pool_img2(x_img)
        x_img = self.flatten_img(x_img)
        x_img = self.dense_img(x_img)
        features_img = self.dropout_img(x_img)
        
        features_ttv = self.ttv_mlp(input_ttv)
        features_rv = self.rv_mlp(input_rv)
        
        concatenated_features = self.concatenate(
            [features_ts, features_img, features_ttv, features_rv]
        )
        
        x = self.bnn_dense1(concatenated_features)
        x = self.bnn_dense2(x)
        logits = self.bnn_logits(x)
        
        return logits
    
    def kl_loss(self):
        """
        Manually calculates the KL divergence from the Bayesian layers.
        This is the most robust way to ensure the KL loss is calculated.
        """
        return (self.bnn_dense1.kernel_divergence +
                self.bnn_dense2.kernel_divergence +
                self.bnn_logits.kernel_divergence)

# ==============================================================================
# 3. MAIN EXECUTION WITH CUSTOM TRAINING LOOP
# ==============================================================================

def main():
    """The main pipeline function."""
    cfg = DummyConfig()

    print("Starting unified multi-modal BNN workflow...")
    
    (X_lc_ts, X_lc_img, X_ttv, X_rv), y = load_and_preprocess_data(cfg)
    
    num_test = int(cfg.NUM_SAMPLES * cfg.TEST_SPLIT_FRACTION)
    num_train = cfg.NUM_SAMPLES - num_test
    
    X_train = [X_lc_ts[:-num_test], X_lc_img[:-num_test], X_ttv[:-num_test], X_rv[:-num_test]]
    y_train = y[:-num_test]
    X_test = [X_lc_ts[-num_test:], X_lc_img[-num_test:], X_ttv[-num_test:], X_rv[-num_test:]]
    y_test = y[-num_test:]
    
    print(f"\nData loaded. Training on {num_train} samples, testing on {num_test} samples.")

    # --- Build the model, optimizer, and metrics ---
    model = UnifiedMultiModalBNN(cfg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
    
    # --- The Custom Training Loop ---
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            nll = negative_log_likelihood(labels, logits)
            
            # *** THIS IS THE FIX ***
            # Manually get the KL loss from our dedicated method.
            kl_loss = model.kl_loss()
            
            # Scale the KL loss
            scaled_kl_loss = kl_loss / num_train * cfg.KL_WEIGHT
            
            total_loss = nll + scaled_kl_loss
        
        gradients = tape.gradient(total_loss, model.trainable_variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, cfg.GRADIENT_CLIP_NORM)
        optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
        return nll, scaled_kl_loss

    print("\nStarting custom training loop...")
    start_time = time.time()
    for epoch in range(cfg.EPOCHS):
        print(f"Epoch {epoch + 1}/{cfg.EPOCHS}")
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train[0], X_train[1], X_train[2], X_train[3]), y_train)).batch(cfg.BATCH_SIZE)
        
        epoch_nll_loss = []
        epoch_kl_loss = []
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            nll, kl = train_step(x_batch, y_batch)
            epoch_nll_loss.append(nll.numpy())
            epoch_kl_loss.append(kl.numpy())
        
        # Print average loss for the epoch
        print(f"  Avg NLL Loss: {np.mean(epoch_nll_loss):.4f}, Avg KL Loss: {np.mean(epoch_kl_loss):.4f}, Avg Total Loss: {np.mean(epoch_nll_loss) + np.mean(epoch_kl_loss):.4f}")
    
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

    # --- Evaluate with Uncertainty ---
    print(f"\nPerforming inference with {cfg.NUM_MC_SAMPLES} Monte Carlo samples...")
    X_test_tuple = (X_test[0], X_test[1], X_test[2], X_test[3])
    
    logits_samples = [model(X_test_tuple, training=False) for _ in range(cfg.NUM_MC_SAMPLES)]
    probs = tf.nn.sigmoid(tf.stack(logits_samples)).numpy()

    mean_probs = np.mean(probs, axis=0).flatten()
    std_dev_probs = np.std(probs, axis=0).flatten()
    predicted_classes = (mean_probs > 0.5).astype(int)
    final_accuracy = np.mean(predicted_classes == y_test)
    print(f"\n✅ Final Accuracy: {final_accuracy:.4f}")

    print("\n--- Example Predictions with Uncertainty ---")
    print("True | Pred | Mean Prob | Uncertainty (Std Dev) | Confidence")
    print("------------------------------------------------------------------")
    for i in range(15):
        uncertainty = std_dev_probs[i]
        confidence_char = "✅ High" if uncertainty < 0.1 else ("🤔 Medium" if uncertainty < 0.3 else "❓ Low")
        print(f"  {int(y_test[i])}  |   {predicted_classes[i]}  |   {mean_probs[i]:.3f}   |  {uncertainty:.3f}             | {confidence_char}")

if __name__ == "__main__":
    main()