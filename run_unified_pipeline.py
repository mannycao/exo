import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import time
import argparse

# Import from your project structure
from models.multimodal_model import build_unified_bnn

# Use TF Probability's alias for distributions
tfd = tfp.distributions

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. DUMMY IMPLEMENTATIONS & CONFIG (You will replace these)
# ==============================================================================

class UnifiedConfig:
    """A new config class for the unified model."""
    LC_IMAGE_SHAPE = (64, 64, 1)
    LC_TIMESERIES_SHAPE = (2048,)
    TTV_DIM = 12
    RV_DIM = 8
    LEARNING_RATE = 0.0001
    EPOCHS = 30
    BATCH_SIZE = 32
    NUM_MC_SAMPLES = 50

def negative_log_likelihood(y_true, y_pred_dist):
    """Calculates NLL from a distribution object."""
    return -tf.reduce_mean(y_pred_dist.log_prob(y_true))

def kl_divergence_function(y_true, y_pred_dist):
    """The KL loss is correctly handled by the model.losses property."""
    return sum(y_pred_dist.model.losses)

def load_unified_dataset(planets_dir, false_positives_dir, cfg):
    """Placeholder for your real data loading logic."""
    print("Generating dummy data for demonstration...")
    print(f"  (Simulating loading from planets_dir: {planets_dir})")
    print(f"  (Simulating loading from false_positives_dir: {false_positives_dir})")
    
    NUM_SAMPLES = 2000
    
    X_lc_timeseries = np.random.randn(NUM_SAMPLES, cfg.LC_TIMESERIES_SHAPE[0]).astype(np.float32)
    X_lc_image = np.random.randn(NUM_SAMPLES, *cfg.LC_IMAGE_SHAPE).astype(np.float32)
    X_ttv = np.random.randn(NUM_SAMPLES, cfg.TTV_DIM).astype(np.float32)
    X_rv = np.random.randn(NUM_SAMPLES, cfg.RV_DIM).astype(np.float32)
    y = np.random.randint(0, 2, size=NUM_SAMPLES).astype(np.float32)
    
    return (X_lc_timeseries, X_lc_image, X_ttv, X_rv), y

# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================

def main():
    """The main pipeline function."""
    parser = argparse.ArgumentParser(description="Run the unified multi-modal BNN pipeline.")
    parser.add_argument(
        '--planets_dir', type=str, required=True,
        help='Directory containing confirmed planet data.'
    )
    parser.add_argument(
        '--false_positives_dir', type=str, required=True,
        help='Directory containing false positive data.'
    )
    args = parser.parse_args()
    
    cfg = UnifiedConfig()

    print("Starting unified multi-modal BNN workflow...")
    
    (X_lc_ts, X_lc_img, X_ttv, X_rv), y = load_unified_dataset(
        planets_dir=args.planets_dir,
        false_positives_dir=args.false_positives_dir,
        cfg=cfg
    )
    
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    X_lc_ts, X_lc_img, X_ttv, X_rv, y = X_lc_ts[indices], X_lc_img[indices], X_ttv[indices], X_rv[indices], y[indices]

    num_test = int(len(y) * 0.2)
    num_train = len(y) - num_test
    
    X_train = [X_lc_ts[:-num_test], X_lc_img[:-num_test], X_ttv[:-num_test], X_rv[:-num_test]]
    y_train = y[:-num_test]
    X_test = [X_lc_ts[-num_test:], X_lc_img[-num_test:], X_ttv[-num_test:], X_rv[-num_test:]]
    y_test = y[-num_test:]
    
    print(f"\nData loaded. Training on {num_train} samples, testing on {num_test} samples.")

    # --- Build and Compile the Model ---
    model = build_unified_bnn(cfg, num_train)
    
    def total_loss(y_true, y_pred_dist):
        return negative_log_likelihood(y_true, y_pred_dist) + kl_divergence_function(y_true, y_pred_dist)
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=total_loss,
        metrics=['accuracy']
    )
    model.summary()
    
    # --- Train the model using model.fit() ---
    print("\nStarting training...")
    start_time = time.time()
    model.fit(
        X_train, y_train,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

    # --- Evaluate with Uncertainty ---
    print(f"\nPerforming inference with {cfg.NUM_MC_SAMPLES} Monte Carlo samples...")
    
    predictions_dist = [model(X_test, training=False) for _ in range(cfg.NUM_MC_SAMPLES)]
    probs = np.array([p.mean().numpy().flatten() for p in predictions_dist])

    mean_probs = np.mean(probs, axis=0)
    std_dev_probs = np.std(probs, axis=0)
    predicted_classes = (mean_probs > 0.5).astype(int)
    final_accuracy = np.mean(predicted_classes == y_test)
    print(f"\n✅ Final Accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()