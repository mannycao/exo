# run_fusion_pipeline.py

import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

# --- Add project root for imports ---
project_root = Path(__file__).resolve().parent
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_user import get_config
from utils.file_utils import setup_logging
from data.fusion_dataset_generator import create_fusion_dataset
from detection.periodicity_analyzer import clear_period_results, generate_period_statistics

from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate, Flatten, LeakyReLU, GaussianNoise, Conv2D, Add, MaxPooling2D, GlobalAveragePooling2D, Conv1D, MaxPooling1D, LSTM, Attention
)
from tensorflow.keras.models import Model
from models.bayesian_fusion_model import create_bayesian_fusion_model
from models.enhanced_trainer import focal_loss

def augment_data(X1, X2, y, alpha=0.2):
    # Mixup
    l = np.random.beta(alpha, alpha, X1.shape[0])
    X1_mix = l[:, None, None, None] * X1 + (1 - l)[:, None, None, None] * X1[::-1]
    X2_mix = l[:, None, None] * X2 + (1 - l)[:, None, None] * X2[::-1]
    y_mix = l * y + (1 - l) * y[::-1]

    # Random noise
    noise_factor = 0.05
    X1_mix += np.random.normal(0, noise_factor, X1_mix.shape)
    X2_mix += np.random.normal(0, noise_factor, X2_mix.shape)

    # Random shift
    shift_factor = 0.1
    shift = int(shift_factor * X1_mix.shape[1])
    X1_mix = np.roll(X1_mix, shift, axis=1)
    shift = int(shift_factor * X2_mix.shape[1])
    X2_mix = np.roll(X2_mix, shift, axis=1)

    return X1_mix, X2_mix, y_mix

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

def build_transit_branch(cfg):
    img_input = Input(shape=cfg.model.transit_image_shape, name='transit_image_input')
    ts_input = Input(shape=cfg.model.transit_ts_shape, name='transit_ts_input')
    # Image branch: Conv2D + ResNet blocks
    x1 = Conv2D(32, 3, padding='same', activation='relu')(img_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D()(x1)
    x1 = resnet_block(x1, 32, stride=1)
    x1 = MaxPooling2D()(x1)
    x1 = resnet_block(x1, 64, stride=2)
    x1 = MaxPooling2D()(x1)
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dropout(0.5)(x1)
    # Timeseries branch: 1D CNN + LSTM
    x2 = Conv1D(32, 5, padding='same', activation='relu')(ts_input)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D()(x2)
    x2 = Conv1D(64, 3, padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D()(x2)
    x2 = LSTM(32, return_sequences=True)(x2)
    x2 = LSTM(16)(x2)
    x2 = Dropout(0.5)(x2)
    # Simpler fusion approach
    concat = Concatenate(name='transit_features')([x1, x2])
    # Dense layers after fusion
    x = Dense(128)(concat)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return [img_input, ts_input], x

def main():
    """Main function to run the enhanced Bayesian Fusion Model pipeline with advanced training strategies."""
    parser = argparse.ArgumentParser(description='Run the Enhanced Bayesian Fusion Model pipeline.')
    parser.add_argument('--run_dir', type=str, default=f"results/fusion_run_{{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}}", help="Directory to save outputs.")
    parser.add_argument('--max_files', type=int, default=None, help="Limit the number of files for testing.")
    parser.add_argument('--use_mixed_precision', action='store_true', help="Enable mixed precision training")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size for training.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file_path=str(run_dir / 'fusion_pipeline.log'))
    logger = logging.getLogger(__name__)
    
    # Import Metal configuration
    import tf_metal_config
    
    # Enable mixed precision only if not using Metal backend
    if args.use_mixed_precision and not tf.config.list_physical_devices('GPU'):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Enabled mixed precision training on CPU")

    logger.info("--- Starting Bayesian Fusion Pipeline with Hybrid Training ---")

    cfg = get_config()
    ttv_dim = 7
    # Allow batch size override
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    # Optionally allow user to select modalities
    import os
    use_rv = bool(int(os.environ.get("USE_RV", "0")))
    use_imaging = bool(int(os.environ.get("USE_IMAGING", "0")))
    dataset = create_fusion_dataset(
        cfg.data.planets_dir,
        cfg.data.false_positives_dir,
        max_files=args.max_files,
        ttv_dim=ttv_dim,
        use_rv=use_rv,
        use_imaging=use_imaging
    )
    if dataset is None:
        logger.error("Failed to create dataset. Exiting.")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.data.random_seed)
    fold_reports = []

    y = dataset['y']
    # Log class distribution and more sample stats
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    logger.info(f"First 10 labels: {y[:10]}")
    logger.info(f"transit_image_input shape: {dataset['X_transit_image'].shape}")
    logger.info(f"transit_ts_input shape: {dataset['X_transit_timeseries'].shape}")
    # Ensure ttv_input is always a NumPy array before logging shape
    if not isinstance(dataset['X_ttv'], np.ndarray):
        dataset['X_ttv'] = np.array(dataset['X_ttv'])
    logger.info(f"ttv_input shape: {dataset['X_ttv'].shape}")

    # Always build X with all 7 model inputs, using zeros for missing modalities/masks
    n = dataset["X_transit_image"].shape[0]
    # Ensure ttv_input is always a NumPy array
    ttv_arr = dataset["X_ttv"]
    if not isinstance(ttv_arr, np.ndarray):
        ttv_arr = np.array(ttv_arr)

    X = {
        "transit_image_input": dataset["X_transit_image"],
        "transit_ts_input": dataset["X_transit_timeseries"],
        "rv_input": dataset["X_rv"] if use_rv else np.zeros((n, 50, 1), dtype=np.float32),
        "imaging_input": dataset["X_imaging"] if use_imaging else np.zeros((n, 2), dtype=np.float32),
        "rv_mask_input": dataset["rv_mask"] if use_rv else np.zeros((n, 1), dtype=np.float32),
        "imaging_mask_input": dataset["imaging_mask"] if use_imaging else np.zeros((n, 1), dtype=np.float32),
        "ttv_input": ttv_arr
    }
    # Log a few sample inputs for inspection
    logger.info(f"Sample transit_image_input[0] min/max: {X['transit_image_input'][0].min()}/{X['transit_image_input'][0].max()}")
    logger.info(f"Sample transit_ts_input[0] min/max: {X['transit_ts_input'][0].min()}/{X['transit_ts_input'][0].max()}")
    logger.info(f"Sample ttv_input[0]: {X['ttv_input'][0]}")
    logger.info(f"Sample rv_input[0] shape: {X['rv_input'][0].shape if 'rv_input' in X else 'N/A'}")
    logger.info(f"Sample imaging_input[0] shape: {X['imaging_input'][0].shape if 'imaging_input' in X else 'N/A'}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X['transit_image_input'], y)):
        logger.info(f"\n--- Fold {fold + 1}/5 ---")
        
        # --- Stage 1: Pre-train the Transit Branch ---
        logger.info("Stage 1: Pre-training Transit Branch on all data...")
        
        # Create the full model with correct num_train_samples for KL scaling
        num_train_samples = len(train_idx)
        # Dynamically select model inputs based on modalities
        model_inputs = ["transit_image_input", "transit_ts_input", "ttv_input"]
        if use_rv:
            model_inputs += ["rv_input", "rv_mask_input"]
        if use_imaging:
            model_inputs += ["imaging_input", "imaging_mask_input"]

        # Build transit branch for pretraining
        transit_inputs, transit_features = build_transit_branch(cfg)
        transit_head = Dense(1, activation='sigmoid')(transit_features)
        transit_pretrain_model = Model(inputs=transit_inputs, outputs=transit_head)

        # Use the original full model for fusion (user's model, but recommend to add Dropout/BatchNorm in its definition)
        full_model = create_bayesian_fusion_model(cfg, num_train_samples=num_train_samples, ttv_dim=ttv_dim)

        neg, pos = np.bincount(y[train_idx])
        alpha = neg / (pos + neg)
        # Compute class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y[train_idx]), y=y[train_idx])
        class_weight_dict = {i: w for i, w in zip(np.unique(y[train_idx]), class_weights)}
        
        # Use label smoothing and precision-recall AUC
        def smooth_labels(y, factor=0.1):
            y = y * (1 - factor) + 0.5 * factor
            return y
        from tensorflow.keras.metrics import AUC
        transit_pretrain_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

        # Simple validation split without mixup
        X_train_transit = {
            'transit_image_input': tf.cast(X['transit_image_input'][train_idx], tf.float32),
            'transit_ts_input': tf.cast(X['transit_ts_input'][train_idx], tf.float32)
        }
        y_train = tf.cast(y[train_idx], tf.float32)
        
        X_val_transit = {
            'transit_image_input': tf.cast(X['transit_image_input'][val_idx], tf.float32),
            'transit_ts_input': tf.cast(X['transit_ts_input'][val_idx], tf.float32)
        }
        y_val = tf.cast(y[val_idx], tf.float32)

        # Augment the training data
        X_train_transit['transit_image_input'], X_train_transit['transit_ts_input'], y_train_augmented = augment_data(
            X_train_transit['transit_image_input'], X_train_transit['transit_ts_input'], y_train
        )

        # Enhanced pre-training with early stopping and simpler learning schedule for Metal
        transit_pretrain_model.fit(
            X_train_transit, y_train_augmented,
            epochs=15,  # Reduced epochs for faster iterations
            batch_size=32,  # Fixed batch size for stability
            validation_data=(X_val_transit, y_val),
            verbose=1,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        )
            
        # --- Stage 2: Fine-tune the Fusion Layers ---
        logger.info("Stage 2: Fine-tuning Fusion Layers on multi-method subset...")
        # Transfer the learned weights from the pre-trained model to our full model
        for layer in transit_pretrain_model.layers:
            if layer.name in [l.name for l in full_model.layers]:
                full_model.get_layer(layer.name).set_weights(layer.get_weights())

        # Freeze the transit layers so their weights don't change
        for layer in full_model.layers:
            if 'transit' in layer.name or 'concatenate' == layer.name:
                layer.trainable = False
            
        # Enhanced loss function combining focal loss, KL divergence, and consistency regularization
        def enhanced_fusion_loss(y_true, y_pred):
            # Convert inputs to float32
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Focal loss with class balancing
            gamma = tf.constant(3.0, dtype=tf.float32)
            alpha_t = tf.where(tf.equal(y_true, 1), 
                             tf.constant(alpha, dtype=tf.float32),
                             tf.constant(1 - alpha, dtype=tf.float32))
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            fl = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0))
            fl = tf.reduce_mean(fl)
            
            # KL divergence with fixed weight
            kl = tf.add_n(full_model.losses) if full_model.losses else tf.constant(0.0, dtype=tf.float32)
            kl_weight = tf.constant(1.0 / (4.0 * num_train_samples), dtype=tf.float32)
            
            # Simple regularization term
            l2_reg = tf.reduce_mean(tf.square(y_pred))
            reg_weight = tf.constant(0.01, dtype=tf.float32)
            
            return fl + kl_weight * kl + reg_weight * l2_reg

        # Enhanced learning rate schedule with config-driven parameters
        initial_learning_rate = getattr(cfg.training, 'initial_learning_rate', cfg.training.learning_rate / 3)
        warmup_steps = getattr(cfg.training, 'warmup_steps', 50)
        decay_steps = getattr(cfg.training, 'decay_steps', 500)

        class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, warmup_steps=50, decay_steps=500):
                super().__init__()
                self.initial_lr = tf.cast(initial_lr, tf.float32)
                self.warmup_steps = warmup_steps
                self.decay_steps = decay_steps
                
            def get_config(self):
                return {
                    "initial_lr": float(self.initial_lr),
                    "warmup_steps": self.warmup_steps,
                    "decay_steps": self.decay_steps,
                }

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                warmup_steps = tf.cast(self.warmup_steps, tf.float32)
                decay_steps = tf.cast(self.decay_steps, tf.float32)
                initial_lr = tf.cast(self.initial_lr, tf.float32)

                # Linear warmup phase
                warmup_factor = tf.minimum(1.0, step / warmup_steps)
                warmup_lr = initial_lr * warmup_factor

                # Cosine decay phase
                step_after_warmup = step - warmup_steps
                cosine_factor = tf.cos(tf.constant(np.pi) * step_after_warmup / decay_steps)
                decay_factor = 0.5 * (1.0 + cosine_factor)
                decay_lr = initial_lr * tf.maximum(0.1, decay_factor)

                # Return warmup or decay learning rate based on step
                lr = tf.where(step < warmup_steps, warmup_lr, decay_lr)
                return lr

        lr_schedule = CustomLRSchedule(initial_learning_rate, warmup_steps=warmup_steps, decay_steps=decay_steps)

        # Enhanced metrics for better model evaluation
        def calibration_error(y_true, y_pred, num_bins=10):
            """Calculate Expected Calibration Error"""
            bin_boundaries = tf.linspace(0.0, 1.0, num_bins + 1)
            bin_indices = tf.cast(tf.floor(y_pred * num_bins), tf.int32)
            
            bin_counts = tf.zeros(num_bins, dtype=tf.int32)
            bin_sums = tf.zeros(num_bins, dtype=tf.float32)
            bin_true_sums = tf.zeros(num_bins, dtype=tf.float32)
            
            for i in range(num_bins):
                mask = tf.equal(bin_indices, i)
                bin_counts += tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)), tf.int32)
                bin_sums += tf.reduce_sum(tf.boolean_mask(y_pred, mask))
                bin_true_sums += tf.reduce_sum(tf.cast(tf.boolean_mask(y_true, mask), tf.float32))
            
            nonzero_bins = tf.cast(bin_counts > 0, tf.float32)
            bin_accuracies = tf.where(bin_counts > 0, bin_true_sums / tf.cast(bin_counts, tf.float32), 0.0)
            bin_confidences = tf.where(bin_counts > 0, bin_sums / tf.cast(bin_counts, tf.float32), 0.0)
            
            ece = tf.reduce_sum(tf.cast(bin_counts, tf.float32) / tf.reduce_sum(tf.cast(bin_counts, tf.float32)) *
                              tf.abs(bin_accuracies - bin_confidences) * nonzero_bins)
            return ece

        # Compile with standard Adam optimizer for stability
        # Use a lower initial learning rate and more stable optimizer config for Metal
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=True  # Enable AMSGrad for more stable training
        )
        
        full_model.compile(
            optimizer=optimizer,
            loss=enhanced_fusion_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        )

        # Find the subset of data that has either RV or Imaging data
        multi_method_mask_train = np.ones_like(y[train_idx], dtype=bool)
        if use_rv and use_imaging:
            multi_method_mask_train = (X['rv_mask_input'][train_idx].reshape(-1) > 0) | (X['imaging_mask_input'][train_idx].reshape(-1) > 0)
        elif use_rv:
            multi_method_mask_train = (X['rv_mask_input'][train_idx].reshape(-1) > 0)
        elif use_imaging:
            multi_method_mask_train = (X['imaging_mask_input'][train_idx].reshape(-1) > 0)

        if np.sum(multi_method_mask_train) > 10:
            selected_indices = np.where(multi_method_mask_train)[0]
            selected_train_idx = train_idx[selected_indices]
            # Always provide all 7 model inputs in correct order
            X_train_fine_tune = {
                'transit_image_input': X['transit_image_input'][selected_train_idx],
                'transit_ts_input': X['transit_ts_input'][selected_train_idx],
                'rv_input': X['rv_input'][selected_train_idx],
                'imaging_input': X['imaging_input'][selected_train_idx],
                'rv_mask_input': X['rv_mask_input'][selected_train_idx],
                'imaging_mask_input': X['imaging_mask_input'][selected_train_idx],
                'ttv_input': X['ttv_input'][selected_train_idx]
            }
            y_train_fine_tune = tf.cast(y[selected_train_idx], tf.float32)

            if isinstance(X_train_fine_tune, tuple):
                X_train_fine_tune = list(X_train_fine_tune)

            # Enhanced training with gradual unfreezing and progressive learning rates
            for phase in range(2):
                if phase == 1:  # Gradually unfreeze transit layers
                    for layer in full_model.layers:
                        if 'transit' in layer.name and 'BatchNormalization' not in layer.name:
                            layer.trainable = True
                
                full_model.fit(
                    X_train_fine_tune, y_train_fine_tune,
                    epochs=15 if phase == 0 else 10,  # More epochs in first phase
                    batch_size=cfg.training.batch_size,
                    verbose=1,
                    shuffle=True,  # Enable shuffling for better regularization
                    callbacks=[
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='loss',
                            factor=0.7,
                            patience=3,
                            min_lr=1e-6
                        )
                    ]
                )
        else:
            logger.warning("Skipping fine-tuning: Not enough multi-method data found in this fold.")

        # --- Evaluate the final, fine-tuned model ---
        X_val_full = {
            'transit_image_input': X['transit_image_input'][val_idx],
            'transit_ts_input': X['transit_ts_input'][val_idx],
            'rv_input': X['rv_input'][val_idx],
            'imaging_input': X['imaging_input'][val_idx],
            'rv_mask_input': X['rv_mask_input'][val_idx],
            'imaging_mask_input': X['imaging_mask_input'][val_idx],
            'ttv_input': X['ttv_input'][val_idx]
        }
        y_val_full = tf.cast(y[val_idx], tf.float32)
        if isinstance(X_val_full, tuple):
            X_val_full = list(X_val_full)

        y_pred_prob = full_model.predict(X_val_full)
        y_pred = (y_pred_prob > 0.5).astype(int)
        report = classification_report(y_val_full, y_pred, output_dict=True, zero_division=0)
        report['roc_auc'] = roc_auc_score(y_val_full, y_pred_prob)
        fold_reports.append(report)
        logger.info(f"Fold {fold + 1} Report:\n{classification_report(y_val_full, y_pred, zero_division=0)}")

    logger.info("\n" + "="*50)
    logger.info("Cross-Validation Finished. Final Averaged Report:")
    avg_report = pd.DataFrame([r['weighted avg'] for r in fold_reports]).mean().to_dict()
    avg_report['roc_auc'] = np.mean([r['roc_auc'] for r in fold_reports])
    logger.info(pd.Series(avg_report).to_string())
    logger.info("="*50)
    
    # Generate period statistics report
    logger.info("\nGenerating Period Analysis Report...")
    stats_dir = run_dir / "period_analysis"
    period_stats = generate_period_statistics(output_dir=str(stats_dir))
    if period_stats:
        logger.info("\nPeriod Analysis Summary:")
        logger.info(f"Total periods analyzed: {period_stats['period_statistics']['n_total']}")
        logger.info(f"Valid periods detected: {period_stats['period_statistics']['n_valid']}")
        logger.info(f"Mean period: {period_stats['period_statistics']['mean_period']:.2f} days")
        logger.info(f"Median period: {period_stats['period_statistics']['median_period']:.2f} days")
        
        reliability = period_stats['reliability_analysis']
        logger.info("\nReliability Analysis:")
        logger.info(f"Mean reliability score: {reliability['mean_reliability']:.3f}")
        logger.info(f"Median reliability score: {reliability['median_reliability']:.3f}")
        percentiles = reliability['reliability_percentiles']
        logger.info(f"Reliability percentiles:")
        logger.info(f"  10th: {percentiles['10th']:.3f}")
        logger.info(f"  25th: {percentiles['25th']:.3f}")
        logger.info(f"  75th: {percentiles['75th']:.3f}")
        logger.info(f"  90th: {percentiles['90th']:.3f}")
    
    # Clear period results for next run
    clear_period_results()

if __name__ == "__main__":
    main()
