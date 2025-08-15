"""
Advanced period validation using statistical tests and machine learning.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import tensorflow as tf
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class PeriodValidator:
    """Advanced period validation using multiple approaches."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination='auto',
            random_state=42,
            n_estimators=100
        )
        self.statistical_tests = {
            'anderson_darling': self._anderson_darling_test,
            'shapiro_wilk': self._shapiro_wilk_test,
            'kolmogorov_smirnov': self._ks_test,
            'durbins_watson': self._durbins_watson_test
        }
        
    def _extract_features(self, result: Dict) -> np.ndarray:
        """Extract features for anomaly detection."""
        features = []
        
        # Basic metrics
        features.extend([
            result.get('median_period', 0),
            result.get('confidence', 0),
            result.get('n_transits', 0),
            result.get('data_span', 0)
        ])
        
        # Quality metrics
        quality = result.get('quality_metrics', {})
        features.extend([
            quality.get('power_snr', 0),
            quality.get('peak_snr', 0),
            quality.get('fap', 1.0),
            quality.get('quality_score', 0)
        ])
        
        # Period/span ratio
        if result.get('data_span', 0) > 0:
            features.append(result.get('median_period', 0) / result.get('data_span', 1))
        else:
            features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def _anderson_darling_test(self, data: np.ndarray) -> Tuple[float, float]:
        """Perform Anderson-Darling test for normality."""
        try:
            result = stats.anderson(data)
            return result.statistic, result.critical_values[-1]
        except:
            return np.inf, np.inf
    
    def _shapiro_wilk_test(self, data: np.ndarray) -> Tuple[float, float]:
        """Perform Shapiro-Wilk test for normality."""
        try:
            statistic, pvalue = stats.shapiro(data)
            return statistic, pvalue
        except:
            return 0, 1
    
    def _ks_test(self, data: np.ndarray) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test against normal distribution."""
        try:
            statistic, pvalue = stats.kstest(data, 'norm')
            return statistic, pvalue
        except:
            return 1, 1
    
    def _durbins_watson_test(self, data: np.ndarray) -> float:
        """Perform Durbin-Watson test for autocorrelation."""
        try:
            diff = np.diff(data)
            dw_statistic = np.sum(diff * diff) / np.sum(data * data)
            return dw_statistic
        except:
            return 2.0  # Return null hypothesis value
    
    def train(self, results: List[Dict]) -> None:
        """Train the validator on a set of period detection results."""
        if not results:
            return
        
        # Extract features for all results
        features = []
        for result in results:
            features.append(self._extract_features(result).flatten())
        
        features = np.array(features)
        if len(features) < 2:
            return
            
        # Fit scaler and isolation forest
        self.scaler.fit(features)
        normalized_features = self.scaler.transform(features)
        self.isolation_forest.fit(normalized_features)
    
    def validate(self, result: Dict) -> Dict:
        """
        Perform comprehensive validation of a period detection result.
        
        Args:
            result: Dictionary containing period detection results
            
        Returns:
            Dictionary containing validation results
        """
        validation = {
            'is_valid': True,
            'confidence': 0.0,
            'anomaly_score': 0.0,
            'statistical_tests': {},
            'warnings': []
        }
        
        # Extract time series data
        transit_times = np.array(result.get('transit_times', []))
        if len(transit_times) < 3:
            validation['is_valid'] = False
            validation['warnings'].append("Insufficient transit times")
            return validation
        
        # Normalize and scale features
        features = self._extract_features(result)
        normalized_features = self.scaler.transform(features)
        
        # Anomaly detection
        anomaly_score = self.isolation_forest.score_samples(normalized_features)[0]
        validation['anomaly_score'] = float(anomaly_score)
        if anomaly_score < -0.5:  # Conservative threshold
            validation['warnings'].append("Possible anomalous detection")
        
        # Run statistical tests
        transit_intervals = np.diff(transit_times)
        for test_name, test_func in self.statistical_tests.items():
            if test_name == 'durbins_watson':
                validation['statistical_tests'][test_name] = float(test_func(transit_intervals))
            else:
                statistic, pvalue = test_func(transit_intervals)
                validation['statistical_tests'][test_name] = {
                    'statistic': float(statistic),
                    'pvalue': float(pvalue)
                }
        
        # Check for problematic period ratios
        period = result.get('median_period', 0)
        data_span = result.get('data_span', 0)
        if data_span > 0:
            period_ratio = period / data_span
            if period_ratio > 0.5:
                validation['warnings'].append(f"Period ({period:.1f} days) is large relative to data span ({data_span:.1f} days)")
        
        # Calculate overall confidence
        base_confidence = result.get('confidence', 0)
        quality_score = result.get('quality_metrics', {}).get('quality_score', 0)
        anomaly_factor = np.clip((anomaly_score + 1) / 2, 0, 1)  # Convert to 0-1 scale
        
        validation['confidence'] = np.mean([
            base_confidence,
            quality_score,
            anomaly_factor
        ])
        
        # Final validity check
        if validation['confidence'] < 0.3:
            validation['is_valid'] = False
            validation['warnings'].append("Low overall confidence")
        if len(validation['warnings']) > 2:
            validation['is_valid'] = False
        
        return validation

class DeepPeriodValidator:
    """Neural network-based period validation."""
    
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model."""
        inputs = []
        
        # Time series input
        ts_input = tf.keras.layers.Input(shape=(None,), name='transit_times')
        ts_masked = tf.keras.layers.Masking(mask_value=0.)(ts_input)
        ts_lstm = tf.keras.layers.LSTM(32, return_sequences=True)(ts_masked)
        ts_lstm = tf.keras.layers.LSTM(16)(ts_lstm)
        inputs.append(ts_input)
        
        # Metadata input
        meta_input = tf.keras.layers.Input(shape=(5,), name='metadata')
        meta_dense = tf.keras.layers.Dense(16, activation='relu')(meta_input)
        inputs.append(meta_input)
        
        # Combine features
        combined = tf.keras.layers.Concatenate()([ts_lstm, meta_dense])
        x = tf.keras.layers.Dense(32, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Multiple outputs
        validity = tf.keras.layers.Dense(1, activation='sigmoid', name='validity')(x)
        confidence = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=[validity, confidence])
        model.compile(
            optimizer='adam',
            loss={
                'validity': 'binary_crossentropy',
                'confidence': 'mse'
            },
            metrics={
                'validity': ['accuracy', tf.keras.metrics.AUC()],
                'confidence': ['mae']
            }
        )
        return model
    
    def _prepare_data(self, result: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for the neural network."""
        # Prepare time series data
        transit_times = np.array(result.get('transit_times', []))
        if len(transit_times) == 0:
            return None, None
            
        # Normalize transit times to 0-1 range
        if len(transit_times) >= 2:
            t_min, t_max = transit_times.min(), transit_times.max()
            transit_times = (transit_times - t_min) / (t_max - t_min)
        
        # Prepare metadata
        metadata = np.array([
            result.get('median_period', 0),
            result.get('confidence', 0),
            result.get('n_transits', 0),
            result.get('data_span', 0),
            result.get('quality_metrics', {}).get('quality_score', 0)
        ])
        
        return transit_times, metadata
    
    def train(self, results: List[Dict], labels: List[bool], confidences: List[float]) -> None:
        """Train the neural network on labeled data."""
        if not results or len(results) < 10:
            return
            
        # Prepare training data
        transit_times_list = []
        metadata_list = []
        max_length = 0
        
        for result in results:
            transit_times, metadata = self._prepare_data(result)
            if transit_times is None:
                continue
            max_length = max(max_length, len(transit_times))
            transit_times_list.append(transit_times)
            metadata_list.append(metadata)
        
        # Pad sequences
        padded_times = tf.keras.preprocessing.sequence.pad_sequences(
            transit_times_list,
            maxlen=max_length,
            padding='post',
            value=0.
        )
        
        metadata_array = np.array(metadata_list)
        labels_array = np.array(labels)
        confidences_array = np.array(confidences)
        
        # Train the model
        self.model.fit(
            [padded_times, metadata_array],
            [labels_array, confidences_array],
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
    
    def validate(self, result: Dict) -> Dict:
        """
        Perform neural network-based validation.
        
        Args:
            result: Dictionary containing period detection results
            
        Returns:
            Dictionary containing validation results
        """
        transit_times, metadata = self._prepare_data(result)
        if transit_times is None:
            return {
                'nn_validity': 0.0,
                'nn_confidence': 0.0
            }
        
        # Prepare input data
        padded_times = tf.keras.preprocessing.sequence.pad_sequences(
            [transit_times],
            maxlen=100,  # Fixed length for prediction
            padding='post',
            value=0.
        )
        metadata_array = np.array([metadata])
        
        # Make prediction
        validity, confidence = self.model.predict([padded_times, metadata_array])
        
        return {
            'nn_validity': float(validity[0][0]),
            'nn_confidence': float(confidence[0][0])
        }
