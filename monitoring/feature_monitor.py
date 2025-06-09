"""
Module for monitoring feature drift and data quality.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Any, Optional, Tuple
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import linalg, stats

class FeatureMonitor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le moniteur de features.
        
        Args:
            config: Configuration du monitoring
        """
        self.config = config.get('features', {})
        self.alert_threshold = self.config.get('alert_threshold', 0.7)
        self.window_size = self.config.get('window_size', 1000)
        self.drift_thresholds = self.config.get('drift_thresholds', {
            'mean_shift': 0.3,
            'std_shift': 0.3,
            'correlation': 1.0,
            'missing_rate': 0.1,
            'mahalanobis': 1.0,
            'default': 0.5
        })
        self.reference_stats = None
        self.scaler = StandardScaler()
        
    def _robust_covariance(self, X: np.ndarray, support_fraction: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robust covariance using a custom implementation that avoids determinant warnings.
        
        Args:
            X: Scaled feature matrix
            support_fraction: Fraction of points to use for robust estimation
            
        Returns:
            Tuple of (location, covariance)
        """
        n_samples, n_features = X.shape
        n_support = int(n_samples * support_fraction)
        
        # Initial location estimate using median
        location = np.median(X, axis=0)
        
        # Compute initial covariance using Ledoit-Wolf
        lw = LedoitWolf(assume_centered=True)
        lw.fit(X - location)
        covariance = lw.covariance_
        
        # Iterative reweighting
        max_iter = 30
        for _ in range(max_iter):
            # Compute Mahalanobis distances
            try:
                inv_cov = linalg.inv(covariance)
                distances = np.array([np.dot(np.dot(x - location, inv_cov), x - location) 
                                    for x in X])
            except linalg.LinAlgError:
                # If matrix is singular, add regularization
                reg = 1e-6 * np.eye(n_features)
                inv_cov = linalg.inv(covariance + reg)
                distances = np.array([np.dot(np.dot(x - location, inv_cov), x - location) 
                                    for x in X])
                
            # Compute weights using chi-square distribution
            weights = 1.0 / (1.0 + distances / n_features)
            
            # Sort weights and keep only the best support_fraction
            sorted_indices = np.argsort(weights)[::-1]
            weights = np.zeros_like(weights)
            weights[sorted_indices[:n_support]] = 1.0
            
            # Update location and covariance
            new_location = np.average(X, weights=weights, axis=0)
            centered = X - new_location
            new_covariance = np.cov(centered.T, aweights=weights)
            
            # Add small regularization
            new_covariance += 1e-6 * np.eye(n_features)
            
            # Check convergence
            if (np.allclose(location, new_location, rtol=1e-4) and 
                np.allclose(covariance, new_covariance, rtol=1e-4)):
                break
                
            location = new_location
            covariance = new_covariance
            
        return location, covariance
        
    def _compute_covariance(self, X: np.ndarray) -> Optional[Any]:
        """
        Compute covariance with multiple estimators and fallback options.
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Covariance estimator or None if all attempts fail
        """
        # First try Ledoit-Wolf estimator which is more stable
        try:
            logger.info("Attempting Ledoit-Wolf covariance estimation")
            lw = LedoitWolf(assume_centered=True)
            lw.fit(X)
            logger.info("Successfully computed Ledoit-Wolf covariance")
            return lw
        except Exception as e:
            logger.warning(f"Ledoit-Wolf estimation failed: {str(e)}")
            
        # Then try our custom robust implementation
        try:
            logger.info("Attempting custom robust covariance estimation")
            location, covariance = self._robust_covariance(X)
            
            # Create estimator-like object
            class RobustEstimator:
                def __init__(self, location, covariance):
                    self.location_ = location
                    self.covariance_ = covariance
                    
                def mahalanobis(self, X):
                    try:
                        inv_cov = linalg.inv(self.covariance_)
                        return np.array([np.sqrt(np.dot(np.dot(x - self.location_, inv_cov), 
                                                      x - self.location_)) 
                                       for x in X])
                    except linalg.LinAlgError:
                        # If matrix is singular, add regularization
                        reg = 1e-6 * np.eye(self.covariance_.shape[0])
                        inv_cov = linalg.inv(self.covariance_ + reg)
                        return np.array([np.sqrt(np.dot(np.dot(x - self.location_, inv_cov), 
                                                      x - self.location_)) 
                                       for x in X])
                        
            estimator = RobustEstimator(location, covariance)
            logger.info("Successfully computed custom robust covariance")
            return estimator
            
        except Exception as e:
            logger.warning(f"Custom robust covariance estimation failed: {str(e)}")
            
        # Finally, fall back to empirical covariance with shrinkage
        try:
            logger.info("Falling back to empirical covariance with shrinkage")
            emp_cov = EmpiricalCovariance(assume_centered=True)
            emp_cov.fit(X)
            
            # Add small regularization to ensure positive definiteness
            n_features = X.shape[1]
            reg = 1e-6 * np.eye(n_features)
            emp_cov.covariance_ = emp_cov.covariance_ + reg
            
            logger.info("Successfully computed empirical covariance")
            return emp_cov
        except Exception as e:
            logger.error(f"All covariance estimation attempts failed: {str(e)}")
            return None
            
    def compute_reference_stats(self, features: pd.DataFrame) -> None:
        """
        Compute reference statistics for drift detection.
        
        Args:
            features (pd.DataFrame): Reference feature matrix
        """
        self.reference_stats = {}
        
        # Compute basic statistics
        for col in features.columns:
            ref_data = features[col].dropna()
            if len(ref_data) == 0:
                logger.warning(f"Column {col} has no valid data for reference statistics")
                continue
                
            self.reference_stats[col] = {
                'mean': ref_data.mean(),
                'std': ref_data.std(),
                'q1': ref_data.quantile(0.25),
                'q3': ref_data.quantile(0.75),
                'iqr': ref_data.quantile(0.75) - ref_data.quantile(0.25)
            }
            
        # Compute correlation matrix
        self.reference_stats['correlation'] = features.corr()
        
        # Compute covariance for Mahalanobis distance
        try:
            # Scale the features
            X_scaled = self.scaler.fit_transform(features)
            
            # Get covariance estimator
            cov_estimator = self._compute_covariance(X_scaled)
            
            if cov_estimator is not None:
                self.reference_stats['robust_covariance'] = cov_estimator
                self.reference_stats['robust_location'] = cov_estimator.location_
            else:
                logger.error("Could not compute any form of covariance")
                self.reference_stats['robust_covariance'] = None
                
        except Exception as e:
            logger.error(f"Error in covariance computation: {str(e)}")
            self.reference_stats['robust_covariance'] = None
            
        logger.info("Statistiques de référence calculées")
        
    def compute_drift_metrics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute drift metrics for current features.
        
        Args:
            features (pd.DataFrame): Current feature matrix
            
        Returns:
            dict: Drift metrics for each feature
        """
        if self.reference_stats is None:
            raise ValueError("Reference statistics not computed. Call compute_reference_stats first.")
            
        drift_metrics = {}
        
        for col in features.columns:
            if col not in self.reference_stats:
                continue
                
            current_data = features[col].dropna()
            if len(current_data) == 0:
                logger.warning(f"Column {col} has no valid data for drift computation")
                continue
                
            ref_stats = self.reference_stats[col]
            
            # Compute drift metrics with safe division
            mean_shift = abs(current_data.mean() - ref_stats['mean'])
            std_shift = abs(current_data.std() - ref_stats['std'])
            
            # Safe IQR shift computation
            current_q1 = current_data.quantile(0.25)
            current_q3 = current_data.quantile(0.75)
            current_iqr = current_q3 - current_q1
            
            if ref_stats['iqr'] != 0 and current_iqr != 0:
                iqr_shift = abs(current_iqr - ref_stats['iqr']) / ref_stats['iqr']
            else:
                iqr_shift = 0.0
                
            # Missing rate
            missing_rate = features[col].isna().mean()
            
            drift_metrics[col] = {
                'mean_shift': mean_shift,
                'std_shift': std_shift,
                'iqr_shift': iqr_shift,
                'missing_rate': missing_rate
            }
            
        # Compute correlation drift
        current_corr = features.corr()
        corr_diff = np.abs(current_corr - self.reference_stats['correlation'])
        drift_metrics['correlation_drift'] = corr_diff.mean().mean()
        
        # Compute Mahalanobis distance if robust covariance is available
        if self.reference_stats['robust_covariance'] is not None:
            try:
                X_scaled = self.scaler.transform(features)
                mahalanobis_dist = self.reference_stats['robust_covariance'].mahalanobis(X_scaled)
                drift_metrics['mahalanobis_drift'] = np.mean(mahalanobis_dist)
            except Exception as e:
                logger.warning(f"Could not compute Mahalanobis drift: {str(e)}")
                drift_metrics['mahalanobis_drift'] = None
                
        return drift_metrics
        
    def check_drift(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for feature drift and return alerts.
        
        Args:
            features (pd.DataFrame): Current feature matrix
            
        Returns:
            dict: Drift alerts and metrics
        """
        drift_metrics = self.compute_drift_metrics(features)
        alerts = {}
        
        for col, metrics in drift_metrics.items():
            if col == 'correlation_drift' or col == 'mahalanobis_drift':
                continue
                
            alerts[col] = {
                'drift_detected': False,
                'metrics': metrics,
                'thresholds': self.drift_thresholds
            }
            
            # Check each metric against thresholds
            for metric_name, value in metrics.items():
                threshold = self.drift_thresholds.get(metric_name, self.drift_thresholds['default'])
                if value > threshold:
                    alerts[col]['drift_detected'] = True
                    alerts[col]['alert_message'] = f"{metric_name} drift detected: {value:.3f} > {threshold:.3f}"
                    
        # Check global metrics
        if 'correlation_drift' in drift_metrics:
            corr_threshold = self.drift_thresholds.get('correlation', self.drift_thresholds['default'])
            if drift_metrics['correlation_drift'] > corr_threshold:
                alerts['correlation'] = {
                    'drift_detected': True,
                    'alert_message': f"Correlation drift detected: {drift_metrics['correlation_drift']:.3f} > {corr_threshold:.3f}"
                }
                
        if 'mahalanobis_drift' in drift_metrics and drift_metrics['mahalanobis_drift'] is not None:
            mahal_threshold = self.drift_thresholds.get('mahalanobis', self.drift_thresholds['default'])
            if drift_metrics['mahalanobis_drift'] > mahal_threshold:
                alerts['mahalanobis'] = {
                    'drift_detected': True,
                    'alert_message': f"Mahalanobis drift detected: {drift_metrics['mahalanobis_drift']:.3f} > {mahal_threshold:.3f}"
                }
                
        return {
            'alerts': alerts,
            'metrics': drift_metrics
        }
        
    def fit(self, features: pd.DataFrame) -> None:
        """
        Fit the feature monitor on reference data.
        
        Args:
            features (pd.DataFrame): Reference feature matrix
        """
        self.compute_reference_stats(features)
        
    def update(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Update drift detection with new data.
        
        Args:
            features (pd.DataFrame): New feature matrix
            
        Returns:
            dict: Drift detection results
        """
        return self.check_drift(features) 