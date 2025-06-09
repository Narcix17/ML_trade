"""
Module for market regime detection using clustering and robust covariance estimation.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from numpy import linalg
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import mlflow
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

from monitoring.feature_monitor import FeatureMonitor
from monitoring.model_monitor import ModelMonitor

# Configuration du détecteur de régimes
regime_config = {
    'n_regimes': 3,
    'random_state': 42,  # Add random_state
    'covariance_type': 'full',
    'n_init': 10,
    'max_iter': 100,
    'tol': 1e-3,
    'support_fraction': 0.7
}

class MarketRegimeDetector:
    """Détecteur de régimes de marché."""
    
    def __init__(self, config):
        """
        Initialize the market regime detector.
        
        Args:
            config (dict): Configuration dictionary containing regime detection parameters
        """
        self.config = config.get('market_regime', {})
        self.n_regimes = self.config.get('n_regimes', 5)
        self.support_fraction = self.config.get('support_fraction', 0.7)  # Increased from 0.501
        self.random_state = self.config.get('random_state', 42)
        self.scaler = StandardScaler()
        self.feature_monitor = FeatureMonitor(config)
        self.model = None
        self.model_monitor = ModelMonitor(config)
        self.regime_centers = None
        self.last_detection_metrics = None
        self.columns_ = None  # Pour sauvegarder les colonnes utilisées lors du fit
        
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
            return None
        
    def detect_regimes(self, features: pd.DataFrame) -> tuple:
        """
        Detect market regimes using robust covariance estimation and GMM.
        
        Args:
            features (pd.DataFrame): Feature matrix
            
        Returns:
            tuple: (regime_labels, regime_metrics)
        """
        # Sauvegarder les colonnes pour validation future
        self.columns_ = features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Get robust covariance estimator
        cov_estimator = self._compute_covariance(X_scaled)
        
        if cov_estimator is None:
            logger.error("Could not compute any form of covariance")
            return pd.Series(0, index=features.index), None
            
        # Compute Mahalanobis distances
        distances = cov_estimator.mahalanobis(X_scaled)
        
        # Use GMM to cluster regimes based on distances
        gmm = GaussianMixture(n_components=self.n_regimes, 
                             random_state=self.random_state,
                             n_init=10)
        
        # Reshape distances for GMM
        distances_2d = distances.reshape(-1, 1)
        
        # Fit GMM and predict regimes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(distances_2d)
            regime_labels = gmm.predict(distances_2d)
            
        # Create regime series
        regimes = pd.Series(regime_labels, index=features.index)
        
        # Log regime distribution
        regime_counts = regimes.value_counts()
        logger.info(f"Detected regimes distribution: {regime_counts.to_dict()}")
        
        # Calculate regime metrics
        metrics = {
            'silhouette_score': silhouette_score(X_scaled, regime_labels),
            'calinski_harabasz_score': calinski_harabasz_score(X_scaled, regime_labels),
            'n_regimes': self.n_regimes,
            'regime_sizes': [np.sum(regime_labels == i) for i in range(self.n_regimes)]
        }
        
        # Mise à jour du monitoring
        self.model_monitor.update_metrics(
            metrics,
            'market_regime',
            datetime.now()
        )
        
        # Vérification des performances
        alerts = self.model_monitor.check_performance(
            metrics,
            'market_regime'
        )
        
        if alerts:
            logger.warning(f"Alertes de performance détectées: {len(alerts)}")
            for alert in alerts:
                logger.warning(f"Alerte: {alert}")
            
        # Log dans MLflow
        self.model_monitor.log_to_mlflow(
            metrics,
            'market_regime',
            f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Sauvegarde des métriques et des centres
        self.last_detection_metrics = metrics
        self.regime_centers = gmm.means_
        
        return regimes, metrics
        
    def predict_regime(
        self,
        features: pd.DataFrame,
        check_drift: bool = True,
        check_performance: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Prédit le régime de marché.
        
        Args:
            features: Features pour la prédiction
            check_drift: Vérifier le drift des features
            check_performance: Vérifier la performance
            
        Returns:
            Tuple (labels des régimes, métriques)
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné")
            
        # Validation des colonnes
        if self.columns_ is not None:
            if not features.columns.equals(pd.Index(self.columns_)):
                raise ValueError("Les colonnes d'entrée ne correspondent pas à celles utilisées lors du fit.")
        else:
            logger.warning("Aucune information sur les colonnes d'entraînement disponible")
            
        # Vérification du drift
        if check_drift:
            drift_alerts = self.feature_monitor.detect_drift(features)
            if drift_alerts:
                logger.warning(f"Drift détecté: {len(drift_alerts)} alertes")
                self.feature_monitor.log_to_mlflow(
                    drift_alerts,
                    f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
        # Normalisation et prédiction
        X = self.scaler.transform(features)
        regime_labels = self.model.predict(X)
        
        # Calcul des métriques si on a les labels réels
        if hasattr(features, 'regime'):
            metrics = {
                'accuracy': accuracy_score(features.regime, regime_labels),
                'adjusted_rand_score': adjusted_rand_score(features.regime, regime_labels),
                'n_regimes': len(np.unique(regime_labels)),
                'regime_sizes': np.bincount(regime_labels).tolist()
            }
            
            if check_performance:
                alerts = self.model_monitor.check_performance(
                    metrics,
                    'market_regime'
                )
                
                if alerts:
                    logger.warning(f"Alertes de performance détectées: {len(alerts)}")
                    self.model_monitor.log_to_mlflow(
                        metrics,
                        'market_regime',
                        f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
        else:
            metrics = {
                'n_regimes': len(np.unique(regime_labels)),
                'regime_sizes': np.bincount(regime_labels).tolist()
            }
            
        return regime_labels, metrics
        
    def get_regime_characteristics(self, features: pd.DataFrame, regime_labels: np.ndarray) -> pd.DataFrame:
        """
        Calculate characteristics for each regime.
        
        Args:
            features (pd.DataFrame): Feature matrix
            regime_labels (np.ndarray): Array of regime labels
            
        Returns:
            pd.DataFrame: Characteristics of each regime
        """
        characteristics = []
        for regime in range(self.n_regimes):
            mask = regime_labels == regime
            regime_features = features[mask]
            
            stats = {
                'regime': regime,
                'size': len(regime_features),
                'proportion': len(regime_features) / len(features)
            }
            
            # Add mean and std for each feature
            for col in features.columns:
                stats[f'{col}_mean'] = regime_features[col].mean()
                stats[f'{col}_std'] = regime_features[col].std()
                stats[f'{col}_min'] = regime_features[col].min()
                stats[f'{col}_max'] = regime_features[col].max()
            
            characteristics.append(stats)
        
        return pd.DataFrame(characteristics)
        
    def save_state(self, path: str) -> None:
        """
        Sauvegarde l'état du détecteur.
        
        Args:
            path: Chemin de sauvegarde
        """
        import joblib
        
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'regime_centers': self.regime_centers,
            'last_detection_metrics': self.last_detection_metrics,
            'feature_monitor': self.feature_monitor,
            'model_monitor': self.model_monitor
        }
        
        joblib.dump(state, path)
        logger.info(f"État du détecteur sauvegardé: {path}")
        
    def load_state(self, path: str) -> None:
        """
        Charge l'état du détecteur.
        
        Args:
            path: Chemin de chargement
        """
        import joblib
        
        try:
            state = joblib.load(path)
            self.model = state['model']
            self.scaler = state['scaler']
            self.regime_centers = state['regime_centers']
            self.last_detection_metrics = state['last_detection_metrics']
            self.feature_monitor = state['feature_monitor']
            self.model_monitor = state['model_monitor']
            logger.info(f"État du détecteur chargé: {path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {e}")
            raise 