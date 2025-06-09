"""
Tests du système de monitoring.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
from sklearn.model_selection import train_test_split

from monitoring.feature_monitor import FeatureMonitor
from monitoring.model_monitor import ModelMonitor
from models.ml_model import MLModel
from models.market_regime import MarketRegimeDetector

@pytest.fixture
def config():
    """Charge la configuration de test."""
    config_path = os.path.join(os.path.dirname(__file__), 'test_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def sample_data():
    """Génère des données de test."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')
    n_samples = len(dates)
    
    # Données de base
    data = pd.DataFrame({
        'open': np.random.normal(100, 1, n_samples),
        'high': np.random.normal(101, 1, n_samples),
        'low': np.random.normal(99, 1, n_samples),
        'close': np.random.normal(100, 1, n_samples),
        'volume': np.random.normal(1000, 100, n_samples)
    }, index=dates)
    
    # Calcul des returns
    data['returns'] = data['close'].pct_change()
    
    # Features techniques
    data['rsi'] = np.random.uniform(0, 100, n_samples)
    data['bb_width'] = np.random.normal(0, 1, n_samples)
    data['atr'] = np.random.normal(1, 0.1, n_samples)
    
    # Labels
    data['target'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    return data

def test_feature_monitor(config, sample_data):
    """Test du monitoring des features."""
    # Nettoyer les NaN dans les données de référence (par exemple, en remplaçant par la moyenne)
    reference_data = sample_data.iloc[:1000].fillna(sample_data.mean())
    monitor = FeatureMonitor(config)
    monitor.compute_reference_stats(reference_data)
    
    # Test de détection de drift
    test_data = sample_data.iloc[1000:1100]
    drift_scores = monitor.detect_drift(test_data)
    
    assert isinstance(drift_scores, dict)
    assert all(0 <= score <= 1 for score in drift_scores.values())
    
    # Test des alertes
    alerts = monitor.check_alerts(drift_scores)
    assert isinstance(alerts, list)
    
    # Test de la visualisation
    fig = monitor.plot_drift_analysis(test_data)
    assert fig is not None

def test_model_monitor(config, sample_data):
    """Test du monitoring des modèles."""
    monitor = ModelMonitor(config)
    X = sample_data[['rsi', 'bb_width', 'atr']].fillna(sample_data.mean())
    y_true = sample_data['target']
    y_pred = np.random.choice([0, 1], len(y_true))
    y_proba = np.random.uniform(0, 1, len(y_true))
    metrics = monitor.compute_metrics(y_true, y_pred, y_proba)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    # Exclure la matrice de confusion (qui est une liste) du calcul de dégradation
    metrics.pop('confusion_matrix', None)
    monitor.update_metrics(metrics, 'test_model')
    assert len(monitor.metrics_history['test_model']) == 1
    alerts = monitor.check_performance(metrics, 'test_model')
    
    # Test de la visualisation
    fig = monitor.plot_performance('test_model')
    assert fig is not None

def test_ml_model_monitoring(config, sample_data):
    """Test du monitoring intégré dans MLModel."""
    model = MLModel(config)
    X = sample_data[['rsi', 'bb_width', 'atr']].fillna(sample_data.mean())
    y = sample_data['target']
    
    # Split explicite train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model.feature_monitor.compute_reference_stats(X_train)
    # Adapter la méthode train pour accepter X_train, y_train
    metrics = model.train(X_train, y_train, X_val, y_val)
    assert isinstance(metrics, dict)
    assert model.model_monitor is not None
    assert len(model.model_monitor.metrics_history) > 0
    
    # Test de la prédiction avec monitoring
    y_pred, y_proba = model.predict(X, check_drift=True, check_performance=True)
    assert len(y_pred) == len(X)
    assert len(y_proba) == len(X)

def test_market_regime_monitoring(config, sample_data):
    """Test du monitoring intégré dans MarketRegimeDetector."""
    detector = MarketRegimeDetector(config)
    features = sample_data[['returns', 'rsi', 'bb_width', 'atr']].fillna(sample_data.mean())
    
    # Test de la détection avec monitoring
    regime_labels, metrics = detector.detect_regimes(features)
    assert len(regime_labels) == len(features)
    assert isinstance(metrics, dict)
    assert detector.model_monitor is not None
    assert len(detector.model_monitor.metrics_history) > 0
    
    # Test de la prédiction avec monitoring
    new_labels, new_metrics = detector.predict_regime(
        features,
        check_drift=True,
        check_performance=True
    )
    assert len(new_labels) == len(features)
    assert isinstance(new_metrics, dict)

def test_monitoring_persistence(config, sample_data, tmp_path):
    """Test de la persistance du monitoring."""
    feature_monitor = FeatureMonitor(config)
    model_monitor = ModelMonitor(config)
    
    # Nettoyer les NaN dans les données de test (par exemple, en remplaçant par la moyenne)
    sample_data_clean = sample_data.fillna(sample_data.mean())
    feature_monitor.compute_reference_stats(sample_data_clean)
    metrics = model_monitor.compute_metrics(
        sample_data_clean['target'],
        np.random.choice([0, 1], len(sample_data_clean)),
        np.random.uniform(0, 1, len(sample_data_clean))
    )
    model_monitor.update_metrics(metrics, 'test_model')
    
    # Sauvegarde
    feature_path = tmp_path / 'feature_monitor.joblib'
    model_path = tmp_path / 'model_monitor.joblib'
    
    feature_monitor.save_state(str(feature_path))
    model_monitor.save_state(str(model_path))
    
    # Chargement
    new_feature_monitor = FeatureMonitor(config)
    new_model_monitor = ModelMonitor(config)
    
    new_feature_monitor.load_state(str(feature_path))
    new_model_monitor.load_state(str(model_path))
    
    # Vérification
    assert new_feature_monitor.reference_stats is not None
    assert len(new_model_monitor.metrics_history) > 0

if __name__ == '__main__':
    pytest.main([__file__]) 