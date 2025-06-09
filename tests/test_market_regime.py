import pytest
import numpy as np
import pandas as pd
from models.market_regime import MarketRegimeDetector

@pytest.fixture
def minimal_config():
    return {
        'clustering': {
            'regime': {'n_regimes': 2},
            'random_state': 42
        },
        'monitoring': {
            'features': {
                'drift_thresholds': { 'default': 0.5, 'mahalanobis': 1.0, 'correlation': 1.0, 'mean_shift': 0.3, 'std_shift': 0.3, 'missing_rate': 0.1 },
                'check_interval': 3600,
                'window_size': 1000,
                'alert_threshold': 0.7
            },
            'performance': {
                'metrics': {
                    'silhouette_score': {'threshold': 0.0, 'degradation_threshold': 1.0},
                    'calinski_harabasz_score': {'threshold': 0.0, 'degradation_threshold': 1.0}
                },
                'window_size': 100,
                'update_frequency': 10
            }
        }
    }

def test_detect_regimes(minimal_config):
    # Generate a small random DataFrame
    np.random.seed(42)
    features = pd.DataFrame({
        'volatility': np.random.rand(20),
        'trend': np.random.rand(20),
        'volume': np.random.rand(20)
    })
    detector = MarketRegimeDetector(minimal_config)
    labels, metrics = detector.detect_regimes(features, n_regimes=2)
    assert isinstance(labels, np.ndarray)
    assert labels.shape[0] == features.shape[0]
    assert isinstance(metrics, dict)
    assert 'silhouette_score' in metrics
    assert 'calinski_harabasz_score' in metrics 