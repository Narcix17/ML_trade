"""
Script de test pour le module FeatureEngineer.
"""

import pandas as pd
import numpy as np
from loguru import logger
from features.feature_engineering import FeatureEngineer

# Configuration de test
test_config = {
    'features': {
        'technical': {
            'momentum': [
                {'name': 'rsi', 'period': 14, 'overbought': 70, 'oversold': 30},
                {'name': 'macd', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            ],
            'volatility': [
                {'name': 'bollinger', 'period': 20, 'std_dev': 2.0},
                {'name': 'atr', 'period': 14}
            ],
            'trend': [
                {'name': 'adx', 'period': 14, 'threshold': 25},
                {'name': 'vwap', 'period': '1d'}
            ]
        },
        'behavioral': {
            'patterns': [
                {'name': 'candlestick_patterns', 'patterns': ['engulfing', 'hammer', 'doji']}
            ],
            'volatility': [
                {'name': 'volatility_bursts', 'threshold': 2.0, 'window': 20},
                {'name': 'range_breakout', 'period': 20, 'threshold': 1.5}
            ],
            'volume': [
                {'name': 'volume_profile', 'periods': [1, 5, 15, 60]},
                {'name': 'tick_imbalance', 'period': 1000}
            ]
        },
        'contextual': {
            'time': [
                {
                    'name': 'sessions',
                    'sessions': ['eu', 'us', 'asia']
                }
            ],
            'market': [
                {'name': 'daily_direction', 'timeframe': 'H1', 'threshold': 0.001},
                {'name': 'market_regime', 'features': ['volatility', 'trend', 'volume'], 'n_regimes': 5}
            ]
        },
        'trading': {
            'sessions': {
                'eu': {'start': '08:00', 'end': '16:00'},
                'us': {'start': '13:30', 'end': '20:00'},
                'asia': {'start': '00:00', 'end': '08:00'}
            }
        }
    }
}

def generate_test_data(n_samples=1000):
    """Génère des données OHLCV de test."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Prix de base
    base_price = 100
    prices = np.random.normal(0, 1, n_samples).cumsum() + base_price
    
    # Génération des données OHLCV
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, n_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_samples)),
        'close': prices * (1 + np.random.normal(0, 0.01, n_samples)),
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Ajustement pour que high soit le plus haut et low le plus bas
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df

def test_feature_engineering():
    """Teste le module FeatureEngineer."""
    try:
        # Génération des données de test
        logger.info("Génération des données de test...")
        df = generate_test_data()
        logger.info(f"Données générées: {len(df)} lignes")
        
        # Initialisation du FeatureEngineer
        logger.info("Initialisation du FeatureEngineer...")
        engineer = FeatureEngineer(test_config)
        
        # Génération des features
        logger.info("Génération des features...")
        features = engineer.generate_features(
            df,
            feature_groups=['technical', 'behavioral', 'contextual'],
            dropna=True,
            scale=True,
            scale_method='robust'
        )
        
        # Vérification des résultats
        logger.info(f"Features générées: {len(features.columns)} colonnes")
        logger.info("\nColonnes générées:")
        for col in features.columns:
            logger.info(f"- {col}")
            
        # Vérification des NaN
        nan_cols = features.columns[features.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Colonnes avec des NaN: {nan_cols}")
        else:
            logger.info("Aucune colonne ne contient de NaN")
            
        # Vérification des infinis
        inf_cols = features.columns[np.isinf(features.select_dtypes(include=np.number)).any()].tolist()
        if inf_cols:
            logger.warning(f"Colonnes avec des infinis: {inf_cols}")
        else:
            logger.info("Aucune colonne ne contient d'infini")
            
        # Statistiques descriptives
        logger.info("\nStatistiques descriptives:")
        logger.info(features.describe())
        
        return features
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")
        raise

if __name__ == "__main__":
    # Configuration du logger
    logger.add("feature_engineering_test.log", rotation="1 day")
    
    # Exécution du test
    features = test_feature_engineering()
    
    # Sauvegarde des features pour inspection
    features.to_csv("test_features.csv")
    logger.info("Features sauvegardées dans test_features.csv")
    
    print("\nTest terminé avec succès.")
    # Call project overview if available
    try:
        from features.feature_engineering import project_overview
        project_overview()
    except ImportError:
        pass 