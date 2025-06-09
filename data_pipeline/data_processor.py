"""
Module de prétraitement des données.

Gère le nettoyage, la normalisation et la préparation des données
pour l'ingestion dans les modèles de ML.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from monitoring import FeatureMonitor
from models.ml_model import MLModel
from models.market_regime import MarketRegimeDetector

class DataProcessor:
    """Gestionnaire de prétraitement des données."""
    
    def __init__(self, config: dict):
        """
        Initialise le processeur de données.
        
        Args:
            config: Configuration du prétraitement
        """
        self.config = config
        self.scalers: Dict[str, Union[StandardScaler, RobustScaler]] = {}
        self.feature_monitor = FeatureMonitor(config)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données OHLCV.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame nettoyé
        """
        df = df.copy()
        
        # Vérification des colonnes requises
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame doit contenir les colonnes: {required_cols}")
            
        # Suppression des doublons
        df = df[~df.index.duplicated(keep='first')]
        
        # Tri par index
        df.sort_index(inplace=True)
        
        # Détection et traitement des gaps
        df = self._handle_gaps(df)
        
        # Détection et traitement des outliers
        df = self._handle_outliers(df)
        
        # Vérification de la cohérence OHLC
        df = self._validate_ohlc(df)
        
        return df
        
    def _handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte et traite les gaps dans les données.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec les gaps traités
        """
        # Calcul des intervalles attendus
        time_diff = df.index.to_series().diff()
        median_diff = time_diff.median()
        
        # Identification des gaps significatifs
        gaps = time_diff[time_diff > median_diff * 1.5]
        
        if not gaps.empty:
            logger.warning(f"Détection de {len(gaps)} gaps dans les données")
            
            # Interpolation des gaps
            for idx in gaps.index:
                prev_idx = df.index[df.index.get_loc(idx) - 1]
                df.loc[idx:prev_idx, :] = df.loc[idx:prev_idx, :].interpolate(method='time')
                
        return df
        
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte et traite les outliers dans les données.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec les outliers traités
        """
        for col in ['open', 'high', 'low', 'close']:
            # Calcul des bornes avec méthode IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identification des outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if not outliers.empty:
                logger.warning(f"Détection de {len(outliers)} outliers dans {col}")
                
                # Remplacement par les bornes
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                
        return df
        
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie et corrige la cohérence des données OHLC.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec OHLC cohérent
        """
        # Correction high/low
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Vérification des volumes négatifs
        df.loc[df['volume'] < 0, 'volume'] = 0
        
        return df
        
    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        method: str = 'zscore',
        window: Optional[int] = None,
        check_drift: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalise les features selon la méthode spécifiée.
        
        Args:
            df: DataFrame avec les features
            feature_cols: Liste des colonnes à normaliser
            method: Méthode de normalisation ('zscore', 'minmax', 'robust')
            window: Fenêtre de normalisation (None pour global)
            check_drift: Vérifier la dérive des features
            
        Returns:
            Tuple (DataFrame normalisé, paramètres de normalisation)
        """
        df = df.copy()
        scaler_params = {}
        
        # Vérification de la dérive des features
        if check_drift:
            drift_scores = self.feature_monitor.detect_drift(df[feature_cols])
            alerts = self.feature_monitor.check_alerts(drift_scores)
            
            if alerts:
                logger.warning(f"Dérive détectée dans les features: {alerts}")
                # Log des alertes dans MLflow
                self.feature_monitor.log_to_mlflow(drift_scores, alerts)
                
        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Colonne {col} non trouvée, ignorée")
                continue
                
            if method == 'zscore':
                if window:
                    # Normalisation glissante
                    mean = df[col].rolling(window=window).mean()
                    std = df[col].rolling(window=window).std()
                    df[f'{col}_norm'] = (df[col] - mean) / std
                    scaler_params[col] = {'method': 'zscore', 'window': window}
                else:
                    # Normalisation globale
                    mean = df[col].mean()
                    std = df[col].std()
                    df[f'{col}_norm'] = (df[col] - mean) / std
                    scaler_params[col] = {'method': 'zscore', 'mean': mean, 'std': std}
                    
            elif method == 'minmax':
                if window:
                    # Normalisation glissante
                    min_val = df[col].rolling(window=window).min()
                    max_val = df[col].rolling(window=window).max()
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                    scaler_params[col] = {'method': 'minmax', 'window': window}
                else:
                    # Normalisation globale
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                    scaler_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                    
            elif method == 'robust':
                if window:
                    # Normalisation glissante avec quantiles
                    q1 = df[col].rolling(window=window).quantile(0.25)
                    q3 = df[col].rolling(window=window).quantile(0.75)
                    iqr = q3 - q1
                    df[f'{col}_norm'] = (df[col] - q1) / iqr
                    scaler_params[col] = {'method': 'robust', 'window': window}
                else:
                    # Normalisation globale avec RobustScaler
                    if col not in self.scalers:
                        self.scalers[col] = RobustScaler()
                    df[f'{col}_norm'] = self.scalers[col].fit_transform(df[[col]])
                    scaler_params[col] = {'method': 'robust', 'scaler': self.scalers[col]}
                    
        # Calcul des statistiques de référence après normalisation
        if check_drift:
            self.feature_monitor.compute_reference_stats(
                df[[f'{col}_norm' for col in feature_cols if f'{col}_norm' in df.columns]]
            )
            
        return df, scaler_params
        
    def inverse_normalize(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Inverse la normalisation des features.
        
        Args:
            df: DataFrame avec les features normalisées
            feature_cols: Liste des colonnes à dénormaliser
            
        Returns:
            DataFrame avec les features dénormalisées
        """
        df = df.copy()
        
        for col in feature_cols:
            norm_col = f'{col}_norm'
            if norm_col not in df.columns:
                logger.warning(f"Colonne normalisée {norm_col} non trouvée, ignorée")
                continue
                
            if col not in self.scalers:
                logger.warning(f"Paramètres de normalisation pour {col} non trouvés")
                continue
                
            params = self.scalers[col]
            method = params['method']
            
            if method == 'zscore':
                if 'window' in params:
                    # Dénormalisation glissante
                    mean = df[col].rolling(window=params['window']).mean()
                    std = df[col].rolling(window=params['window']).std()
                    df[col] = df[norm_col] * std + mean
                else:
                    # Dénormalisation globale
                    df[col] = df[norm_col] * params['std'] + params['mean']
                    
            elif method == 'minmax':
                if 'window' in params:
                    # Dénormalisation glissante
                    min_val = df[col].rolling(window=params['window']).min()
                    max_val = df[col].rolling(window=params['window']).max()
                    df[col] = df[norm_col] * (max_val - min_val) + min_val
                else:
                    # Dénormalisation globale
                    df[col] = df[norm_col] * (params['max'] - params['min']) + params['min']
                    
            elif method == 'robust':
                if 'window' in params:
                    # Dénormalisation glissante robuste
                    median = df[col].rolling(window=params['window']).median()
                    iqr = df[col].rolling(window=params['window']).apply(
                        lambda x: stats.iqr(x) if len(x) > 1 else 1
                    )
                    df[col] = df[norm_col] * iqr + median
                else:
                    # Dénormalisation globale robuste
                    df[col] = df[norm_col] * params['iqr'] + params['median']
                    
        return df
        
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features techniques basiques.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec les features techniques
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moyennes mobiles
        for window in [5, 10, 20, 50, 100]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
        # Volatilité
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Range
        df['range'] = df['high'] - df['low']
        df['range_ma'] = df['range'].rolling(window=20).mean()
        df['range_ratio'] = df['range'] / df['range_ma']
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df 

    def save_state(self, path: str) -> None:
        """
        Sauvegarde l'état du processeur.
        
        Args:
            path: Chemin de sauvegarde
        """
        import joblib
        
        state = {
            'scalers': self.scalers,
            'feature_monitor': self.feature_monitor
        }
        
        joblib.dump(state, path)
        logger.info(f"État du processeur sauvegardé: {path}")
        
    def load_state(self, path: str) -> None:
        """
        Charge l'état du processeur.
        
        Args:
            path: Chemin de chargement
        """
        import joblib
        
        try:
            state = joblib.load(path)
            self.scalers = state['scalers']
            self.feature_monitor = state['feature_monitor']
            logger.info(f"État du processeur chargé: {path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {e}")
            raise 