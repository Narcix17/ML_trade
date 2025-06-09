"""
Module de feature engineering.

Gère la génération des features techniques, comportementales
et contextuelles pour les modèles de ML.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
import ta
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

class FeatureEngineer:
    """Gestionnaire de feature engineering."""
    
    def __init__(self, config):
        """
        Initialise l'ingénieur de features.
        Args:
            config: Configuration complète contenant les sections 'features' et 'trading'
        """
        self.features_config = config['features']
        self.trading_config = config['trading']
        self.feature_groups = {
            'technical': self._get_technical_features,
            'behavioral': self._get_behavioral_features,
            'contextual': self._get_contextual_features
        }
        self.scalers: Dict[str, Union[StandardScaler, RobustScaler]] = {}
        
    def _compute_base_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les métriques de base nécessaires pour les features.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec les métriques de base
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Volatilité
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volume metrics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        # Price metrics
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close']
        
        # Trend metrics
        df['trend'] = np.where(
            df['close'] > df['close'].rolling(window=20).mean(),
            1,
            -1
        )
        
        return df
        
    def generate_features(
        self,
        df: pd.DataFrame,
        feature_groups: Optional[List[str]] = None,
        dropna: bool = True,
        scale: bool = True,
        scale_method: str = 'robust'
    ) -> pd.DataFrame:
        """
        Génère l'ensemble des features demandées.
        
        Args:
            df: DataFrame OHLCV
            feature_groups: Liste des groupes de features à générer
            dropna: Supprimer les lignes avec NaN
            scale: Normaliser les features
            scale_method: Méthode de normalisation ('robust' ou 'standard')
            
        Returns:
            DataFrame avec les features générées
        """
        if feature_groups is None:
            feature_groups = list(self.feature_groups.keys())
            
        df = df.copy()
        
        # Calcul des métriques de base
        df = self._compute_base_metrics(df)
        
        # Génération des features par groupe
        for group in feature_groups:
            if group not in self.feature_groups:
                logger.warning(f"Groupe de features non supporté: {group}")
                continue
                
            df = self.feature_groups[group](df)
            
        # Gestion des NaN
        if dropna:
            initial_len = len(df)
            df = df.dropna()
            dropped = initial_len - len(df)
            if dropped > 0:
                logger.info(f"Suppression de {dropped} lignes avec NaN")
                
        # Normalisation
        if scale:
            df = self._scale_features(df, method=scale_method)
            
        return df
        
    def _scale_features(
        self,
        df: pd.DataFrame,
        method: str = 'robust'
    ) -> pd.DataFrame:
        """
        Normalise les features numériques.
        
        Args:
            df: DataFrame des features
            method: Méthode de normalisation
            
        Returns:
            DataFrame avec les features normalisées
        """
        # Colonnes à ne pas normaliser
        exclude_cols = [
            'hour', 'minute', 'day_of_week', 'month', 'quarter',
            'is_eu_session', 'is_us_session', 'is_session_overlap',
            'trend', 'market_regime'
        ]
        
        # Sélection des colonnes à normaliser
        scale_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not scale_cols:
            return df
            
        # Création du scaler si nécessaire
        if method not in self.scalers:
            self.scalers[method] = (
                RobustScaler() if method == 'robust' else StandardScaler()
            )
            
        # Normalisation
        df[scale_cols] = self.scalers[method].fit_transform(df[scale_cols])
        
        return df
        
    def _get_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les features techniques classiques.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec les features techniques
        """
        # RSI
        rsi_config = next(
            (f for f in self.features_config['technical']['momentum'] if f['name'] == 'rsi'),
            {'period': 14, 'overbought': 70, 'oversold': 30}
        )
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=rsi_config['period']).rsi()
        df['rsi'] = rsi
        df['rsi_ma'] = rsi.rolling(window=5).mean()
        df['rsi_signal'] = (rsi > rsi_config['overbought']).astype(int) - (rsi < rsi_config['oversold']).astype(int)
        
        # MACD
        macd_config = next(
            (f for f in self.features_config['technical']['momentum'] if f['name'] == 'macd'),
            {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        )
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=macd_config['slow_period'],
            window_fast=macd_config['fast_period'],
            window_sign=macd_config['signal_period']
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Bollinger Bands
        bb_config = next(
            (f for f in self.features_config['technical']['volatility'] if f['name'] == 'bollinger'),
            {'period': 20, 'std_dev': 2.0}
        )
        bollinger = ta.volatility.BollingerBands(
            close=df['close'],
            window=bb_config['period'],
            window_dev=bb_config['std_dev']
        )
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        atr_config = next(
            (f for f in self.features_config['technical']['volatility'] if f['name'] == 'atr'),
            {'period': 14}
        )
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=atr_config['period']
        ).average_true_range()
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']
        
        # ADX
        adx_config = next(
            (f for f in self.features_config['technical']['trend'] if f['name'] == 'adx'),
            {'period': 14, 'threshold': 25}
        )
        adx = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=adx_config['period']
        )
        df['adx'] = adx.adx()
        df['adx_signal'] = np.where(df['adx'] > adx_config['threshold'], 1, 0)
        
        # VWAP
        vwap_config = next(
            (f for f in self.features_config['technical']['trend'] if f['name'] == 'vwap'),
            {'period': '1d'}
        )
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=20
        ).volume_weighted_average_price()
        df['vwap_ratio'] = df['close'] / df['vwap']
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(
                close=df['close'],
                window=period
            ).roc()
            
        return df
        
    def _get_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les features comportementales basées sur les patterns de chandeliers.
        
        Args:
            df: DataFrame OHLCV
            
        Returns:
            DataFrame avec les features comportementales
        """
        # Engulfing Pattern
        df['engulfing'] = np.where(
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open']) &  # Current candle is bullish
            (df['close'] > df['open'].shift(1)) &  # Current close > previous open
            (df['open'] < df['close'].shift(1)),  # Current open < previous close
            1,  # Bullish engulfing
            np.where(
                (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
                (df['close'] < df['open']) &  # Current candle is bearish
                (df['close'] < df['open'].shift(1)) &  # Current close < previous open
                (df['open'] > df['close'].shift(1)),  # Current open > previous close
                -1,  # Bearish engulfing
                0  # No pattern
            )
        )
        
        # Doji Pattern
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        avg_shadow = (upper_shadow + lower_shadow) / 2
        
        df['doji'] = np.where(
            (body_size <= 0.1 * avg_shadow) &  # Small body
            (upper_shadow > 0) &  # Has upper shadow
            (lower_shadow > 0),  # Has lower shadow
            1,
            0
        )
        
        # Hammer Pattern
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        df['hammer'] = np.where(
            (lower_shadow > 2 * body_size) &  # Long lower shadow
            (upper_shadow < 0.1 * body_size) &  # Very small upper shadow
            (df['close'] > df['open']),  # Bullish candle
            1,
            np.where(
                (upper_shadow > 2 * body_size) &  # Long upper shadow
                (lower_shadow < 0.1 * body_size) &  # Very small lower shadow
                (df['close'] < df['open']),  # Bearish candle
                -1,
                0
            )
        )
        
        # Morning/Evening Star Pattern
        df['star'] = np.where(
            (df['close'].shift(2) < df['open'].shift(2)) &  # First candle is bearish
            (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * abs(df['close'].shift(2) - df['open'].shift(2))) &  # Second candle has small body
            (df['close'] > df['open']) &  # Third candle is bullish
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2),  # Third close > midpoint of first candle
            1,  # Morning star
            np.where(
                (df['close'].shift(2) > df['open'].shift(2)) &  # First candle is bullish
                (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * abs(df['close'].shift(2) - df['open'].shift(2))) &  # Second candle has small body
                (df['close'] < df['open']) &  # Third candle is bearish
                (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2),  # Third close < midpoint of first candle
                -1,  # Evening star
                0  # No pattern
            )
        )
        
        # Volume Profile
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_trend'] = np.where(
            df['volume'] > df['volume_ma'] * 1.5,
            1,  # High volume
            np.where(
                df['volume'] < df['volume_ma'] * 0.5,
                -1,  # Low volume
                0  # Normal volume
            )
        )
        
        return df
        
    def _get_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les features contextuelles (sessions, direction journalière, régime de marché).
        """
        # Sessions
        for session in self.trading_config['sessions']:
            session_config = self.trading_config['sessions'][session]
            start_hour = int(session_config['start'].split(':')[0])
            end_hour = int(session_config['end'].split(':')[0])
            df[f'in_{session}_session'] = ((df.index.hour >= start_hour) & (df.index.hour < end_hour)).astype(int)
        
        # Direction journalière
        daily_config = next(
            (f for f in self.features_config['contextual']['market'] if f['name'] == 'daily_direction'),
            {'timeframe': 'H1', 'threshold': 0.001}
        )
        # Calculate daily returns with explicit handling of NaN values
        daily_prices = df['close'].resample('D').last()
        daily_returns = daily_prices.pct_change(fill_method=None)
        # Forward fill any NaN values that aren't at the start
        daily_returns = daily_returns.fillna(method='ffill')
        df['daily_direction'] = daily_returns.reindex(df.index, method='ffill')
        df['daily_direction_signal'] = np.where(
            abs(df['daily_direction']) > daily_config['threshold'],
            np.sign(df['daily_direction']),
            0
        )
        
        # Régime de marché
        regime_config = next(
            (f for f in self.features_config['contextual']['market'] if f['name'] == 'market_regime'),
            {'features': ['volatility', 'trend', 'volume'], 'n_regimes': 5}
        )
        
        # Calcul des composantes du régime
        regime_features = []
        if 'volatility' in regime_config['features']:
            regime_features.append(df['volatility'])
        if 'trend' in regime_config['features']:
            regime_features.append(df['trend'])
        if 'volume' in regime_config['features']:
            regime_features.append(df['volume_ratio'])
            
        if regime_features:
            # Combinaison des features
            regime_score = pd.concat(regime_features, axis=1).mean(axis=1)
            # Classification en régimes (numérique au lieu de catégoriel)
            df['market_regime'] = pd.qcut(
                regime_score,
                q=regime_config['n_regimes'],
                labels=False
            )
            
        # Saisonnalité
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Tendance globale
        for window in [20, 50, 100]:
            df[f'trend_{window}'] = np.where(
                df['close'] > df['close'].rolling(window=window).mean(),
                1,
                -1
            )
            
        return df
        
    def get_feature_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: str = 'mutual_info'
    ) -> pd.Series:
        """
        Calcule l'importance des features.
        
        Args:
            features: DataFrame des features
            target: Série cible
            method: Méthode de calcul ('mutual_info', 'correlation')
            
        Returns:
            Série avec l'importance des features
        """
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            importance = mutual_info_regression(features, target)
        elif method == 'correlation':
            importance = features.corrwith(target).abs()
        else:
            raise ValueError(f"Méthode non supportée: {method}")
            
        return pd.Series(importance, index=features.columns).sort_values(ascending=False)
        
    def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        threshold: float = 0.01,
        method: str = 'mutual_info'
    ) -> List[str]:
        """
        Sélectionne les features les plus importantes.
        
        Args:
            features: DataFrame des features
            target: Série cible
            threshold: Seuil d'importance
            method: Méthode de sélection
            
        Returns:
            Liste des features sélectionnées
        """
        importance = self.get_feature_importance(features, target, method)
        return importance[importance > threshold].index.tolist()
        
    def save_scalers(self, path: str) -> None:
        """
        Sauvegarde les scalers.
        
        Args:
            path: Chemin de sauvegarde
        """
        import joblib
        joblib.dump(self.scalers, path)
        logger.info(f"Scalers sauvegardés: {path}")
        
    def load_scalers(self, path: str) -> None:
        """
        Charge les scalers.
        
        Args:
            path: Chemin des scalers
        """
        import joblib
        try:
            self.scalers = joblib.load(path)
            logger.info(f"Scalers chargés: {path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des scalers: {e}")
            raise 