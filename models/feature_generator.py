"""
Module de génération des features techniques.

Gère la génération des indicateurs techniques et leur monitoring.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from datetime import datetime

# Constantes pour les noms de colonnes
PRICE_COLS = ['open', 'high', 'low', 'close', 'volume']
TECHNICAL_COLS = {
    'momentum': ['rsi', 'roc', 'macd', 'macd_signal', 'macd_hist'],
    'volatility': ['bb_width', 'atr', 'natr'],
    'volume': ['obv', 'cmf', 'mfi', 'tick_imbalance'],
    'trend': ['adx', 'dmi_plus', 'dmi_minus']
}

class FeatureGenerator:
    """Générateur de features techniques."""
    
    def __init__(self, config: dict):
        """
        Initialise le générateur de features.
        
        Args:
            config: Configuration des features
        """
        self.config = config.get('features', {})
        self.scalers = {}  # Dictionnaire des scalers par méthode
        self.last_df = None  # Dernier DataFrame traité
        self.feature_groups = {}  # Suivi des features par groupe
        
    def _get_technical_config(self, indicator_name: str, group: str) -> dict:
        """
        Récupère la configuration d'un indicateur technique avec des valeurs par défaut.
        
        Args:
            indicator_name: Nom de l'indicateur
            group: Groupe de l'indicateur (momentum, volatility, etc.)
            
        Returns:
            Configuration de l'indicateur
        """
        default_configs = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'bb_width': {'period': 20, 'std': 2},
            'atr': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'adx': {'period': 14},
            'tick_imbalance': {'period': 1000}
        }
        
        return next(
            (f for f in self.config.get(group, {}).get('indicators', []) 
             if f.get('name') == indicator_name),
            default_configs.get(indicator_name, {})
        )
        
    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de momentum."""
        initial_cols = set(df.columns)
        
        # RSI
        rsi_config = self._get_technical_config('rsi', 'momentum')
        try:
            df['rsi'] = ta.momentum.RSIIndicator(
                close=df['close'],
                window=rsi_config['period']
            ).rsi()
        except Exception as e:
            logger.warning(f"RSI non calculé: {e}")
            df['rsi'] = np.nan
            
        # MACD
        macd_config = self._get_technical_config('macd', 'momentum')
        try:
            macd = ta.trend.MACD(
                close=df['close'],
                window_slow=macd_config['slow'],
                window_fast=macd_config['fast'],
                window_sign=macd_config['signal']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
        except Exception as e:
            logger.warning(f"MACD non calculé: {e}")
            df[['macd', 'macd_signal', 'macd_hist']] = np.nan
            
        # ROC (Rate of Change)
        try:
            df['roc'] = ta.momentum.ROCIndicator(
                close=df['close'],
                window=14
            ).roc()
        except Exception as e:
            logger.warning(f"ROC non calculé: {e}")
            df['roc'] = np.nan
            
        # Log des nouvelles features
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe momentum: {list(new_cols)}")
        self.feature_groups['momentum'] = list(new_cols)
        
        return df
        
    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de volatilité."""
        initial_cols = set(df.columns)
        
        # Bollinger Bands
        bb_config = self._get_technical_config('bb_width', 'volatility')
        try:
            bb = ta.volatility.BollingerBands(
                close=df['close'],
                window=bb_config['period'],
                window_dev=bb_config['std']
            )
            df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        except Exception as e:
            logger.warning(f"Bollinger Bands non calculées: {e}")
            df['bb_width'] = np.nan
            
        # ATR
        atr_config = self._get_technical_config('atr', 'volatility')
        try:
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=atr_config['period']
            ).average_true_range()
            df['natr'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=atr_config['period']
            ).average_true_range()
        except Exception as e:
            logger.warning(f"ATR non calculé: {e}")
            df[['atr', 'natr']] = np.nan
            
        # Log des nouvelles features
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe volatility: {list(new_cols)}")
        self.feature_groups['volatility'] = list(new_cols)
        
        return df
        
    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de volume."""
        initial_cols = set(df.columns)
        
        # OBV (On Balance Volume)
        try:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()
        except Exception as e:
            logger.warning(f"OBV non calculé: {e}")
            df['obv'] = np.nan
            
        # CMF (Chaikin Money Flow)
        try:
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=20
            ).chaikin_money_flow()
        except Exception as e:
            logger.warning(f"CMF non calculé: {e}")
            df['cmf'] = np.nan
            
        # MFI (Money Flow Index)
        try:
            df['mfi'] = ta.volume.MFIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=14
            ).money_flow_index()
        except Exception as e:
            logger.warning(f"MFI non calculé: {e}")
            df['mfi'] = np.nan
            
        # Tick Imbalance
        tick_config = self._get_technical_config('tick_imbalance', 'volume')
        try:
            df['tick_imbalance'] = np.where(
                df['close'] > df['open'],
                df['volume'],
                -df['volume']
            ).rolling(window=tick_config['period']).sum()
        except Exception as e:
            logger.warning(f"Tick Imbalance non calculé: {e}")
            df['tick_imbalance'] = np.nan
            
        # Log des nouvelles features
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe volume: {list(new_cols)}")
        self.feature_groups['volume'] = list(new_cols)
        
        return df
        
    def _compute_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features de tendance."""
        initial_cols = set(df.columns)
        
        # ADX (Average Directional Index)
        adx_config = self._get_technical_config('adx', 'trend')
        try:
            adx = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=adx_config['period']
            )
            df['adx'] = adx.adx()
            df['dmi_plus'] = adx.adx_pos()
            df['dmi_minus'] = adx.adx_neg()
        except Exception as e:
            logger.warning(f"ADX non calculé: {e}")
            df[['adx', 'dmi_plus', 'dmi_minus']] = np.nan
            
        # Log des nouvelles features
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe trend: {list(new_cols)}")
        self.feature_groups['trend'] = list(new_cols)
        
        return df
        
    def _get_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features contextuelles (à implémenter).
        
        Args:
            df: DataFrame des données
            
        Returns:
            DataFrame avec les features contextuelles
        """
        logger.info("Features contextuelles non encore implémentées")
        return df
        
    def _get_scalable_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Retourne la liste des colonnes à scaler.
        
        Args:
            df: DataFrame des données
            
        Returns:
            Liste des colonnes à scaler
        """
        return [col for col in df.columns if col not in PRICE_COLS]
        
    def fit_scaler(self, df: pd.DataFrame, method: str = 'robust') -> None:
        """
        Fit le scaler sur les données.
        
        Args:
            df: DataFrame des données
            method: Méthode de scaling ('robust' ou 'standard')
        """
        cols = self._get_scalable_columns(df)
        scaler = RobustScaler() if method == 'robust' else StandardScaler()
        self.scalers[method] = scaler.fit(df[cols])
        logger.info(f"Scaler {method} fitté sur {len(cols)} colonnes")
        
    def transform(self, df: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        """
        Transforme les données avec le scaler.
        
        Args:
            df: DataFrame des données
            method: Méthode de scaling ('robust' ou 'standard')
            
        Returns:
            DataFrame transformé
        """
        if method not in self.scalers:
            raise ValueError(f"Scaler {method} non fitté")
            
        cols = self._get_scalable_columns(df)
        df[cols] = self.scalers[method].transform(df[cols])
        return df
        
    def _compute_entropy_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute une feature d'entropie sur les returns en utilisant la config (ou des valeurs par défaut)."""
        import scipy.stats
        initial_cols = set(df.columns)
        # Récupérer la config (ou utiliser des valeurs par défaut)
        entropy_config = self.config.get('advanced_features', {}).get('entropy', {})
        bins = entropy_config.get('bins', 20)
        try:
            returns = df['close'].pct_change().dropna()
            hist, _ = np.histogram(returns, bins=bins, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist))
            df['entropy_returns'] = entropy
        except Exception as e:
            logger.warning(f"Entropie non calculée: {e}")
            df['entropy_returns'] = np.nan
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe advanced: {list(new_cols)}")
        self.feature_groups.setdefault('advanced', []).extend(list(new_cols))
        return df

    def _compute_regime_switching(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute un indicateur de changement de régime (rolling volatility) en utilisant la config (ou une valeur par défaut)."""
        initial_cols = set(df.columns)
        regime_config = self.config.get('advanced_features', {}).get('regime_switching', {})
        window = regime_config.get('window', 50)
        try:
            df['rolling_volatility'] = df['close'].pct_change().rolling(window=window).std()
        except Exception as e:
            logger.warning(f"Volatilité rolling non calculée: {e}")
            df['rolling_volatility'] = np.nan
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe advanced: {list(new_cols)}")
        self.feature_groups.setdefault('advanced', []).extend(list(new_cols))
        return df

    def _compute_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute le z-score sur certains indicateurs en utilisant la config (ou des valeurs par défaut)."""
        initial_cols = set(df.columns)
        zscore_config = self.config.get('advanced_features', {}).get('zscore', {})
        cols_to_zscore = zscore_config.get('columns', ['rsi', 'macd'])
        window = zscore_config.get('window', 100)
        for col in cols_to_zscore:
            if col in df.columns:
                try:
                    df[f'{col}_zscore'] = (df[col] - df[col].rolling(window=window).mean()) / df[col].rolling(window=window).std()
                except Exception as e:
                    logger.warning(f"Z-score {col} non calculé: {e}")
                    df[f'{col}_zscore'] = np.nan
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe advanced: {list(new_cols)}")
        self.feature_groups.setdefault('advanced', []).extend(list(new_cols))
        return df

    def _compute_rolling_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute la corrélation rolling entre returns et volume, RSI et ROC en utilisant la config (ou des valeurs par défaut)."""
        initial_cols = set(df.columns)
        corr_config = self.config.get('advanced_features', {}).get('rolling_correlation', {})
        window = corr_config.get('window', 50)
        try:
            df['returns'] = df['close'].pct_change()
            if 'rsi' in df.columns and 'roc' in df.columns:
                df['corr_rsi_roc'] = df['rsi'].rolling(window=window).corr(df['roc'])
            df['corr_returns_volume'] = df['returns'].rolling(window=window).corr(df['volume'])
        except Exception as e:
            logger.warning(f"Corrélation rolling non calculée: {e}")
            df['corr_rsi_roc'] = np.nan
            df['corr_returns_volume'] = np.nan
        new_cols = set(df.columns) - initial_cols
        logger.info(f"Groupe advanced: {list(new_cols)}")
        self.feature_groups.setdefault('advanced', []).extend(list(new_cols))
        return df

    def _get_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical features using ta library (RSI, MACD, Bollinger, ATR, ADX, VWAP, Momentum, ROC).
        """
        # RSI
        rsi_period = self.config.get('rsi_period', 14)
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=rsi_period).rsi()
        df['rsi_ma'] = df['rsi'].rolling(window=5).mean()
        df['rsi_signal'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
        # MACD
        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        # Bollinger Bands
        boll = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = boll.bollinger_hband()
        df['bb_middle'] = boll.bollinger_mavg()
        df['bb_lower'] = boll.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        df['atr_ma'] = df['atr'].rolling(window=20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']
        # ADX
        adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_signal'] = np.where(df['adx'] > 25, 1, 0)
        # VWAP
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20).volume_weighted_average_price()
        df['vwap_ratio'] = df['close'] / df['vwap']
        # Momentum & ROC
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(close=df['close'], window=period).roc()
        return df

    def _get_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate behavioral features using custom logic for candlestick patterns and volume profile.
        """
        # Engulfing Pattern
        df['engulfing'] = np.where(
            (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)),
            1,
            np.where(
                (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)),
                -1,
                0
            )
        )
        # Doji Pattern
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        avg_shadow = (upper_shadow + lower_shadow) / 2
        df['doji'] = np.where((body_size <= 0.1 * avg_shadow) & (upper_shadow > 0) & (lower_shadow > 0), 1, 0)
        # Hammer Pattern
        df['hammer'] = np.where((lower_shadow > 2 * body_size) & (upper_shadow < 0.1 * body_size) & (df['close'] > df['open']), 1,
            np.where((upper_shadow > 2 * body_size) & (lower_shadow < 0.1 * body_size) & (df['close'] < df['open']), -1, 0))
        # Morning/Evening Star
        df['star'] = np.where(
            (df['close'].shift(2) < df['open'].shift(2)) & (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * abs(df['close'].shift(2) - df['open'].shift(2))) & (df['close'] > df['open']) & (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2),
            1,
            np.where(
                (df['close'].shift(2) > df['open'].shift(2)) & (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.1 * abs(df['close'].shift(2) - df['open'].shift(2))) & (df['close'] < df['open']) & (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2),
                -1,
                0
            )
        )
        # Volume Profile
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_trend'] = np.where(df['volume'] > df['volume_ma'] * 1.5, 1, np.where(df['volume'] < df['volume_ma'] * 0.5, -1, 0))
        return df

    def generate_features(
        self,
        df: pd.DataFrame,
        scale: bool = True,
        scale_method: str = 'robust'
    ) -> pd.DataFrame:
        """
        Génère toutes les features techniques.
        
        Args:
            df: DataFrame des données OHLCV
            scale: Appliquer le scaling
            scale_method: Méthode de scaling
            
        Returns:
            DataFrame avec les features
        """
        # Vérification des colonnes requises
        missing_cols = [col for col in PRICE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")
            
        # Copie du DataFrame
        df = df.copy()
        
        # Génération des features par groupe
        df = self._compute_momentum_features(df)
        df = self._compute_volatility_features(df)
        df = self._compute_volume_features(df)
        df = self._compute_trend_features(df)
        df = self._get_contextual_features(df)
        # --- Features avancées ---
        df = self._compute_entropy_feature(df)
        df = self._compute_regime_switching(df)
        df = self._compute_zscore_features(df)
        df = self._compute_rolling_correlation(df)
        df = self._get_technical_features(df)
        df = self._get_behavioral_features(df)
        
        # Scaling si demandé
        if scale:
            if scale_method not in self.scalers:
                self.fit_scaler(df, scale_method)
            df = self.transform(df, scale_method)
            
        # Sauvegarde du dernier DataFrame
        self.last_df = df
        
        return df
        
    def get_feature_names(self) -> List[str]:
        """
        Retourne la liste des features générées.
        
        Returns:
            Liste des features
        """
        if self.last_df is None:
            raise ValueError("Aucune feature n'a été générée")
        return [col for col in self.last_df.columns if col not in PRICE_COLS]
        
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Retourne les features par groupe.
        
        Returns:
            Dictionnaire des features par groupe
        """
        return self.feature_groups
