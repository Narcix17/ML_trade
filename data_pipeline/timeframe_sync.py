"""
Module de synchronisation des données multi-timeframes.

Gère l'alignement et la synchronisation des données entre différents
timeframes pour assurer la cohérence des features et des signaux.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

class TimeframeSynchronizer:
    """Gestionnaire de synchronisation des timeframes."""
    
    def __init__(self, timeframes: List[str]):
        """
        Initialise le synchroniseur de timeframes.
        
        Args:
            timeframes: Liste des timeframes à synchroniser (ex: ['M5', 'M15', 'H1'])
        """
        self.timeframes = sorted(timeframes, key=self._get_timeframe_minutes)
        self.data: Dict[str, pd.DataFrame] = {}
        
    @staticmethod
    def _get_timeframe_minutes(tf: str) -> int:
        """Convertit un timeframe en minutes."""
        if tf.startswith('M'):
            return int(tf[1:])
        elif tf.startswith('H'):
            return int(tf[1:]) * 60
        elif tf == 'D1':
            return 24 * 60
        else:
            raise ValueError(f"Timeframe non supporté: {tf}")
            
    def add_dataframe(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Ajoute un DataFrame pour un timeframe spécifique.
        
        Args:
            timeframe: Timeframe du DataFrame
            df: DataFrame OHLCV
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe non supporté: {timeframe}")
            
        # Vérification des colonnes requises
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame doit contenir les colonnes: {required_cols}")
            
        # Copie et indexation
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("L'index doit être un DatetimeIndex")
            
        self.data[timeframe] = df
        logger.info(f"Données ajoutées pour le timeframe {timeframe}")
        
    def get_aligned_data(
        self,
        base_timeframe: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Retourne les données alignées sur le timeframe de base.
        
        Args:
            base_timeframe: Timeframe de référence pour l'alignement
            start_date: Date de début optionnelle
            end_date: Date de fin optionnelle
            
        Returns:
            Dictionnaire des DataFrames alignés par timeframe
        """
        if base_timeframe not in self.timeframes:
            raise ValueError(f"Timeframe de base non supporté: {base_timeframe}")
            
        if base_timeframe not in self.data:
            raise ValueError(f"Pas de données pour le timeframe de base: {base_timeframe}")
            
        # Récupération des données de base
        base_df = self.data[base_timeframe]
        if start_date:
            base_df = base_df[base_df.index >= start_date]
        if end_date:
            base_df = base_df[base_df.index <= end_date]
            
        aligned_data = {base_timeframe: base_df}
        
        # Alignement des autres timeframes
        for tf in self.timeframes:
            if tf == base_timeframe:
                continue
                
            if tf not in self.data:
                logger.warning(f"Pas de données pour le timeframe {tf}")
                continue
                
            # Alignement sur le timeframe de base
            aligned_df = self._align_timeframe(
                self.data[tf],
                base_df.index,
                tf,
                base_timeframe
            )
            aligned_data[tf] = aligned_df
            
        return aligned_data
        
    def _align_timeframe(
        self,
        df: pd.DataFrame,
        target_index: pd.DatetimeIndex,
        source_tf: str,
        target_tf: str
    ) -> pd.DataFrame:
        """
        Aligne les données d'un timeframe sur un autre.
        
        Args:
            df: DataFrame source
            target_index: Index temporel cible
            source_tf: Timeframe source
            target_tf: Timeframe cible
            
        Returns:
            DataFrame aligné
        """
        # Création d'un DataFrame vide avec l'index cible
        aligned_df = pd.DataFrame(index=target_index)
        
        # Pour chaque barre cible, on prend la dernière barre source disponible
        for col in df.columns:
            aligned_df[col] = df[col].reindex(target_index, method='ffill')
            
        # Gestion des valeurs manquantes au début
        aligned_df.fillna(method='bfill', inplace=True)
        
        return aligned_df
        
    def get_forward_returns(
        self,
        timeframe: str,
        periods: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Calcule les rendements forward pour différents horizons.
        
        Args:
            timeframe: Timeframe pour le calcul
            periods: Liste des périodes forward
            
        Returns:
            DataFrame avec les rendements forward
        """
        if timeframe not in self.data:
            raise ValueError(f"Pas de données pour le timeframe: {timeframe}")
            
        df = self.data[timeframe].copy()
        returns = pd.DataFrame(index=df.index)
        
        for period in periods:
            # Calcul du rendement forward
            forward_returns = df['close'].shift(-period) / df['close'] - 1
            returns[f'return_{period}'] = forward_returns
            
        return returns
        
    def get_volatility(
        self,
        timeframe: str,
        window: int = 20
    ) -> pd.Series:
        """
        Calcule la volatilité historique.
        
        Args:
            timeframe: Timeframe pour le calcul
            window: Fenêtre de calcul
            
        Returns:
            Série de volatilité
        """
        if timeframe not in self.data:
            raise ValueError(f"Pas de données pour le timeframe: {timeframe}")
            
        df = self.data[timeframe]
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        
        return volatility 