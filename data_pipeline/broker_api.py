"""
Module de connexion à MetaTrader 5.

Gère la connexion, l'authentification et la récupération des données
depuis la plateforme MetaTrader 5.
"""

import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import MetaTrader5 as mt5
from loguru import logger
import yaml

class MT5Connector:
    """Gestionnaire de connexion à MetaTrader 5."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le connecteur MT5.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.connected = False
        self._connect()
        
    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration depuis le fichier YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)['broker']
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la config: {e}")
            raise
            
    def _connect(self) -> bool:
        """
        Établit la connexion avec MetaTrader 5.
        
        Returns:
            bool: True si la connexion est réussie
        """
        if not mt5.initialize():
            logger.error(f"Échec de l'initialisation MT5: {mt5.last_error()}")
            return False
            
        # Authentification
        if not mt5.login(
            login=self.config['login'],
            password=self.config['password'],
            server=self.config['server']
        ):
            logger.error(f"Échec de l'authentification: {mt5.last_error()}")
            mt5.shutdown()
            return False
            
        self.connected = True
        logger.info("Connexion MT5 établie avec succès")
        return True
        
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        num_bars: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Récupère les données historiques OHLCV.
        
        Args:
            symbol: Symbole de l'instrument
            timeframe: Timeframe (M5, M15, H1, etc.)
            start_date: Date de début
            end_date: Date de fin
            num_bars: Nombre de barres à récupérer
            
        Returns:
            DataFrame avec les données OHLCV
        """
        if not self.connected:
            raise ConnectionError("Non connecté à MT5")
            
        # Conversion du timeframe
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'D1': mt5.TIMEFRAME_D1
        }
        
        if timeframe not in tf_map:
            raise ValueError(f"Timeframe non supporté: {timeframe}")
            
        # Préparation des paramètres
        rates = mt5.copy_rates_from_pos(
            symbol,
            tf_map[timeframe],
            0,
            num_bars if num_bars else 1000
        )
        
        if rates is None:
            logger.error(f"Erreur lors de la récupération des données: {mt5.last_error()}")
            return pd.DataFrame()
            
        # Conversion en DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Filtrage par date si nécessaire
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df
        
    def get_current_tick(self, symbol: str) -> Dict:
        """
        Récupère le dernier tick pour un symbole.
        
        Args:
            symbol: Symbole de l'instrument
            
        Returns:
            Dictionnaire avec les données du tick
        """
        if not self.connected:
            raise ConnectionError("Non connecté à MT5")
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Erreur lors de la récupération du tick: {mt5.last_error()}")
            return {}
            
        return {
            'time': pd.to_datetime(tick.time, unit='s'),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume
        }
        
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Récupère les informations sur un symbole.
        
        Args:
            symbol: Symbole de l'instrument
            
        Returns:
            Dictionnaire avec les informations du symbole
        """
        if not self.connected:
            raise ConnectionError("Non connecté à MT5")
            
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Erreur lors de la récupération des infos: {mt5.last_error()}")
            return {}
            
        return {
            'name': info.name,
            'bid': info.bid,
            'ask': info.ask,
            'point': info.point,
            'digits': info.digits,
            'spread': info.spread,
            'trade_contract_size': info.trade_contract_size,
            'volume_min': info.volume_min,
            'volume_max': info.volume_max,
            'volume_step': info.volume_step
        }
        
    def __del__(self):
        """Ferme la connexion MT5 à la destruction de l'objet."""
        if self.connected:
            mt5.shutdown()
            logger.info("Connexion MT5 fermée") 