"""
Module de gestion du pipeline de données.

Ce module gère :
- L'ingestion des données depuis MT5
- La synchronisation multi-timeframes
- Le préprocessing des données
- Le stockage et la mise en cache
"""

from .broker_api import MT5Connector
from .data_processor import DataProcessor
from .timeframe_sync import TimeframeSynchronizer

__all__ = ['MT5Connector', 'DataProcessor', 'TimeframeSynchronizer'] 