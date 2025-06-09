"""
Trading Servo - Bridge between ML predictions and order execution.

This module integrates:
1. ML model predictions
2. Market regime detection
3. Risk management
4. Order execution
5. Position management
6. Performance monitoring
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
import time
import threading
from queue import Queue
import joblib
import yaml
import os
from dataclasses import dataclass
from enum import Enum
import json
import glob

from execution.order_manager import OrderManager, OrderRequest, OrderResult
from models.ml_model import MLModel
from models.market_regime import MarketRegimeDetector
from features.feature_engineering import FeatureEngineer
from monitoring.feature_monitor import FeatureMonitor
from monitoring.model_monitor import ModelMonitor

class SignalType(Enum):
    """Types de signaux de trading."""
    NEUTRAL = 0
    BUY = 1
    SELL = 2

class RegimeType(Enum):
    """Types de régimes de marché."""
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    VOLATILE = 2
    RANGING = 3
    BREAKOUT = 4

@dataclass
class TradingSignal:
    """Signal de trading avec métadonnées."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float
    regime: int
    features: Dict[str, float]
    model_version: str
    metadata: Dict[str, Any]

@dataclass
class Position:
    """Position de trading."""
    symbol: str
    side: str  # 'long' or 'short'
    volume: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    regime: Optional[int] = None
    signal_confidence: Optional[float] = None

class TradingServo:
    """Servo de trading principal."""
    
    def __init__(self, config: dict, mode: str = "paper"):
        """
        Initialise le servo de trading.
        
        Args:
            config: Configuration dictionary
            mode: Mode de trading ("paper", "live", "backtest")
        """
        # Configuration
        self.config = config
        self.mode = mode
        
        # Initialisation des composants
        self.order_manager = OrderManager(config) if mode == "live" else None
        self.ml_model = None
        self.regime_detector = None
        self.feature_engineer = None
        self.feature_monitor = None
        self.model_monitor = None
        
        # État du système
        self.is_running = False
        self.positions: Dict[str, Position] = {}
        self.signal_history: List[TradingSignal] = []
        self.performance_metrics = {}
        
        # Queues et threads
        self.signal_queue = Queue()
        self.execution_queue = Queue()
        self.monitoring_queue = Queue()
        
        # Paramètres de trading
        self.trading_config = self.config['trading']
        self.execution_config = self.config['execution']
        self.risk_config = self.config['trading']['risk_management']
        
        # Chargement des modèles
        self._load_models()
        
        # Démarrage des threads
        self._start_threads()
        
        logger.info(f"Trading Servo initialisé en mode {mode}")
        
    def _load_models(self) -> None:
        """Charge les modèles entraînés."""
        try:
            # Récupération des paramètres de configuration
            model_type = self.config['ml']['model']['type']  # xgboost ou lightgbm
            symbol = self.config['broker']['symbols'][0]['name']  # EURUSD
            timeframe = self.config['broker']['timeframes'][0]['name']  # M5
            
            # Recherche du modèle le plus récent pour cette configuration
            model_files = self._find_latest_models(model_type, symbol, timeframe)
            
            if not model_files:
                logger.error(f"Aucun modèle trouvé pour {model_type}_{symbol}_{timeframe}")
                return
            
            # Chargement du modèle ML
            if 'ml_model' in model_files:
                self.ml_model = joblib.load(model_files['ml_model'])
                logger.info(f"Modèle ML chargé: {model_files['ml_model']}")
            else:
                logger.warning("Modèle ML non trouvé")
                
            # Chargement du détecteur de régimes
            if 'regime_detector' in model_files:
                self.regime_detector = joblib.load(model_files['regime_detector'])
                logger.info(f"Détecteur de régimes chargé: {model_files['regime_detector']}")
            else:
                logger.warning("Détecteur de régimes non trouvé")
                
            # Chargement du feature engineer
            if 'feature_engineer' in model_files:
                self.feature_engineer = joblib.load(model_files['feature_engineer'])
                logger.info(f"Feature Engineer chargé: {model_files['feature_engineer']}")
            else:
                logger.warning("Feature Engineer non trouvé")
                
            # Chargement du feature monitor
            if 'feature_monitor' in model_files:
                self.feature_monitor = joblib.load(model_files['feature_monitor'])
                logger.info(f"Feature Monitor chargé: {model_files['feature_monitor']}")
            else:
                logger.warning("Feature Monitor non trouvé")
                
            # Chargement des métadonnées
            if 'metadata' in model_files:
                with open(model_files['metadata'], 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Métadonnées chargées: {model_files['metadata']}")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {e}")
            
    def _find_latest_models(self, model_type: str, symbol: str, timeframe: str) -> Dict[str, str]:
        """
        Trouve les modèles les plus récents pour une configuration donnée.
        
        Args:
            model_type: Type de modèle (xgboost, lightgbm)
            symbol: Symbole de trading (EURUSD, etc.)
            timeframe: Timeframe (M5, H1, etc.)
            
        Returns:
            Dictionnaire avec les chemins des fichiers de modèles
        """
        models_dir = "models/saved"
        if not os.path.exists(models_dir):
            return {}
            
        # Pattern de recherche
        pattern = f"*_{model_type}_{symbol}_{timeframe}_*.joblib"
        metadata_pattern = f"metadata_{model_type}_{symbol}_{timeframe}_*.json"
        
        # Recherche des fichiers
        model_files = glob.glob(os.path.join(models_dir, pattern))
        metadata_files = glob.glob(os.path.join(models_dir, metadata_pattern))
        
        if not model_files:
            return {}
            
        # Tri par timestamp (le plus récent en premier)
        model_files.sort(reverse=True)
        metadata_files.sort(reverse=True)
        
        # Extraction du timestamp du fichier le plus récent
        latest_timestamp = None
        for file in model_files:
            # Extraction du timestamp du nom de fichier
            filename = os.path.basename(file)
            parts = filename.split('_')
            if len(parts) >= 5:
                timestamp = parts[-1].replace('.joblib', '')
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
        
        if latest_timestamp is None:
            return {}
            
        # Construction des chemins des modèles
        models = {}
        
        # ML Model
        ml_model_path = os.path.join(models_dir, f"ml_model_{model_type}_{symbol}_{timeframe}_{latest_timestamp}.joblib")
        if os.path.exists(ml_model_path):
            models['ml_model'] = ml_model_path
            
        # Régime Detector
        regime_path = os.path.join(models_dir, f"regime_detector_{symbol}_{timeframe}_{latest_timestamp}.joblib")
        if os.path.exists(regime_path):
            models['regime_detector'] = regime_path
            
        # Feature Engineer
        feature_engineer_path = os.path.join(models_dir, f"feature_engineer_{symbol}_{timeframe}_{latest_timestamp}.joblib")
        if os.path.exists(feature_engineer_path):
            models['feature_engineer'] = feature_engineer_path
            
        # Feature Monitor
        feature_monitor_path = os.path.join(models_dir, f"feature_monitor_{symbol}_{timeframe}_{latest_timestamp}.joblib")
        if os.path.exists(feature_monitor_path):
            models['feature_monitor'] = feature_monitor_path
            
        # Métadonnées
        metadata_path = os.path.join(models_dir, f"metadata_{model_type}_{symbol}_{timeframe}_{latest_timestamp}.json")
        if os.path.exists(metadata_path):
            models['metadata'] = metadata_path
            
        return models
        
    def _start_threads(self) -> None:
        """Démarre les threads de traitement."""
        # Thread de génération de signaux
        self.signal_thread = threading.Thread(
            target=self._signal_generator,
            daemon=True
        )
        self.signal_thread.start()
        
        # Thread d'exécution
        self.execution_thread = threading.Thread(
            target=self._execution_processor,
            daemon=True
        )
        self.execution_thread.start()
        
        # Thread de monitoring
        self.monitoring_thread = threading.Thread(
            target=self._performance_monitor,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Threads de trading démarrés")
        
    def start(self) -> None:
        """Démarre le servo de trading en boucle continue."""
        if self.is_running:
            logger.warning("Le servo est déjà en cours d'exécution")
            return
        self.is_running = True
        logger.info("Trading Servo démarré (boucle continue)")
        try:
            while self.is_running:
                self._process_market_data()
                time.sleep(self.config.get('servo', {}).get('interval', 60))  # Default: 60s
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
            self.stop()
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale du servo: {e}")
            self.stop()

    def stop(self) -> None:
        """Arrête le servo de trading."""
        self.is_running = False
        
        # Fermeture des positions ouvertes
        self._close_all_positions()
        
        # Arrêt des threads
        self.signal_queue.put(None)  # Signal d'arrêt
        self.execution_queue.put(None)
        self.monitoring_queue.put(None)
        
        # Arrêt du gestionnaire d'ordres
        if self.order_manager:
            self.order_manager.shutdown()
            
        logger.info("Trading Servo arrêté")
        
    def _process_market_data(self) -> None:
        """Traite les données de marché en temps réel."""
        try:
            # Récupération des données récentes depuis 2025/06/01
            symbol = self.config['broker']['symbols'][0]['name']
            timeframe = self.config['broker']['timeframes'][0]['name']
            
            # Fetch data from 2025/06/01 onwards (unseen data)
            data = self._fetch_recent_data(symbol, timeframe)
            
            if data is None or len(data) == 0:
                logger.warning("Aucune donnée récente disponible")
                return
                
            # Génération des features
            if self.feature_engineer:
                features = self.feature_engineer.generate_features(data)
                
                # Détection du régime
                if self.regime_detector:
                    regime_labels, _ = self.regime_detector.detect_regimes(features)
                    current_regime = regime_labels.iloc[-1]
                else:
                    current_regime = 0
                    
                # Prédiction ML
                if self.ml_model:
                    prediction, confidence = self.ml_model.predict(
                        features.iloc[-1:],
                        return_proba=True
                    )
                    
                    signal_type = SignalType(prediction[0])
                    signal_confidence = np.max(confidence[0])
                    
                    # Logs détaillés des prédictions
                    logger.info(f"Prédiction ML: {signal_type.name} (conf: {signal_confidence:.3f})")
                    logger.info(f"Probabilités: {confidence[0]}")
                    
                    # Création du signal
                    signal = TradingSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=signal_confidence,
                        regime=current_regime,
                        features=features.iloc[-1].to_dict(),
                        model_version="v1.0",
                        metadata={
                            'timeframe': timeframe,
                            'regime_confidence': 0.8,  # À calculer
                            'data_source': 'real_time_2025'
                        }
                    )
                    
                    # Ajout à la queue des signaux
                    self.signal_queue.put(signal)
                    logger.info(f"Signal ajouté à la queue: {signal_type.name}")
                    
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {e}")
            
    def _fetch_recent_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Récupère les données récentes depuis 2025/06/01.
        
        Args:
            symbol: Symbole de trading
            timeframe: Timeframe (M5, H1, etc.)
            
        Returns:
            DataFrame avec les données OHLCV
        """
        try:
            import MetaTrader5 as mt5
            
            # Initialisation de MT5
            if not mt5.initialize():
                logger.error("Échec de l'initialisation MT5")
                return None
                
            # Configuration du timeframe
            tf_map = {
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Date de début (2025/06/01)
            start_date = pd.Timestamp("2025-06-01")
            end_date = pd.Timestamp.now()
            
            # Récupération des données
            rates = mt5.copy_rates_range(
                symbol,
                mt5_timeframe,
                start_date,
                end_date
            )
            
            mt5.shutdown()
            
            if rates is None or len(rates) == 0:
                logger.warning(f"Aucune donnée trouvée pour {symbol} depuis {start_date}")
                return None
                
            # Conversion en DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Renommage des colonnes
            df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            logger.info(f"Données récupérées: {len(df)} bars pour {symbol} depuis {start_date}")
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            return None
        
    def _signal_generator(self) -> None:
        """Générateur de signaux de trading."""
        while self.is_running:
            try:
                signal = self.signal_queue.get(timeout=1)
                if signal is None:  # Signal d'arrêt
                    break
                    
                # Validation du signal
                if self._validate_signal(signal):
                    # Ajout à l'historique
                    self.signal_history.append(signal)
                    
                    # Envoi à la queue d'exécution
                    self.execution_queue.put(signal)
                    
                    logger.info(
                        f"Signal généré: {signal.symbol} "
                        f"{signal.signal_type.name} "
                        f"(conf: {signal.confidence:.3f}, "
                        f"régime: {signal.regime})"
                    )
                    
            except Exception as e:
                logger.error(f"Erreur dans le générateur de signaux: {e}")
                
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Valide un signal de trading."""
        # Vérification de la confiance minimale
        min_confidence = self.trading_config.get('min_confidence', 0.7)
        if signal.confidence < min_confidence:
            logger.debug(f"Signal rejeté - confiance insuffisante: {signal.confidence:.3f} < {min_confidence}")
            return False
            
        # Vérification des filtres de trading
        if not self._check_trading_filters(signal):
            logger.debug(f"Signal rejeté - filtres de trading non respectés")
            return False
            
        # Vérification des contraintes de risque
        if not self._check_risk_constraints(signal):
            logger.debug(f"Signal rejeté - contraintes de risque non respectées")
            return False
            
        logger.info(f"Signal validé: {signal.signal_type.name} (conf: {signal.confidence:.3f})")
        return True
        
    def _check_trading_filters(self, signal: TradingSignal) -> bool:
        """Vérifie les filtres de trading."""
        filters = self.trading_config.get('filters', {})
        
        # Filtre de spread
        max_spread = filters.get('max_spread', 0.0002)
        # TODO: Récupérer le spread actuel
        
        # Filtre de session
        if filters.get('session_filter', True):
            if not self._is_trading_session():
                return False
                
        # Filtre de volatilité
        min_volatility = filters.get('min_volatility', 0.0001)
        # TODO: Calculer la volatilité actuelle
        
        return True
        
    def _check_risk_constraints(self, signal: TradingSignal) -> bool:
        """Vérifie les contraintes de risque."""
        # Nombre maximum de positions
        max_positions = self.risk_config.get('max_positions', 3)
        if len(self.positions) >= max_positions:
            return False
            
        # Taille maximale des positions
        max_position_size = self.risk_config.get('position_size', 0.02)
        # TODO: Vérifier la taille de position proposée
        
        # Drawdown maximum
        max_drawdown = self.risk_config.get('max_drawdown', 0.05)
        # TODO: Calculer le drawdown actuel
        
        return True
        
    def _is_trading_session(self) -> bool:
        """Vérifie si on est dans une session de trading."""
        sessions = self.trading_config.get('sessions', {})
        current_time = datetime.now()
        
        # Vérification des sessions configurées
        for session_name, session_config in sessions.items():
            if session_name == 'overlap':
                continue  # Session de chevauchement
                
            start_time = datetime.strptime(session_config['start'], '%H:%M').time()
            end_time = datetime.strptime(session_config['end'], '%H:%M').time()
            
            if start_time <= current_time.time() <= end_time:
                return True
                
        return False
        
    def _execution_processor(self) -> None:
        """Processeur d'exécution des ordres."""
        while self.is_running:
            try:
                signal = self.execution_queue.get(timeout=1)
                if signal is None:  # Signal d'arrêt
                    break
                    
                # Exécution du signal
                self._execute_signal(signal)
                
            except Exception as e:
                logger.error(f"Erreur dans le processeur d'exécution: {e}")
                
    def _execute_signal(self, signal: TradingSignal) -> None:
        """Exécute un signal de trading."""
        try:
            # Calcul de la taille de position
            position_size = self._calculate_position_size(signal)
            
            # Calcul des stops
            stop_loss, take_profit = self._calculate_stops(signal)
            
            # Création de l'ordre
            if signal.signal_type == SignalType.BUY:
                order_request = OrderRequest(
                    symbol=signal.symbol,
                    order_type='market',
                    volume=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    comment=f"ML_Signal_Buy_{signal.confidence:.3f}"
                )
                
            elif signal.signal_type == SignalType.SELL:
                order_request = OrderRequest(
                    symbol=signal.symbol,
                    order_type='market',
                    volume=-position_size,  # Volume négatif pour la vente
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    comment=f"ML_Signal_Sell_{signal.confidence:.3f}"
                )
                
            else:  # NEUTRAL
                return
                
            # Exécution selon le mode
            if self.mode == "live" and self.order_manager:
                # Mode live - envoi réel
                result = self.order_manager.send_order(order_request)
                
                if result.status == 'filled':
                    # Création de la position
                    position = Position(
                        symbol=signal.symbol,
                        side='long' if signal.signal_type == SignalType.BUY else 'short',
                        volume=abs(position_size),
                        entry_price=result.fill_price,
                        entry_time=result.fill_time,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        regime=signal.regime,
                        signal_confidence=signal.confidence
                    )
                    
                    self.positions[signal.symbol] = position
                    
                    logger.info(
                        f"Position ouverte (LIVE): {signal.symbol} "
                        f"{position.side} {position.volume} "
                        f"@ {position.entry_price}"
                    )
                    
                else:
                    logger.error(f"Échec de l'ordre (LIVE): {result.error}")
                    
            else:
                # Mode paper/backtest - simulation
                simulated_price = 1.1000  # Prix simulé (à améliorer)
                simulated_time = datetime.now()
                
                # Création de la position simulée
                position = Position(
                    symbol=signal.symbol,
                    side='long' if signal.signal_type == SignalType.BUY else 'short',
                    volume=abs(position_size),
                    entry_price=simulated_price,
                    entry_time=simulated_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    regime=signal.regime,
                    signal_confidence=signal.confidence
                )
                
                self.positions[signal.symbol] = position
                
                logger.info(
                    f"Position simulée ({self.mode.upper()}): {signal.symbol} "
                    f"{position.side} {position.volume} "
                    f"@ {position.entry_price} "
                    f"(conf: {signal.confidence:.3f}, régime: {signal.regime})"
                )
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal: {e}")
            
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calcule la taille de position basée sur le risque."""
        # Récupération du capital
        account_info = self.order_manager.get_account_info()
        if not account_info:
            return 0.01  # Taille par défaut
            
        capital = account_info['equity']
        
        # Taille de position en % du capital
        position_pct = self.risk_config.get('position_size', 0.02)
        
        # Ajustement basé sur la confiance
        confidence_multiplier = signal.confidence
        
        # Ajustement basé sur le régime
        regime_multiplier = self._get_regime_multiplier(signal.regime)
        
        # Calcul final
        position_value = capital * position_pct * confidence_multiplier * regime_multiplier
        
        # Conversion en volume (à adapter selon l'instrument)
        symbol_config = next(
            (s for s in self.config['broker']['symbols'] if s['name'] == signal.symbol),
            None
        )
        
        if symbol_config:
            point_value = symbol_config.get('point', 0.0001)
            position_size = position_value / (point_value * 100000)  # Pour forex
        else:
            position_size = 0.01  # Taille par défaut
            
        # Limites
        min_lot = symbol_config.get('min_lot', 0.01) if symbol_config else 0.01
        max_lot = symbol_config.get('max_lot', 100.0) if symbol_config else 100.0
        
        return max(min_lot, min(max_lot, position_size))
        
    def _get_regime_multiplier(self, regime: int) -> float:
        """Retourne le multiplicateur de taille basé sur le régime."""
        # Multiplicateurs par régime (à ajuster selon l'analyse)
        regime_multipliers = {
            0: 0.8,   # Régime 0: Réduire la taille
            1: 0.5,   # Régime 1: Très conservateur
            2: 1.2,   # Régime 2: Légèrement plus agressif
            3: 1.0,   # Régime 3: Normal
            4: 1.5    # Régime 4: Plus agressif
        }
        
        return regime_multipliers.get(regime, 1.0)
        
    def _calculate_stops(self, signal: TradingSignal) -> Tuple[Optional[float], Optional[float]]:
        """Calcule les stops loss et take profit."""
        stops_config = self.trading_config.get('stops', {})
        
        # Récupération du prix actuel (à implémenter)
        current_price = 1.1000  # Placeholder
        
        # Calcul basé sur l'ATR ou un pourcentage fixe
        atr_multiplier = stops_config.get('stop_loss_atr', 2.0)
        tp_multiplier = stops_config.get('take_profit_atr', 3.0)
        
        # TODO: Calculer l'ATR réel
        atr = 0.001  # Placeholder
        
        if signal.signal_type == SignalType.BUY:
            stop_loss = current_price - (atr * atr_multiplier)
            take_profit = current_price + (atr * tp_multiplier)
        else:  # SELL
            stop_loss = current_price + (atr * atr_multiplier)
            take_profit = current_price - (atr * tp_multiplier)
            
        return stop_loss, take_profit
        
    def _performance_monitor(self) -> None:
        """Moniteur de performance."""
        while self.is_running:
            try:
                # Calcul des métriques de performance
                self._update_performance_metrics()
                
                # Vérification des alertes
                self._check_alerts()
                
                # Log des métriques
                self._log_performance()
                
                time.sleep(60)  # Mise à jour toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur dans le moniteur de performance: {e}")
                
    def _update_performance_metrics(self) -> None:
        """Met à jour les métriques de performance."""
        # Calcul du P&L
        total_pnl = 0.0
        for position in self.positions.values():
            # TODO: Calculer le P&L de chaque position
            pass
            
        # Calcul du drawdown
        # TODO: Calculer le drawdown
        
        # Métriques de trading
        self.performance_metrics = {
            'total_pnl': total_pnl,
            'open_positions': len(self.positions),
            'total_signals': len(self.signal_history),
            'win_rate': 0.0,  # À calculer
            'sharpe_ratio': 0.0,  # À calculer
            'max_drawdown': 0.0  # À calculer
        }
        
    def _check_alerts(self) -> None:
        """Vérifie les alertes de performance."""
        # Alertes de drawdown
        max_drawdown = self.risk_config.get('max_drawdown', 0.05)
        if self.performance_metrics.get('max_drawdown', 0) > max_drawdown:
            logger.warning(f"Drawdown maximum atteint: {self.performance_metrics['max_drawdown']:.2%}")
            
        # Alertes de perte quotidienne
        max_daily_loss = self.risk_config.get('max_daily_loss', 0.02)
        # TODO: Calculer la perte quotidienne
        
    def _log_performance(self) -> None:
        """Log les métriques de performance."""
        logger.info(
            f"Performance: P&L={self.performance_metrics.get('total_pnl', 0):.2f}, "
            f"Positions={self.performance_metrics.get('open_positions', 0)}, "
            f"Signals={self.performance_metrics.get('total_signals', 0)}"
        )
        
    def _close_all_positions(self) -> None:
        """Ferme toutes les positions ouvertes."""
        for symbol, position in self.positions.items():
            try:
                self.order_manager.close_position(symbol)
                logger.info(f"Position fermée: {symbol}")
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture de {symbol}: {e}")
                
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du servo."""
        return {
            'is_running': self.is_running,
            'open_positions': len(self.positions),
            'total_signals': len(self.signal_history),
            'performance_metrics': self.performance_metrics,
            'last_signal': self.signal_history[-1] if self.signal_history else None
        }
        
    def get_positions(self) -> Dict[str, Position]:
        """Retourne les positions ouvertes."""
        return self.positions.copy()
        
    def get_signal_history(self) -> List[TradingSignal]:
        """Retourne l'historique des signaux."""
        return self.signal_history.copy() 