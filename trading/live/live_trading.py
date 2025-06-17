#!/usr/bin/env python3
"""
Système de Trading Live avec MT5
Intégration sécurisée des modèles ML+RL pour le trading en temps réel
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import yaml
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger
import joblib
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

# Import des modules partagés
from utils.data_loading import get_mt5_connector, load_mt5_data
from utils.config import load_config
from utils.logging import setup_logger, log_trading_action, log_model_prediction, log_data_loading

# Import des modules du projet
from features.feature_engineering import FeatureEngineer
from models.ml_model import MLModel
from monitoring.feature_monitor import FeatureMonitor


class LiveTradingSystem:
    """Système de trading live avec gestion des risques avancée."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialise le système de trading live."""
        self.config = self._load_config(config_path)
        self.mt5_connected = False
        self.ml_model = None
        self.rl_model = None
        self.feature_engineer = None
        self.monitor = None
        self.account_info = None
        self.positions = []
        self.trades_history = []
        self.risk_manager = RiskManager(self.config)
        self.entry_detector = EntryPointDetector()  # Nouveau détecteur
        
        # Initialisation des modèles
        self._initialize_models()
        
        # Configuration du logging
        self._setup_logging()
        
        logger.info("🚀 Système de Trading Live initialisé")
    
    def _load_config(self, config_path):
        """Charge la configuration."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✅ Configuration chargée")
            return config
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de la config: {e}")
            raise
    
    def _setup_logging(self):
        """Configure le logging pour le trading live."""
        logger.add(
            f"logs/live_trading_{datetime.now().strftime('%Y%m%d')}.log",
            rotation="1 day",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    def _initialize_models(self):
        """Initialise les modèles ML et RL avec gestion d'erreurs robuste."""
        try:
            # Chargement du modèle ML
            symbol = self.config['training']['symbol']
            timeframe = self.config['training']['timeframes'][0]
            model_path = f"models/saved/xgboost/{symbol}/{timeframe}/ml_model.joblib"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle ML non trouvé: {model_path}")
                
            self.ml_model = joblib.load(model_path)
            logger.info(f"✅ Modèle ML chargé: {model_path}")
            
            # Chargement du modèle RL avec gestion d'erreurs
            self.rl_model = self._load_rl_model_with_fallback()
            
            # Chargement du feature engineer sauvegardé
            feature_engineer_path = f"models/saved/xgboost/{symbol}/{timeframe}/feature_engineer.joblib"
            
            if not os.path.exists(feature_engineer_path):
                logger.warning(f"⚠️ Feature engineer non trouvé: {feature_engineer_path}")
                self.feature_engineer = None
            else:
                self.feature_engineer = joblib.load(feature_engineer_path)
                logger.info(f"✅ Feature engineer chargé: {feature_engineer_path}")
            
            # Initialisation du monitor
            self.monitor = FeatureMonitor(self.config)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des modèles: {e}")
            raise
    
    def _load_rl_model_with_fallback(self, max_retries: int = 3) -> Optional[PPO]:
        """
        Charge le modèle RL avec plusieurs tentatives et fallbacks.
        
        Args:
            max_retries: Nombre maximum de tentatives
            
        Returns:
            Modèle RL chargé ou None si échec
        """
        rl_paths = [
            "models/ppo_smoteenn/ppo_smoteenn_final.zip",
            "models/ppo_smoteenn/best_model.zip",
            "models/ppo_smoteenn/ppo_smoteenn_100000_steps.zip"
        ]
        
        for attempt in range(max_retries):
            for rl_path in rl_paths:
                try:
                    if os.path.exists(rl_path):
                        logger.info(f"🔄 Tentative {attempt + 1}: Chargement RL depuis {rl_path}")
                        rl_model = PPO.load(rl_path)
                        logger.info(f"✅ Modèle RL chargé avec succès: {rl_path}")
                        return rl_model
                    else:
                        logger.debug(f"📁 Fichier non trouvé: {rl_path}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Échec du chargement RL depuis {rl_path}: {e}")
                    continue
        
        # Si tous les chemins échouent, essayer de charger depuis le wrapper RL
        try:
            from models.rl.rl_model import RLModel
            rl_wrapper = RLModel()
            if rl_wrapper.is_trained:
                logger.info("✅ Modèle RL chargé via wrapper")
                return rl_wrapper.model
        except Exception as e:
            logger.warning(f"⚠️ Échec du chargement via wrapper RL: {e}")
        
        logger.warning("⚠️ Aucun modèle RL disponible - Trading en mode ML uniquement")
        return None
    
    def connect_mt5(self):
        """Connexion sécurisée à MT5."""
        try:
            # Initialisation de MT5
            if not mt5.initialize():
                logger.error(f"❌ Échec de l'initialisation MT5: {mt5.last_error()}")
                return False
            
            # Connexion au compte
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ Impossible de récupérer les informations du compte")
                return False
            
            self.account_info = {
                'login': account_info.login,
                'server': account_info.server,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'leverage': account_info.leverage,
                'currency': account_info.currency
            }
            
            logger.info(f"✅ Connecté à MT5 - Compte: {self.account_info['login']}")
            logger.info(f"💰 Balance: {self.account_info['balance']:.2f} {self.account_info['currency']}")
            logger.info(f"📊 Equity: {self.account_info['equity']:.2f} {self.account_info['currency']}")
            logger.info(f"🔧 Levier: 1:{self.account_info['leverage']}")
            
            self.mt5_connected = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur de connexion MT5: {e}")
            return False
    
    def get_market_data(self, symbol, timeframe, bars=100, max_retries=3):
        """Récupère les données de marché en temps réel avec retry mechanism."""
        for attempt in range(max_retries):
            try:
                # Conversion du timeframe
                tf_map = {
                    'M1': mt5.TIMEFRAME_M1,
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'M30': mt5.TIMEFRAME_M30,
                    'H1': mt5.TIMEFRAME_H1,
                    'H4': mt5.TIMEFRAME_H4,
                    'D1': mt5.TIMEFRAME_D1
                }
                
                mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
                
                # Récupération des données
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
                if rates is None:
                    raise ValueError(f"Aucune donnée reçue pour {symbol}")
                
                # Conversion en DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Validation des données
                if len(df) < bars * 0.8:  # Au moins 80% des barres demandées
                    raise ValueError(f"Données insuffisantes: {len(df)}/{bars} barres")
                
                logger.info(f"📊 Données récupérées: {symbol} {timeframe} - {len(df)} barres")
                return df
                
            except Exception as e:
                logger.warning(f"⚠️ Tentative {attempt + 1}/{max_retries} échouée: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"❌ Impossible de récupérer les données après {max_retries} tentatives")
                    return None
        
        return None
    
    def generate_features(self, df):
        """Génère les features pour la prédiction."""
        try:
            # Ajouter une colonne volume factice pour MT5 (les données MT5 n'ont pas de volume)
            if 'volume' not in df.columns:
                df['volume'] = 1000  # Volume factice pour MT5
            
            # Vérifier que toutes les colonnes nécessaires sont présentes
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Colonnes manquantes: {missing_cols}")
                return None
            
            # Créer une copie pour éviter de modifier l'original
            features_df = df.copy()
            
            # Calculer les features de base
            features_df['returns'] = features_df['close'].pct_change()
            features_df['returns_5'] = features_df['close'].pct_change(5)
            features_df['returns_10'] = features_df['close'].pct_change(10)
            
            # Volatilité
            features_df['volatility'] = features_df['returns'].rolling(window=20).std()
            features_df['volatility_5'] = features_df['returns'].rolling(window=5).std()
            
            # Moyennes mobiles
            features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
            features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
            features_df['sma_ratio'] = features_df['close'] / features_df['sma_20']
            
            # Volume
            features_df['volume_ma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma']
            
            # Range
            features_df['range'] = features_df['high'] - features_df['low']
            features_df['range_ma'] = features_df['range'].rolling(window=20).mean()
            
            # RSI
            delta = features_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = features_df['close'].ewm(span=12).mean()
            exp2 = features_df['close'].ewm(span=26).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            
            # Supprimer les lignes avec NaN
            features_df = features_df.dropna()
            
            # Debug: vérifier le type de features
            logger.info(f"🔍 Type de features retourné: {type(features_df)}")
            if isinstance(features_df, pd.DataFrame):
                logger.info(f"🔍 Shape du DataFrame: {features_df.shape}")
                logger.info(f"🔍 Colonnes: {list(features_df.columns)}")
            elif isinstance(features_df, np.ndarray):
                logger.info(f"🔍 Shape du numpy array: {features_df.shape}")
            
            return features_df
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération des features: {e}")
            return None
    
    def get_ml_prediction(self, features, max_retries=2):
        """Obtient la prédiction du modèle ML avec retry mechanism."""
        for attempt in range(max_retries):
            try:
                logger.info("🔍 Début get_ml_prediction")
                
                # Vérifier que features est un DataFrame
                if not isinstance(features, pd.DataFrame):
                    raise ValueError(f"Type de features non supporté: {type(features)}")
                
                logger.info(f"✅ Features est un DataFrame: {features.shape}")
                
                # Utiliser seulement les colonnes requises par le modèle dans l'ordre exact
                required_columns = [
                    'returns', 'returns_5', 'returns_10', 'volatility', 'volatility_5', 
                    'sma_5', 'sma_20', 'sma_ratio', 'rsi', 'macd', 'macd_signal', 
                    'volume_ma', 'volume_ratio', 'range', 'range_ma'
                ]
                
                logger.info(f"🔍 Colonnes requises: {required_columns}")
                logger.info(f"🔍 Colonnes disponibles: {list(features.columns)}")
                
                # Vérifier que toutes les colonnes requises sont présentes
                missing_cols = [col for col in required_columns if col not in features.columns]
                if missing_cols:
                    raise ValueError(f"Colonnes manquantes pour le modèle: {missing_cols}")
                
                logger.info("✅ Toutes les colonnes requises sont présentes")
                
                # Sélectionner seulement les colonnes requises dans l'ordre exact
                features_df = features[required_columns].iloc[-1:].copy()
                
                logger.info(f"🔍 Features pour ML: {features_df.shape}")
                logger.info(f"🔍 Colonnes ML: {list(features_df.columns)}")
                logger.info(f"🔍 Valeurs: {features_df.values}")
                
                # Vérifier les valeurs NaN
                if features_df.isna().any().any():
                    raise ValueError("Valeurs NaN détectées dans les features")
                
                logger.info("✅ Pas de valeurs NaN")
                
                # Prédiction ML
                logger.info("🤖 Appel du modèle ML...")
                prediction = self.ml_model.predict(features_df)[0]
                probabilities = self.ml_model.predict_proba(features_df)[0]
                confidence = np.max(probabilities)
                
                logger.info(f"✅ Prédiction obtenue: {prediction}")
                logger.info(f"✅ Probabilités obtenues: {probabilities}")
                logger.info(f"✅ Prédiction ML: {prediction}, Confiance: {confidence:.3f}")
                
                result = {
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'confidence': confidence
                }
                
                logger.info(f"✅ Prédiction ML obtenue: {result}")
                return result
                
            except Exception as e:
                logger.warning(f"⚠️ Tentative {attempt + 1}/{max_retries} échouée: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Short delay before retry
                else:
                    logger.error(f"❌ Échec de la prédiction ML après {max_retries} tentatives: {e}")
                    return None
        
        return None
    
    def get_rl_action(self, features, ml_prediction):
        """Obtient l'action du modèle RL avec gestion d'erreurs robuste."""
        try:
            # Vérifier si le modèle RL est disponible
            if self.rl_model is None:
                logger.debug("ℹ️ Modèle RL non disponible - utilisation de la prédiction ML uniquement")
                return {
                    'action': int(ml_prediction['prediction']) if ml_prediction else 0,
                    'confidence': ml_prediction.get('confidence', 0.5) if ml_prediction else 0.5,
                    'source': 'ml_fallback'
                }
            
            # Vérifier le type de features et convertir si nécessaire
            if isinstance(features, np.ndarray):
                # Si c'est un numpy array, on prend la dernière ligne
                observation = features[-1:].flatten()
            elif isinstance(features, pd.DataFrame):
                # Si c'est un DataFrame, on sélectionne les colonnes de features
                # Exclure les colonnes OHLCV de base
                exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'tick_volume', 'spread', 'real_volume']
                feature_columns = [col for col in features.columns if col not in exclude_cols]
                observation = features[feature_columns].iloc[-1:].values.flatten()
            else:
                logger.error(f"❌ Type de features non supporté: {type(features)}")
                return None
            
            # Normalisation et clipping
            observation = np.clip(observation, -10, 10)
            
            # Vérifier les valeurs NaN
            if np.any(np.isnan(observation)):
                logger.warning("⚠️ Valeurs NaN détectées dans l'observation RL - utilisation du fallback ML")
                return {
                    'action': int(ml_prediction['prediction']) if ml_prediction else 0,
                    'confidence': ml_prediction.get('confidence', 0.5) if ml_prediction else 0.5,
                    'source': 'ml_fallback_nan'
                }
            
            # Prédiction RL avec gestion d'erreurs
            try:
                action, _ = self.rl_model.predict(observation, deterministic=True)
                logger.debug(f"🤖 Action RL prédite: {action}")
                
                return {
                    'action': int(action),
                    'observation': observation,
                    'source': 'rl_model'
                }
                
            except Exception as rl_error:
                logger.warning(f"⚠️ Erreur lors de la prédiction RL: {rl_error} - fallback vers ML")
                return {
                    'action': int(ml_prediction['prediction']) if ml_prediction else 0,
                    'confidence': ml_prediction.get('confidence', 0.5) if ml_prediction else 0.5,
                    'source': 'ml_fallback_error'
                }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction RL: {e}")
            # Fallback vers ML en cas d'erreur
            return {
                'action': int(ml_prediction['prediction']) if ml_prediction else 0,
                'confidence': ml_prediction.get('confidence', 0.5) if ml_prediction else 0.5,
                'source': 'ml_fallback_exception'
            }
    
    def get_current_positions(self):
        """Récupère les positions actuelles."""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            self.positions = []
            for pos in positions:
                self.positions.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'time': datetime.fromtimestamp(pos.time)
                })
            
            return self.positions
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des positions: {e}")
            return []
    
    def calculate_position_size(self, symbol, action_confidence):
        """Calcule la taille de position basée sur le risque en pips."""
        try:
            # Récupération des informations du compte
            account_info = mt5.account_info()
            if account_info is None:
                return 0.01
            
            # Récupération des informations du symbole
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01
            
            # Paramètres de risque
            balance = account_info.balance
            risk_percent = self.config['risk_management']['position_size']  # 2%
            max_risk_amount = balance * risk_percent
            
            # Ajustement selon la confiance (plus conservateur)
            confidence_multiplier = min(action_confidence, 1.0)  # Max 100% même avec haute confiance
            adjusted_risk = max_risk_amount * confidence_multiplier
            
            # Paramètres de SL/TP
            sl_pips = 50  # Stop-loss de 50 pips
            tp_pips = 75  # Take-profit de 75 pips
            
            # Calcul de la valeur d'un pip
            point = symbol_info.point
            contract_size = symbol_info.trade_contract_size  # Taille du contrat (100,000 pour forex)
            
            # Valeur d'un pip = (point * contract_size) / 10
            pip_value = (point * contract_size) / 10
            
            # Calcul du volume basé sur le risque en pips
            if pip_value > 0:
                volume = adjusted_risk / (sl_pips * pip_value)
            else:
                # Fallback: calcul basé sur tick_value
                tick_value = symbol_info.trade_tick_value
                if tick_value > 0:
                    volume = adjusted_risk / (sl_pips * tick_value)
                else:
                    volume = 0.01
            
            # Limites du broker
            min_volume = symbol_info.volume_min
            max_volume = symbol_info.volume_max
            volume_step = symbol_info.volume_step
            
            # Limite de sécurité supplémentaire (max 1% du capital par lot)
            max_safe_volume = balance * 0.01 / 100000  # 1 lot = 100,000 USD
            max_volume = min(max_volume, max_safe_volume)
            
            # Ajustement aux limites
            volume = max(min_volume, min(max_volume, volume))
            volume = round(volume / volume_step) * volume_step
            
            # Calcul du risque réel
            actual_risk = volume * sl_pips * pip_value
            potential_profit = volume * tp_pips * pip_value
            
            logger.info(f"📊 Calcul position - Balance: {balance:.2f}, Risk: {adjusted_risk:.2f}, "
                       f"SL: {sl_pips} pips, TP: {tp_pips} pips, Volume: {volume:.2f} lots")
            logger.info(f"💰 Risque réel: {actual_risk:.2f} USD, Profit potentiel: {potential_profit:.2f} USD")
            
            return volume
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul de la taille de position: {e}")
            return 0.01
    
    def execute_trade(self, symbol, action, volume, price=None):
        """Exécute un ordre de trading avec SL/TP automatiques."""
        try:
            # Vérifications de sécurité
            if not self.risk_manager.check_trading_allowed():
                logger.warning("⚠️ Trading bloqué par le gestionnaire de risques")
                return None
            
            # Récupération du prix actuel
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logger.error(f"❌ Impossible de récupérer le prix pour {symbol}")
                    return None
                price = tick.ask if action == 1 else tick.bid
            
            # Calcul des SL/TP en pips
            sl_pips = 50  # Stop-loss de 50 pips
            tp_pips = 75  # Take-profit de 75 pips (ratio 1:1.5)
            
            # Calcul des prix SL/TP
            point = mt5.symbol_info(symbol).point
            if action == 1:  # BUY
                sl_price = price - (sl_pips * point)
                tp_price = price + (tp_pips * point)
            else:  # SELL
                sl_price = price + (sl_pips * point)
                tp_price = price - (tp_pips * point)
            
            # Préparation de l'ordre
            order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
            
            # Paramètres de l'ordre avec SL/TP
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,  # Déviation maximale en points
                "magic": 123456,  # Identifiant magique
                "comment": f"ML+RL Trade {datetime.now().strftime('%H:%M:%S')}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Exécution de l'ordre
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Erreur d'exécution: {result.retcode} - {result.comment}")
                return None
            
            # Enregistrement du trade
            trade_info = {
                'ticket': result.order,
                'symbol': symbol,
                'action': 'BUY' if action == 1 else 'SELL',
                'volume': volume,
                'price': price,
                'sl': sl_price,
                'tp': tp_price,
                'time': datetime.now(),
                'retcode': result.retcode
            }
            
            self.trades_history.append(trade_info)
            logger.info(f"✅ Trade exécuté: {trade_info['action']} {volume} lots {symbol} @ {price} "
                       f"(SL: {sl_price:.5f}, TP: {tp_price:.5f})")
            
            return trade_info
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution du trade: {e}")
            return None
    
    def run_live_trading(self, symbol=None, timeframe=None):
        """Lance le trading live avec détection de points d'entrée stratégiques."""
        if not self.mt5_connected:
            logger.error("❌ Non connecté à MT5")
            return
        
        symbol = symbol or self.config['training']['symbol']
        timeframe = timeframe or self.config['training']['timeframes'][0]
        
        logger.info(f"🚀 Démarrage du trading live: {symbol} {timeframe}")
        logger.info("🎯 Mode: Détection de points d'entrée stratégiques")
        
        # Variables pour éviter les trades continus
        last_action = None
        last_confidence = 0.0
        consecutive_holds = 0
        max_consecutive_holds = 5
        
        try:
            iteration = 0
            while True:
                iteration += 1
                logger.info(f"🔄 Itération {iteration}")
                
                # Récupération des données de marché
                df = self.get_market_data(symbol, timeframe)
                if df is None:
                    logger.warning("⚠️ Impossible de récupérer les données, attente...")
                    time.sleep(30)
                    continue
                
                # Génération des features
                features = self.generate_features(df)
                if features is None:
                    logger.warning("⚠️ Impossible de générer les features, attente...")
                    time.sleep(30)
                    continue
                
                # Prédiction ML
                logger.info("🤖 Début de la prédiction ML...")
                ml_result = self.get_ml_prediction(features)
                if ml_result is None:
                    logger.warning("⚠️ Impossible d'obtenir la prédiction ML, attente...")
                    time.sleep(30)
                    continue
                
                logger.info(f"✅ Prédiction ML obtenue: {ml_result}")
                
                # Détection des points d'entrée stratégiques
                logger.info("🎯 Analyse des points d'entrée...")
                entry_analysis = self.entry_detector.detect_entry_points(
                    features, ml_result['prediction'], ml_result['confidence']
                )
                
                if entry_analysis is None:
                    logger.warning("⚠️ Erreur dans l'analyse des points d'entrée")
                    time.sleep(30)
                    continue
                
                # Affichage détaillé de l'analyse
                self._display_entry_analysis(entry_analysis)
                
                # Décision de trading basée sur les points d'entrée
                entry_point = entry_analysis['entry_point']
                signal_strength = entry_analysis['signal_strength']
                convergence = entry_analysis['convergence']
                
                logger.info(f"🎯 Point d'entrée détecté: {entry_point['action']} - {entry_point['reason']}")
                logger.info(f"📊 Force du signal: {signal_strength:.2f}, Convergence: {convergence:.2f}")
                
                # Logique de trading intelligente
                should_trade = False
                action = 0
                
                if entry_point['action'] in ['BUY', 'SELL']:
                    # Vérifier si c'est un nouveau signal ou une confirmation
                    if (last_action != entry_point['action'] or 
                        signal_strength > last_confidence + 0.1):
                        
                        should_trade = True
                        action = 1 if entry_point['action'] == 'BUY' else 2
                        consecutive_holds = 0
                        
                        logger.info(f"✅ Signal de trading valide: {entry_point['action']}")
                    else:
                        logger.info(f"ℹ️ Signal déjà actif: {entry_point['action']}")
                else:
                    consecutive_holds += 1
                    logger.info(f"⏸️ Aucun point d'entrée - Consécutifs: {consecutive_holds}")
                
                # Forcer une action après trop de holds consécutifs (seulement si signal modéré)
                if (consecutive_holds >= max_consecutive_holds and 
                    signal_strength >= 0.5 and 
                    entry_analysis['confidence'] >= 0.6):
                    
                    logger.info(f"🔄 Forçage d'action après {consecutive_holds} holds consécutifs")
                    should_trade = True
                    action = ml_result['prediction'] if ml_result['prediction'] in [1, 2] else 1
                    consecutive_holds = 0
                
                if should_trade:
                    logger.info(f"📈 Action décidée: {action} (Buy=1, Sell=2)")
                    
                    # Vérification des conditions de trading
                    if self.risk_manager.check_trading_conditions(symbol, action, signal_strength):
                        logger.info("✅ Conditions de trading remplies")
                        
                        # Calcul de la taille de position optimisée
                        volume = self.calculate_position_size(symbol, signal_strength)
                        logger.info(f"📊 Volume calculé: {volume}")
                        
                        # Exécution du trade
                        trade_result = self.execute_trade(symbol, action, volume)
                        if trade_result:
                            logger.info(f"✅ Trade exécuté: {trade_result}")
                            last_action = entry_point['action']
                            last_confidence = signal_strength
                        else:
                            logger.warning("⚠️ Échec de l'exécution du trade")
                    else:
                        logger.info("ℹ️ Conditions de trading non remplies")
                else:
                    logger.info(f"ℹ️ Aucune action - Signal: {signal_strength:.3f}, Action précédente: {last_action}")
                
                # Attente adaptative
                if should_trade:
                    wait_time = 30  # Plus court après un trade
                elif signal_strength > 0.7:
                    wait_time = 45  # Moyen si signal fort
                else:
                    wait_time = 60  # Plus long si pas de signal
                
                logger.info(f"⏳ Attente de {wait_time} secondes...")
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt du trading live demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"❌ Erreur dans le trading live: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.disconnect()
    
    def _display_entry_analysis(self, analysis):
        """Affiche l'analyse détaillée des points d'entrée."""
        try:
            logger.info("📊 ANALYSE DES POINTS D'ENTRÉE:")
            logger.info(f"   🎯 Action: {analysis['entry_point']['action']}")
            logger.info(f"   📝 Raison: {analysis['entry_point']['reason']}")
            logger.info(f"   💪 Force du signal: {analysis['signal_strength']:.2f}")
            logger.info(f"   🔗 Convergence: {analysis['convergence']:.2f}")
            logger.info(f"   🤖 Confiance ML: {analysis['confidence']:.2f}")
            
            # Signaux techniques
            logger.info("   📈 Signaux techniques:")
            logger.info(f"      Tendance: {analysis['trend_signal']['direction']} ({analysis['trend_signal']['strength']:.1f})")
            logger.info(f"      Momentum: {analysis['momentum_signal']['signal']} ({analysis['momentum_signal']['strength']:.1f})")
            logger.info(f"      Volatilité: {analysis['volatility_signal']['condition']} ({analysis['volatility_signal']['strength']:.1f})")
            logger.info(f"      S/R: {analysis['sr_signal']['level']} ({analysis['sr_signal']['strength']:.1f})")
            logger.info(f"      Volume: {analysis['volume_signal']['condition']} ({analysis['volume_signal']['strength']:.1f})")
            
        except Exception as e:
            logger.error(f"❌ Erreur affichage analyse: {e}")
    
    def disconnect(self):
        """Déconnexion de MT5."""
        if self.mt5_connected:
            mt5.shutdown()
            self.mt5_connected = False
            logger.info("🔌 Déconnecté de MT5")


class RiskManager:
    """Gestionnaire de risques avancé."""
    
    def __init__(self, config):
        self.config = config
        self.daily_loss = 0
        self.daily_trades = 0
        self.max_daily_loss = config['risk_management']['max_daily_loss']
        self.max_daily_trades = config['risk_management'].get('max_daily_trades', 10)
        self.last_reset = datetime.now().date()
    
    def check_trading_allowed(self):
        """Vérifie si le trading est autorisé."""
        # Reset quotidien
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            self.daily_loss = 0
            self.daily_trades = 0
            self.last_reset = current_date
        
        # Vérifications
        if self.daily_loss >= self.max_daily_loss:
            logger.warning("⚠️ Limite de perte quotidienne atteinte")
            return False
        
        if self.daily_trades >= self.max_daily_trades:
            logger.warning("⚠️ Limite de trades quotidiens atteinte")
            return False
        
        return True
    
    def check_trading_conditions(self, symbol, action, confidence):
        """Vérifie les conditions de trading."""
        # Vérification de la confiance minimale
        min_confidence = self.config['risk_management'].get('min_confidence', 0.6)
        if confidence < min_confidence:
            return False
        
        # Vérification des heures de trading
        current_hour = datetime.now().hour
        trading_hours = self.config['risk_management'].get('trading_hours', [0, 24])
        if not (trading_hours[0] <= current_hour <= trading_hours[1]):
            return False
        
        return True
    
    def update_trading_stats(self, trade_result):
        """Met à jour les statistiques de trading."""
        self.daily_trades += 1
        # La perte sera calculée lors de la fermeture des positions


class EntryPointDetector:
    """Détecteur de points d'entrée stratégiques."""
    
    def __init__(self):
        self.last_signal = None
        self.signal_strength = 0.0
        self.confirmation_count = 0
        self.min_confirmation = 2  # Nombre minimum de confirmations
        
    def detect_entry_points(self, features, ml_prediction, confidence):
        """Détecte les points d'entrée stratégiques."""
        try:
            # Récupération des dernières valeurs
            current_data = features.iloc[-1]
            
            # 1. Signaux de tendance
            trend_signal = self._analyze_trend(features)
            
            # 2. Signaux de momentum
            momentum_signal = self._analyze_momentum(current_data)
            
            # 3. Signaux de volatilité
            volatility_signal = self._analyze_volatility(features)
            
            # 4. Signaux de support/résistance
            sr_signal = self._analyze_support_resistance(features)
            
            # 5. Signaux de volume
            volume_signal = self._analyze_volume(current_data)
            
            # 6. Convergence des signaux
            convergence = self._check_signal_convergence(
                ml_prediction, trend_signal, momentum_signal, 
                volatility_signal, sr_signal, volume_signal
            )
            
            # 7. Force du signal
            signal_strength = self._calculate_signal_strength(
                confidence, trend_signal, momentum_signal, 
                volatility_signal, sr_signal, volume_signal
            )
            
            # 8. Décision finale
            entry_point = self._make_entry_decision(
                ml_prediction, convergence, signal_strength, confidence
            )
            
            return {
                'entry_point': entry_point,
                'signal_strength': signal_strength,
                'trend_signal': trend_signal,
                'momentum_signal': momentum_signal,
                'volatility_signal': volatility_signal,
                'sr_signal': sr_signal,
                'volume_signal': volume_signal,
                'convergence': convergence,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur dans la détection des points d'entrée: {e}")
            return None
    
    def _analyze_trend(self, features):
        """Analyse la tendance du marché."""
        try:
            # Utiliser les moyennes mobiles
            sma_5 = features['sma_5'].iloc[-1]
            sma_20 = features['sma_20'].iloc[-1]
            sma_ratio = features['sma_ratio'].iloc[-1]
            
            # Tendance haussière
            if sma_5 > sma_20 and sma_ratio > 1.001:
                return {'direction': 'bullish', 'strength': min(sma_ratio - 1, 0.1) * 100}
            
            # Tendance baissière
            elif sma_5 < sma_20 and sma_ratio < 0.999:
                return {'direction': 'bearish', 'strength': min(1 - sma_ratio, 0.1) * 100}
            
            # Pas de tendance claire
            else:
                return {'direction': 'neutral', 'strength': 0}
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse tendance: {e}")
            return {'direction': 'neutral', 'strength': 0}
    
    def _analyze_momentum(self, current_data):
        """Analyse le momentum du marché."""
        try:
            rsi = current_data['rsi']
            macd = current_data['macd']
            macd_signal = current_data['macd_signal']
            
            # RSI conditions
            rsi_oversold = rsi < 30
            rsi_overbought = rsi > 70
            rsi_neutral = 40 <= rsi <= 60
            
            # MACD conditions
            macd_bullish = macd > macd_signal and macd > 0
            macd_bearish = macd < macd_signal and macd < 0
            
            # Signaux de momentum
            if rsi_oversold and macd_bullish:
                return {'signal': 'bullish_reversal', 'strength': 80}
            elif rsi_overbought and macd_bearish:
                return {'signal': 'bearish_reversal', 'strength': 80}
            elif rsi_neutral and macd_bullish:
                return {'signal': 'bullish_continuation', 'strength': 60}
            elif rsi_neutral and macd_bearish:
                return {'signal': 'bearish_continuation', 'strength': 60}
            else:
                return {'signal': 'neutral', 'strength': 0}
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse momentum: {e}")
            return {'signal': 'neutral', 'strength': 0}
    
    def _analyze_volatility(self, features):
        """Analyse la volatilité du marché."""
        try:
            current_vol = features['volatility'].iloc[-1]
            vol_5 = features['volatility_5'].iloc[-1]
            
            # Volatilité élevée = opportunités
            if current_vol > vol_5 * 1.2:
                return {'condition': 'high_volatility', 'strength': 70}
            elif current_vol < vol_5 * 0.8:
                return {'condition': 'low_volatility', 'strength': 30}
            else:
                return {'condition': 'normal_volatility', 'strength': 50}
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse volatilité: {e}")
            return {'condition': 'normal_volatility', 'strength': 50}
    
    def _analyze_support_resistance(self, features):
        """Analyse les niveaux de support/résistance."""
        try:
            # Utiliser les données des dernières barres
            recent_highs = features['high'].tail(20).max()
            recent_lows = features['low'].tail(20).min()
            current_price = features['close'].iloc[-1]
            
            # Distance aux niveaux clés
            distance_to_high = (recent_highs - current_price) / current_price
            distance_to_low = (current_price - recent_lows) / current_price
            
            # Proximité des niveaux
            if distance_to_high < 0.001:  # Proche de la résistance
                return {'level': 'resistance', 'strength': 80}
            elif distance_to_low < 0.001:  # Proche du support
                return {'level': 'support', 'strength': 80}
            else:
                return {'level': 'neutral', 'strength': 0}
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse support/résistance: {e}")
            return {'level': 'neutral', 'strength': 0}
    
    def _analyze_volume(self, current_data):
        """Analyse le volume."""
        try:
            volume_ratio = current_data['volume_ratio']
            
            if volume_ratio > 1.5:  # Volume élevé
                return {'condition': 'high_volume', 'strength': 70}
            elif volume_ratio < 0.5:  # Volume faible
                return {'condition': 'low_volume', 'strength': 30}
            else:
                return {'condition': 'normal_volume', 'strength': 50}
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse volume: {e}")
            return {'condition': 'normal_volume', 'strength': 50}
    
    def _check_signal_convergence(self, ml_prediction, trend, momentum, volatility, sr, volume):
        """Vérifie la convergence des signaux."""
        try:
            convergence_score = 0
            total_signals = 0
            
            # ML prediction
            if ml_prediction in [1, 2]:  # Buy ou Sell
                convergence_score += 1
            total_signals += 1
            
            # Tendance
            if trend['direction'] != 'neutral':
                convergence_score += 1
            total_signals += 1
            
            # Momentum
            if momentum['signal'] != 'neutral':
                convergence_score += 1
            total_signals += 1
            
            # Support/Résistance
            if sr['level'] != 'neutral':
                convergence_score += 1
            total_signals += 1
            
            # Volume
            if volume['condition'] == 'high_volume':
                convergence_score += 1
            total_signals += 1
            
            convergence_rate = convergence_score / total_signals
            return convergence_rate
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification convergence: {e}")
            return 0
    
    def _calculate_signal_strength(self, ml_confidence, trend, momentum, volatility, sr, volume):
        """Calcule la force globale du signal."""
        try:
            # Pondération des signaux
            weights = {
                'ml': 0.4,      # ML a le plus de poids
                'trend': 0.2,   # Tendance importante
                'momentum': 0.2, # Momentum important
                'sr': 0.15,     # Support/Résistance
                'volume': 0.05  # Volume moins important
            }
            
            # Calcul du score pondéré
            score = (
                ml_confidence * weights['ml'] +
                trend['strength'] / 100 * weights['trend'] +
                momentum['strength'] / 100 * weights['momentum'] +
                sr['strength'] / 100 * weights['sr'] +
                volume['strength'] / 100 * weights['volume']
            )
            
            return min(score, 1.0)  # Normaliser entre 0 et 1
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul force signal: {e}")
            return 0
    
    def _make_entry_decision(self, ml_prediction, convergence, signal_strength, confidence):
        """Prend la décision finale d'entrée."""
        try:
            # Conditions pour un point d'entrée valide
            min_convergence = 0.6  # 60% des signaux doivent converger
            min_signal_strength = 0.7  # Force minimale du signal
            min_confidence = 0.7  # Confiance ML minimale
            
            # Vérification des conditions
            if (convergence >= min_convergence and 
                signal_strength >= min_signal_strength and 
                confidence >= min_confidence and 
                ml_prediction in [1, 2]):
                
                # Déterminer la direction
                if ml_prediction == 1:
                    return {'action': 'BUY', 'reason': 'strong_bullish_signal'}
                else:
                    return {'action': 'SELL', 'reason': 'strong_bearish_signal'}
            
            # Conditions pour un signal de confirmation
            elif (convergence >= 0.4 and 
                  signal_strength >= 0.5 and 
                  confidence >= 0.6):
                
                if ml_prediction == 1:
                    return {'action': 'BUY', 'reason': 'moderate_bullish_signal'}
                elif ml_prediction == 2:
                    return {'action': 'SELL', 'reason': 'moderate_bearish_signal'}
            
            # Pas de signal valide
            return {'action': 'HOLD', 'reason': 'no_clear_signal'}
            
        except Exception as e:
            logger.error(f"❌ Erreur décision entrée: {e}")
            return {'action': 'HOLD', 'reason': 'error'}


def main():
    """Fonction principale."""
    logger.info("🚀 Démarrage du système de trading live")
    
    # Initialisation du système
    trading_system = LiveTradingSystem()
    
    # Connexion à MT5
    if not trading_system.connect_mt5():
        logger.error("❌ Impossible de se connecter à MT5")
        return
    
    # Affichage des informations du compte
    logger.info("📊 Informations du compte:")
    logger.info(f"   Balance: {trading_system.account_info['balance']:.2f}")
    logger.info(f"   Equity: {trading_system.account_info['equity']:.2f}")
    logger.info(f"   Marge libre: {trading_system.account_info['free_margin']:.2f}")
    logger.info(f"   Levier: 1:{trading_system.account_info['leverage']}")
    
    # Démarrage du trading live
    try:
        trading_system.run_live_trading()
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
    finally:
        trading_system.disconnect()


if __name__ == "__main__":
    main()
