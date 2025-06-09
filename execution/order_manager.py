"""
Module de gestion des ordres pour l'exécution via MetaTrader 5.

Gère la connexion au broker, l'envoi d'ordres, la gestion des positions
et le monitoring des transactions.
"""

from typing import Dict, List, Optional, Tuple, Union
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import threading
from queue import Queue
import signal

@dataclass
class OrderRequest:
    """Classe représentant une requête d'ordre."""
    symbol: str
    order_type: str  # 'market', 'limit', 'stop'
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    magic: int = 0
    expiration: Optional[datetime] = None

@dataclass
class OrderResult:
    """Classe représentant le résultat d'un ordre."""
    request: OrderRequest
    order_id: Optional[int]
    status: str  # 'filled', 'rejected', 'cancelled', 'expired'
    fill_price: Optional[float]
    fill_time: Optional[datetime]
    error: Optional[str]
    fees: float = 0.0
    slippage: float = 0.0

class MT5ConnectionHandler(FileSystemEventHandler):
    """Handler pour la surveillance de la connexion MT5."""
    
    def __init__(self, order_manager: 'OrderManager'):
        """
        Initialise le handler.
        
        Args:
            order_manager: Instance du gestionnaire d'ordres
        """
        self.order_manager = order_manager
        self.last_check = time.time()
        
    def on_modified(self, event):
        """Vérifie la connexion MT5 lors d'une modification."""
        if time.time() - self.last_check < 1:  # Évite les doubles vérifications
            return
            
        self.last_check = time.time()
        if not self.order_manager.check_connection():
            logger.warning("Connexion MT5 perdue, tentative de reconnexion...")
            self.order_manager.reconnect()

class OrderManager:
    """Gestionnaire d'ordres pour MetaTrader 5."""
    
    def __init__(self, config: dict):
        """
        Initialise le gestionnaire d'ordres.
        
        Args:
            config: Configuration du broker
        """
        self.config = config
        self.connected = False
        self.order_queue = Queue()
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[int, Dict] = {}
        self.last_error = None
        self.error_count = 0
        
        # Paramètres d'exécution
        self.order_config = config['execution']['order']
        self.position_config = config['execution']['position']
        self.risk_config = config['execution']['risk']
        self.monitoring_config = config['execution']['monitoring']
        
        # Configuration du broker
        self.broker_config = config['broker']
        self.login = self.broker_config['login']
        self.password = self.broker_config['password']
        self.server = self.broker_config['server']
        self.timeout = self.broker_config['timeout']
        
        # Initialisation de MT5
        if not mt5.initialize():
            logger.error(f"Échec de l'initialisation MT5: {mt5.last_error()}")
            return
            
        # Connexion au compte
        self._connect()
        
        # Démarrage du watchdog
        self._start_watchdog()
        
        # Démarrage du thread de traitement des ordres
        self._start_order_thread()
        
    def _connect(self) -> bool:
        """
        Établit la connexion avec MT5.
        
        Returns:
            True si la connexion est réussie
        """
        try:
            # Tentative de connexion
            if not mt5.login(
                login=self.login,
                password=self.password,
                server=self.server,
                timeout=self.timeout
            ):
                error = mt5.last_error()
                logger.error(f"Échec de la connexion MT5: {error}")
                self.last_error = error
                return False
                
            # Vérification de la connexion
            if not mt5.terminal_info():
                logger.error("Terminal MT5 non disponible")
                return False
                
            # Récupération des informations du compte
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Impossible de récupérer les informations du compte")
                return False
                
            logger.info(f"Connecté au compte {account_info.login} ({account_info.server})")
            logger.info(f"Balance: {account_info.balance:.2f}, Equity: {account_info.equity:.2f}")
            
            self.connected = True
            self.error_count = 0
            return True
            
        except Exception as e:
            logger.exception(f"Erreur lors de la connexion MT5: {e}")
            self.last_error = str(e)
            return False
            
    def _start_watchdog(self) -> None:
        """Démarre la surveillance de la connexion."""
        event_handler = MT5ConnectionHandler(self)
        observer = Observer()
        observer.schedule(
            event_handler,
            path='logs',
            recursive=False
        )
        observer.start()
        
    def _start_order_thread(self) -> None:
        """Démarre le thread de traitement des ordres."""
        def order_processor():
            while True:
                try:
                    # Récupération d'un ordre de la queue
                    order_request = self.order_queue.get()
                    
                    # Tentative d'exécution
                    result = self._execute_order(order_request)
                    
                    # Log du résultat
                    if result.status == 'success':
                        logger.info(
                            f"Ordre exécuté: {order_request.symbol} "
                            f"{order_request.order_type} {order_request.volume}"
                        )
                    else:
                        logger.error(
                            f"Échec de l'ordre: {order_request.symbol} "
                            f"{order_request.order_type} - {result.error}"
                        )
                        
                    # Mise à jour de la queue
                    self.order_queue.task_done()
                    
                except Exception as e:
                    logger.exception(f"Erreur dans le thread d'ordres: {e}")
                    time.sleep(1)
                    
        # Démarrage du thread
        thread = threading.Thread(target=order_processor, daemon=True)
        thread.start()
        
    def check_connection(self) -> bool:
        """
        Vérifie l'état de la connexion.
        
        Returns:
            True si la connexion est active
        """
        if not self.connected:
            return self._connect()
            
        try:
            # Test de la connexion
            if not mt5.terminal_info():
                logger.warning("Connexion MT5 perdue, tentative de reconnexion")
                self.connected = False
                return self._connect()
                
            return True
            
        except Exception as e:
            logger.exception(f"Erreur lors de la vérification de la connexion: {e}")
            self.connected = False
            return self._connect()
            
    def _execute_order(self, request: OrderRequest) -> OrderResult:
        """
        Exécute un ordre.
        
        Args:
            request: Requête d'ordre
            
        Returns:
            Résultat de l'ordre
        """
        if not self.connected and not self._connect():
            return OrderResult(
                request=request,
                order_id=None,
                status='error',
                fill_price=None,
                fill_time=None,
                error="Non connecté à MT5"
            )
            
        try:
            # Préparation de la requête
            symbol_info = mt5.symbol_info(request.symbol)
            if symbol_info is None:
                return OrderResult(
                    request=request,
                    order_id=None,
                    status='error',
                    fill_price=None,
                    fill_time=None,
                    error=f"Symbole {request.symbol} non trouvé"
                )
                
            # Vérification du trading
            if not symbol_info.visible:
                if not mt5.symbol_select(request.symbol, True):
                    return OrderResult(
                        request=request,
                        order_id=None,
                        status='error',
                        fill_price=None,
                        fill_time=None,
                        error=f"Impossible de sélectionner {request.symbol}"
                    )
                    
            # Préparation des paramètres
            point = symbol_info.point
            price = request.price or mt5.symbol_info_tick(request.symbol).ask
            
            # Ajustement des niveaux
            if request.stop_loss:
                request.stop_loss = round(request.stop_loss / point) * point
            if request.take_profit:
                request.take_profit = round(request.take_profit / point) * point
                
            # Création de la requête
            if request.order_type == 'market':
                trade_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": request.symbol,
                    "volume": request.volume,
                    "type": mt5.ORDER_TYPE_BUY if request.volume > 0 else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "deviation": 10,
                    "magic": request.magic,
                    "comment": request.comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                if request.stop_loss:
                    trade_request["sl"] = request.stop_loss
                if request.take_profit:
                    trade_request["tp"] = request.take_profit
                    
            elif request.order_type == 'limit':
                trade_request = {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": request.symbol,
                    "volume": request.volume,
                    "type": mt5.ORDER_TYPE_BUY_LIMIT if request.volume > 0 else mt5.ORDER_TYPE_SELL_LIMIT,
                    "price": request.price,
                    "deviation": 10,
                    "magic": request.magic,
                    "comment": request.comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                if request.stop_loss:
                    trade_request["sl"] = request.stop_loss
                if request.take_profit:
                    trade_request["tp"] = request.take_profit
                if request.expiration:
                    trade_request["expiration"] = int(request.expiration.timestamp())
                    
            else:
                return OrderResult(
                    request=request,
                    order_id=None,
                    status='error',
                    fill_price=None,
                    fill_time=None,
                    error=f"Type d'ordre non supporté: {request.order_type}"
                )
                
            # Envoi de l'ordre
            result = mt5.order_send(trade_request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return OrderResult(
                    request=request,
                    order_id=None,
                    status='error',
                    fill_price=None,
                    fill_time=None,
                    error=result.comment
                )
                
            # Succès
            return OrderResult(
                request=request,
                order_id=result.order,
                status='success',
                fill_price=result.price,
                fill_time=datetime.now(),
                error=None
            )
            
        except Exception as e:
            logger.exception(f"Erreur lors de l'exécution de l'ordre: {e}")
            return OrderResult(
                request=request,
                order_id=None,
                status='error',
                fill_price=None,
                fill_time=None,
                error=str(e)
            )
            
    def send_order(self, request: OrderRequest) -> OrderResult:
        """
        Envoie un ordre à la queue.
        
        Args:
            request: Requête d'ordre
            
        Returns:
            Résultat de l'ordre
        """
        # Vérification de la connexion
        if not self.check_connection():
            return OrderResult(
                request=request,
                order_id=None,
                status='error',
                fill_price=None,
                fill_time=None,
                error="Non connecté à MT5"
            )
            
        # Ajout à la queue
        self.order_queue.put(request)
        
        # Attente du résultat
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # Vérification des positions
            positions = mt5.positions_get(symbol=request.symbol)
            if positions:
                for pos in positions:
                    if pos.magic == request.magic:
                        return OrderResult(
                            request=request,
                            order_id=pos.ticket,
                            status='filled',
                            fill_price=pos.price_open,
                            fill_time=datetime.fromtimestamp(pos.time),
                            error=None
                        )
                        
            time.sleep(0.1)
            
        return OrderResult(
            request=request,
            order_id=None,
            status='expired',
            fill_price=None,
            fill_time=datetime.now(),
            error="Timeout de l'ordre"
        )
        
    def modify_position(
        self,
        position_id: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Modifie une position existante.
        
        Args:
            position_id: ID de la position
            stop_loss: Nouveau stop loss
            take_profit: Nouveau take profit
            
        Returns:
            True si la modification est réussie
        """
        if not self.check_connection():
            return False
            
        try:
            # Récupération de la position
            position = mt5.positions_get(ticket=position_id)
            if not position:
                logger.error(f"Position {position_id} non trouvée")
                return False
                
            position = position[0]
            
            # Préparation de la requête
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position_id,
                "sl": stop_loss if stop_loss is not None else position.sl,
                "tp": take_profit if take_profit is not None else position.tp
            }
            
            # Envoi de la requête
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Échec de la modification: {result.comment}")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"Erreur lors de la modification de la position: {e}")
            return False
            
    def close_position(
        self,
        position_id: int,
        volume: Optional[float] = None
    ) -> bool:
        """
        Ferme une position.
        
        Args:
            position_id: ID de la position
            volume: Volume à fermer (None pour tout fermer)
            
        Returns:
            True si la fermeture est réussie
        """
        if not self.check_connection():
            return False
            
        try:
            # Récupération de la position
            position = mt5.positions_get(ticket=position_id)
            if not position:
                logger.error(f"Position {position_id} non trouvée")
                return False
                
            position = position[0]
            
            # Volume à fermer
            close_volume = volume if volume is not None else position.volume
            
            # Préparation de la requête
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position_id,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 10,
                "magic": position.magic,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Envoi de la requête
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Échec de la fermeture: {result.comment}")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"Erreur lors de la fermeture de la position: {e}")
            return False
            
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les positions ouvertes.
        
        Args:
            symbol: Symbole à filtrer
            
        Returns:
            Liste des positions
        """
        if not self.check_connection():
            return []
            
        try:
            # Récupération des positions
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            
            if positions is None:
                return []
                
            # Conversion en dictionnaires
            return [{
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'magic': pos.magic,
                'comment': pos.comment,
                'time': datetime.fromtimestamp(pos.time)
            } for pos in positions]
            
        except Exception as e:
            logger.exception(f"Erreur lors de la récupération des positions: {e}")
            return []
            
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Récupère les ordres en attente.
        
        Args:
            symbol: Symbole à filtrer
            
        Returns:
            Liste des ordres
        """
        if not self.check_connection():
            return []
            
        try:
            # Récupération des ordres
            orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
            
            if orders is None:
                return []
                
            # Conversion en dictionnaires
            return [{
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': self._get_order_type_name(order.type),
                'volume': order.volume_initial,
                'price': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'magic': order.magic,
                'comment': order.comment,
                'time': datetime.fromtimestamp(order.time_setup),
                'expiration': datetime.fromtimestamp(order.time_expiration) if order.time_expiration > 0 else None
            } for order in orders]
            
        except Exception as e:
            logger.exception(f"Erreur lors de la récupération des ordres: {e}")
            return []
            
    def _get_order_type_name(self, order_type: int) -> str:
        """
        Convertit le type d'ordre en nom.
        
        Args:
            order_type: Type d'ordre MT5
            
        Returns:
            Nom du type d'ordre
        """
        types = {
            mt5.ORDER_TYPE_BUY: 'buy',
            mt5.ORDER_TYPE_SELL: 'sell',
            mt5.ORDER_TYPE_BUY_LIMIT: 'buy_limit',
            mt5.ORDER_TYPE_SELL_LIMIT: 'sell_limit',
            mt5.ORDER_TYPE_BUY_STOP: 'buy_stop',
            mt5.ORDER_TYPE_SELL_STOP: 'sell_stop'
        }
        return types.get(order_type, 'unknown')
        
    def get_account_info(self) -> Optional[Dict]:
        """
        Récupère les informations du compte.
        
        Returns:
            Informations du compte
        """
        if not self.check_connection():
            return None
            
        try:
            info = mt5.account_info()
            if info is None:
                return None
                
            return {
                'login': info.login,
                'server': info.server,
                'balance': info.balance,
                'equity': info.equity,
                'margin': info.margin,
                'free_margin': info.margin_free,
                'leverage': info.leverage,
                'currency': info.currency
            }
            
        except Exception as e:
            logger.exception(f"Erreur lors de la récupération des informations du compte: {e}")
            return None
            
    def shutdown(self) -> None:
        """Arrête le gestionnaire d'ordres."""
        try:
            # Fermeture de MT5
            mt5.shutdown()
            logger.info("MT5 arrêté")
            
        except Exception as e:
            logger.exception(f"Erreur lors de l'arrêt de MT5: {e}")
            
    def __del__(self):
        """Destructeur."""
        self.shutdown() 