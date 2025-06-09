"""
Risk Manager - Gestion avancée des risques pour le trading servo.

Gère:
1. Position sizing dynamique
2. Stop loss et take profit adaptatifs
3. Gestion du drawdown
4. Contrôles de risque en temps réel
5. Alertes et arrêts automatiques
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time

class RiskLevel(Enum):
    """Niveaux de risque."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

@dataclass
class RiskMetrics:
    """Métriques de risque."""
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    max_consecutive_losses: int
    current_consecutive_losses: int
    volatility: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float

@dataclass
class PositionRisk:
    """Risque d'une position."""
    symbol: str
    side: str
    volume: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    time_in_trade: timedelta
    risk_reward_ratio: float
    margin_used: float
    margin_level: float

class RiskManager:
    """Gestionnaire de risque avancé."""
    
    def __init__(self, config: dict):
        """
        Initialise le gestionnaire de risque.
        
        Args:
            config: Configuration de risque
        """
        self.config = config
        self.risk_config = config['trading']['risk_management']
        self.execution_config = config['execution']['risk']
        
        # Métriques de risque
        self.risk_metrics = RiskMetrics(
            current_drawdown=0.0,
            max_drawdown=0.0,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            monthly_pnl=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            max_consecutive_losses=0,
            current_consecutive_losses=0,
            volatility=0.0,
            var_95=0.0,
            expected_shortfall=0.0
        )
        
        # Historique des trades
        self.trade_history: List[Dict] = []
        self.pnl_history: List[float] = []
        self.equity_history: List[float] = []
        
        # État du système
        self.is_risk_ok = True
        self.risk_level = RiskLevel.LOW
        self.last_update = datetime.now()
        
        # Thread de monitoring
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Démarrage du monitoring
        self._start_monitoring()
        
        logger.info("Risk Manager initialisé")
        
    def _start_monitoring(self) -> None:
        """Démarre le monitoring de risque."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._risk_monitor,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def _risk_monitor(self) -> None:
        """Moniteur de risque en continu."""
        while self.is_monitoring:
            try:
                # Mise à jour des métriques
                self._update_risk_metrics()
                
                # Vérification des seuils
                self._check_risk_thresholds()
                
                # Log des alertes
                self._log_risk_status()
                
                time.sleep(5)  # Vérification toutes les 5 secondes
                
            except Exception as e:
                logger.error(f"Erreur dans le moniteur de risque: {e}")
                
    def _update_risk_metrics(self) -> None:
        """Met à jour les métriques de risque."""
        if not self.trade_history:
            return
            
        # Calcul du drawdown
        self._calculate_drawdown()
        
        # Calcul des ratios
        self._calculate_ratios()
        
        # Calcul de la VaR
        self._calculate_var()
        
        # Mise à jour du niveau de risque
        self._update_risk_level()
        
        self.last_update = datetime.now()
        
    def _calculate_drawdown(self) -> None:
        """Calcule le drawdown actuel et maximum."""
        if not self.equity_history:
            return
            
        equity_series = pd.Series(self.equity_history)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        
        self.risk_metrics.current_drawdown = drawdown.iloc[-1]
        self.risk_metrics.max_drawdown = drawdown.min()
        
    def _calculate_ratios(self) -> None:
        """Calcule les ratios de performance."""
        if len(self.pnl_history) < 2:
            return
            
        returns = pd.Series(self.pnl_history).pct_change().dropna()
        
        if len(returns) > 0:
            # Ratio de Sharpe
            if returns.std() > 0:
                self.risk_metrics.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                
            # Ratio de Sortino (basé sur les pertes)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                self.risk_metrics.sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252)
                
            # Volatilité
            self.risk_metrics.volatility = returns.std() * np.sqrt(252)
            
    def _calculate_var(self) -> None:
        """Calcule la Value at Risk."""
        if len(self.pnl_history) < 10:
            return
            
        returns = pd.Series(self.pnl_history).pct_change().dropna()
        
        if len(returns) > 0:
            # VaR 95%
            self.risk_metrics.var_95 = np.percentile(returns, 5)
            
            # Expected Shortfall (CVaR)
            var_threshold = np.percentile(returns, 5)
            tail_returns = returns[returns <= var_threshold]
            if len(tail_returns) > 0:
                self.risk_metrics.expected_shortfall = tail_returns.mean()
                
    def _update_risk_level(self) -> None:
        """Met à jour le niveau de risque."""
        # Critères pour chaque niveau
        if (self.risk_metrics.current_drawdown > -0.02 and 
            self.risk_metrics.sharpe_ratio > 1.0):
            self.risk_level = RiskLevel.LOW
        elif (self.risk_metrics.current_drawdown > -0.05 and 
              self.risk_metrics.sharpe_ratio > 0.5):
            self.risk_level = RiskLevel.MEDIUM
        elif (self.risk_metrics.current_drawdown > -0.10 and 
              self.risk_metrics.sharpe_ratio > 0.0):
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.EXTREME
            
    def _check_risk_thresholds(self) -> None:
        """Vérifie les seuils de risque."""
        was_ok = self.is_risk_ok
        
        # Vérification du drawdown maximum
        max_drawdown = self.risk_config.get('max_drawdown', 0.05)
        if abs(self.risk_metrics.current_drawdown) > max_drawdown:
            self.is_risk_ok = False
            logger.warning(f"Drawdown maximum dépassé: {self.risk_metrics.current_drawdown:.2%}")
            
        # Vérification de la perte quotidienne
        max_daily_loss = self.risk_config.get('max_daily_loss', 0.02)
        if self.risk_metrics.daily_pnl < -max_daily_loss:
            self.is_risk_ok = False
            logger.warning(f"Perte quotidienne maximale dépassée: {self.risk_metrics.daily_pnl:.2%}")
            
        # Vérification de la perte hebdomadaire
        max_weekly_loss = self.risk_config.get('max_weekly_loss', 0.05)
        if self.risk_metrics.weekly_pnl < -max_weekly_loss:
            self.is_risk_ok = False
            logger.warning(f"Perte hebdomadaire maximale dépassée: {self.risk_metrics.weekly_pnl:.2%}")
            
        # Vérification de la perte mensuelle
        max_monthly_loss = self.risk_config.get('max_monthly_loss', 0.10)
        if self.risk_metrics.monthly_pnl < -max_monthly_loss:
            self.is_risk_ok = False
            logger.warning(f"Perte mensuelle maximale dépassée: {self.risk_metrics.monthly_pnl:.2%}")
            
        # Changement de statut
        if was_ok and not self.is_risk_ok:
            logger.error("🚨 ALERTE DE RISQUE: Le système de trading a été arrêté pour cause de risque excessif")
            
    def _log_risk_status(self) -> None:
        """Log le statut de risque."""
        if not self.is_risk_ok:
            logger.warning(
                f"RISQUE ÉLEVÉ: DD={self.risk_metrics.current_drawdown:.2%}, "
                f"Niveau={self.risk_level.name}, "
                f"Sharpe={self.risk_metrics.sharpe_ratio:.2f}"
            )
            
    def calculate_position_size(
        self,
        signal_confidence: float,
        regime: int,
        account_equity: float,
        symbol_config: dict
    ) -> float:
        """
        Calcule la taille de position optimale.
        
        Args:
            signal_confidence: Confiance du signal (0-1)
            regime: Régime de marché
            account_equity: Équité du compte
            symbol_config: Configuration du symbole
            
        Returns:
            Taille de position en lots
        """
        # Taille de base
        base_size_pct = self.risk_config.get('position_size', 0.02)
        
        # Ajustements
        confidence_multiplier = signal_confidence
        regime_multiplier = self._get_regime_multiplier(regime)
        risk_multiplier = self._get_risk_multiplier()
        
        # Calcul de la taille
        position_value = (
            account_equity * 
            base_size_pct * 
            confidence_multiplier * 
            regime_multiplier * 
            risk_multiplier
        )
        
        # Conversion en lots
        point = symbol_config.get('point', 0.0001)
        lot_size = position_value / (point * 100000)  # Pour forex
        
        # Limites
        min_lot = symbol_config.get('min_lot', 0.01)
        max_lot = symbol_config.get('max_lot', 100.0)
        
        return max(min_lot, min(max_lot, lot_size))
        
    def _get_regime_multiplier(self, regime: int) -> float:
        """Retourne le multiplicateur basé sur le régime."""
        multipliers = {
            0: 0.8,   # Régime conservateur
            1: 0.5,   # Régime très conservateur
            2: 1.2,   # Régime modérément agressif
            3: 1.0,   # Régime normal
            4: 1.5    # Régime agressif
        }
        return multipliers.get(regime, 1.0)
        
    def _get_risk_multiplier(self) -> float:
        """Retourne le multiplicateur basé sur le niveau de risque."""
        multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.5,
            RiskLevel.EXTREME: 0.2
        }
        return multipliers.get(self.risk_level, 0.5)
        
    def calculate_stops(
        self,
        entry_price: float,
        side: str,
        atr: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calcule les stops adaptatifs.
        
        Args:
            entry_price: Prix d'entrée
            side: 'long' ou 'short'
            atr: Average True Range
            volatility: Volatilité actuelle
            
        Returns:
            Tuple (stop_loss, take_profit)
        """
        stops_config = self.config['trading']['stops']
        
        # Multiplicateurs de base
        sl_multiplier = stops_config.get('stop_loss_atr', 2.0)
        tp_multiplier = stops_config.get('take_profit_atr', 3.0)
        
        # Ajustement basé sur la volatilité
        vol_adjustment = min(volatility / 0.01, 2.0)  # Normalisé
        
        # Ajustement basé sur le niveau de risque
        risk_adjustment = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 1.2,
            RiskLevel.HIGH: 1.5,
            RiskLevel.EXTREME: 2.0
        }.get(self.risk_level, 1.5)
        
        # Calculs finaux
        sl_distance = atr * sl_multiplier * vol_adjustment * risk_adjustment
        tp_distance = atr * tp_multiplier * vol_adjustment * risk_adjustment
        
        if side == 'long':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            
        return stop_loss, take_profit
        
    def can_open_position(self, symbol: str, required_margin: float) -> bool:
        """
        Vérifie si une nouvelle position peut être ouverte.
        
        Args:
            symbol: Symbole de trading
            required_margin: Marge requise
            
        Returns:
            True si la position peut être ouverte
        """
        # Vérification du statut de risque
        if not self.is_risk_ok:
            return False
            
        # Vérification du nombre maximum de positions
        max_positions = self.risk_config.get('max_positions', 3)
        # TODO: Récupérer le nombre de positions actuelles
        
        # Vérification de la marge disponible
        # TODO: Récupérer la marge disponible
        
        # Vérification du niveau de risque
        if self.risk_level == RiskLevel.EXTREME:
            return False
            
        return True
        
    def add_trade(self, trade_data: Dict) -> None:
        """
        Ajoute un trade à l'historique.
        
        Args:
            trade_data: Données du trade
        """
        self.trade_history.append(trade_data)
        
        # Mise à jour des métriques
        if 'pnl' in trade_data:
            self.pnl_history.append(trade_data['pnl'])
            
        if 'equity' in trade_data:
            self.equity_history.append(trade_data['equity'])
            
        # Calcul du win rate
        if len(self.trade_history) > 0:
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            self.risk_metrics.win_rate = winning_trades / len(self.trade_history)
            
        # Calcul du profit factor
        if len(self.trade_history) > 0:
            gross_profit = sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) > 0)
            gross_loss = abs(sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) < 0))
            
            if gross_loss > 0:
                self.risk_metrics.profit_factor = gross_profit / gross_loss
                
    def get_risk_metrics(self) -> RiskMetrics:
        """Retourne les métriques de risque actuelles."""
        return self.risk_metrics
        
    def get_risk_level(self) -> RiskLevel:
        """Retourne le niveau de risque actuel."""
        return self.risk_level
        
    def is_risk_ok(self) -> bool:
        """Retourne True si le risque est acceptable."""
        return self.is_risk_ok
        
    def force_stop_trading(self) -> None:
        """Force l'arrêt du trading pour cause de risque."""
        self.is_risk_ok = False
        logger.critical("🚨 ARRÊT FORCÉ DU TRADING: Risque excessif détecté")
        
    def reset_risk_status(self) -> None:
        """Réinitialise le statut de risque."""
        self.is_risk_ok = True
        logger.info("Statut de risque réinitialisé")
        
    def shutdown(self) -> None:
        """Arrête le gestionnaire de risque."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Risk Manager arrêté") 