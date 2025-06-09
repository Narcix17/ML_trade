"""
Moteur de backtesting pour tester les stratégies de trading.

Gère l'exécution des backtests avec prise en compte des coûts,
de la latence et de la synchronisation multi-timeframes.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import json

@dataclass
class Trade:
    """Classe représentant une transaction."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position: float  # >0 pour long, <0 pour short
    pnl: float
    fees: float
    slippage: float
    regime: Optional[int]
    strategy: str
    timeframe: str
    metadata: Dict

@dataclass
class BacktestResult:
    """Résultats d'un backtest."""
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict
    regime_metrics: Optional[Dict]
    strategy_metrics: Optional[Dict]
    timeframe_metrics: Optional[Dict]
    session_metrics: Optional[Dict]

class BacktestEngine:
    """Moteur de backtesting."""
    
    def __init__(
        self,
        config: dict,
        data: Dict[str, pd.DataFrame],
        strategies: Dict[str, callable],
        regime_detector: Optional[callable] = None
    ):
        """
        Initialise le moteur de backtest.
        
        Args:
            config: Configuration du backtest
            data: Dictionnaire de DataFrames par timeframe
            strategies: Dictionnaire de fonctions de stratégie
            regime_detector: Détecteur de régime de marché
        """
        self.config = config['backtesting']
        self.data = data
        self.strategies = strategies
        self.regime_detector = regime_detector
        
        # Paramètres de trading
        self.initial_balance = self.config['general']['initial_capital']
        self.commission = self.config['general']['commission']
        self.slippage = self.config['general']['slippage']
        
        # Paramètres d'exécution
        self.execution_latency = self.config['execution']['latency']
        self.partial_fills = self.config['execution']['partial_fills']
        self.market_impact = self.config['execution']['market_impact']
        self.impact_factor = self.config['execution']['impact_factor']
        
        # Paramètres de risque
        self.position_sizing = self.config['risk']['position_sizing']
        self.max_positions = self.config['risk']['max_positions']
        self.max_drawdown = self.config['risk']['max_drawdown']
        
        # Timeframes
        self.timeframes = list(data.keys())
        self.primary_tf = next(
            (tf for tf in config['broker']['timeframes'] if tf['primary']),
            self.timeframes[0]
        )['name']
        
        # Sessions de trading
        self.sessions = self._define_sessions()
        
        # État du backtest
        self.trades: List[Trade] = []
        self.positions: Dict[str, float] = {}
        self.balance = self.initial_balance
        self.equity_curve = []
        
    def _define_sessions(self) -> Dict:
        """
        Définit les sessions de trading.
        
        Returns:
            Dictionnaire des sessions
        """
        return {
            'EU': {
                'start': '08:00',
                'end': '16:30',
                'timezone': 'Europe/Berlin'
            },
            'US': {
                'start': '14:30',
                'end': '21:00',
                'timezone': 'America/New_York'
            },
            'overlap': {
                'start': '14:30',
                'end': '16:30',
                'timezone': 'Europe/Berlin'
            }
        }
        
    def _is_trading_session(self, timestamp: datetime) -> bool:
        """
        Vérifie si le timestamp est dans une session de trading.
        
        Args:
            timestamp: Timestamp à vérifier
            
        Returns:
            True si dans une session
        """
        for session, times in self.sessions.items():
            tz = pytz.timezone(times['timezone'])
            local_time = timestamp.astimezone(tz).time()
            start = datetime.strptime(times['start'], '%H:%M').time()
            end = datetime.strptime(times['end'], '%H:%M').time()
            
            if start <= local_time <= end:
                return True
        return False
        
    def _get_current_regime(self, timestamp: datetime) -> Optional[int]:
        """
        Obtient le régime de marché actuel.
        
        Args:
            timestamp: Timestamp actuel
            
        Returns:
            ID du régime ou None
        """
        if self.regime_detector is None:
            return None
            
        # Utilise le timeframe principal pour la détection
        df = self.data[self.primary_tf]
        current_data = df[df.index <= timestamp].iloc[-1:]
        
        if len(current_data) > 0:
            return self.regime_detector.predict_regime(current_data)
        return None
        
    def _calculate_slippage(self, price: float, position: float) -> float:
        """
        Calcule le slippage pour une transaction.
        
        Args:
            price: Prix de base
            position: Taille de la position
            
        Returns:
            Slippage en points
        """
        # Slippage proportionnel à la taille de la position
        base_slippage = self.slippage * abs(position)
        
        # Ajout de bruit aléatoire
        noise = np.random.normal(0, base_slippage * 0.1)
        
        return base_slippage + noise
        
    def _execute_trade(
        self,
        timestamp: datetime,
        strategy: str,
        timeframe: str,
        position: float,
        price: float,
        metadata: Dict
    ) -> Optional[Trade]:
        """
        Exécute une transaction.
        
        Args:
            timestamp: Timestamp de la transaction
            strategy: Nom de la stratégie
            timeframe: Timeframe
            position: Taille de la position
            price: Prix d'exécution
            metadata: Métadonnées
            
        Returns:
            Trade ou None si rejeté
        """
        # Vérification des sessions
        if not self._is_trading_session(timestamp):
            logger.debug(f"Transaction rejetée hors session: {timestamp}")
            return None
            
        # Calcul des frais et slippage
        fees = abs(position) * self.commission
        slippage = self._calculate_slippage(price, position)
        
        # Ajustement du prix
        execution_price = price + (slippage if position > 0 else -slippage)
        
        # Création de la transaction
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=execution_price,
            exit_price=None,
            position=position,
            pnl=0.0,
            fees=fees,
            slippage=slippage,
            regime=self._get_current_regime(timestamp),
            strategy=strategy,
            timeframe=timeframe,
            metadata=metadata
        )
        
        # Mise à jour des positions
        self.positions[strategy] = position
        
        # Mise à jour du capital
        self.balance -= fees
        
        return trade
        
    def _close_trade(
        self,
        trade: Trade,
        timestamp: datetime,
        price: float
    ) -> None:
        """
        Ferme une transaction.
        
        Args:
            trade: Transaction à fermer
            timestamp: Timestamp de fermeture
            price: Prix de fermeture
        """
        # Calcul du PnL
        price_change = price - trade.entry_price
        trade.pnl = trade.position * price_change - trade.fees
        
        # Mise à jour de la transaction
        trade.exit_time = timestamp
        trade.exit_price = price
        
        # Mise à jour du capital
        self.balance += trade.pnl
        
        # Mise à jour des positions
        self.positions[trade.strategy] = 0.0
        
    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Exécute le backtest.
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Résultats du backtest
        """
        # Initialisation
        self.trades = []
        self.positions = {s: 0.0 for s in self.strategies}
        self.balance = self.initial_balance
        self.equity_curve = []
        
        # Synchronisation des données
        df = self._synchronize_data(start_date, end_date)
        
        # Boucle principale
        for timestamp, row in df.iterrows():
            # Mise à jour de la courbe d'équité
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.balance
            })
            
            # Exécution des stratégies
            for strategy_name, strategy_func in self.strategies.items():
                # Vérification des positions maximales
                if len(self.trades) >= self.max_positions:
                    continue
                    
                # Appel de la stratégie
                signal = strategy_func(row, self.data, timestamp)
                
                if signal is not None:
                    position, metadata = signal
                    
                    # Exécution de la transaction
                    trade = self._execute_trade(
                        timestamp=timestamp,
                        strategy=strategy_name,
                        timeframe=self.primary_tf,
                        position=position,
                        price=row['close'],
                        metadata=metadata
                    )
                    
                    if trade is not None:
                        self.trades.append(trade)
                        
            # Gestion des positions existantes
            self._manage_positions(timestamp, row)
            
        # Calcul des résultats
        return self._calculate_results()
        
    def _synchronize_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Synchronise les données multi-timeframes.
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame synchronisé
        """
        # Utilise le timeframe principal comme base
        df = self.data[self.primary_tf].copy()
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Ajout des données des autres timeframes
        for tf in self.timeframes:
            if tf != self.primary_tf:
                tf_data = self.data[tf]
                # Resample sur le timeframe principal
                tf_data = tf_data.resample(self.primary_tf).ffill()
                # Merge avec le DataFrame principal
                df = df.join(tf_data, rsuffix=f'_{tf}')
                
        return df
        
    def _manage_positions(
        self,
        timestamp: datetime,
        data: pd.Series
    ) -> None:
        """
        Gère les positions existantes.
        
        Args:
            timestamp: Timestamp actuel
            data: Données de marché
        """
        for trade in self.trades:
            if trade.exit_time is None:
                # Vérification des conditions de sortie
                if self._should_close_trade(trade, timestamp, data):
                    self._close_trade(trade, timestamp, data['close'])
                    
    def _should_close_trade(
        self,
        trade: Trade,
        timestamp: datetime,
        data: pd.Series
    ) -> bool:
        """
        Détermine si une position doit être fermée.
        
        Args:
            trade: Transaction à vérifier
            timestamp: Timestamp actuel
            data: Données de marché
            
        Returns:
            True si la position doit être fermée
        """
        # Vérification du stop loss
        if trade.position > 0:
            if data['low'] <= trade.entry_price * (1 - self.config['backtest']['stop_loss']):
                return True
        else:
            if data['high'] >= trade.entry_price * (1 + self.config['backtest']['stop_loss']):
                return True
                
        # Vérification du take profit
        if trade.position > 0:
            if data['high'] >= trade.entry_price * (1 + self.config['backtest']['take_profit']):
                return True
        else:
            if data['low'] <= trade.entry_price * (1 - self.config['backtest']['take_profit']):
                return True
                
        # Vérification du time stop
        max_duration = timedelta(hours=self.config['backtest']['max_trade_duration'])
        if timestamp - trade.entry_time >= max_duration:
            return True
            
        return False
        
    def _calculate_results(self) -> BacktestResult:
        """
        Calcule les résultats du backtest.
        
        Returns:
            Résultats du backtest
        """
        # Conversion de la courbe d'équité
        equity_df = pd.DataFrame(self.equity_curve)
        equity_curve = equity_df.set_index('timestamp')['equity']
        
        # Calcul des métriques globales
        returns = equity_curve.pct_change().dropna()
        metrics = {
            'total_return': (equity_curve.iloc[-1] / self.initial_balance - 1) * 100,
            'sharpe_ratio': self._calculate_sharpe(returns),
            'sortino_ratio': self._calculate_sortino(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'calmar_ratio': self._calculate_calmar(returns, equity_curve),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'expectancy': self._calculate_expectancy(),
            'avg_trade': np.mean([t.pnl for t in self.trades]),
            'avg_win': np.mean([t.pnl for t in self.trades if t.pnl > 0]),
            'avg_loss': np.mean([t.pnl for t in self.trades if t.pnl < 0]),
            'max_consecutive_wins': self._calculate_max_consecutive(self.trades, True),
            'max_consecutive_losses': self._calculate_max_consecutive(self.trades, False)
        }
        
        # Métriques par régime
        regime_metrics = self._calculate_regime_metrics()
        
        # Métriques par stratégie
        strategy_metrics = self._calculate_strategy_metrics()
        
        # Métriques par timeframe
        timeframe_metrics = self._calculate_timeframe_metrics()
        
        # Métriques par session
        session_metrics = self._calculate_session_metrics()
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            metrics=metrics,
            regime_metrics=regime_metrics,
            strategy_metrics=strategy_metrics,
            timeframe_metrics=timeframe_metrics,
            session_metrics=session_metrics
        )
        
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calcule le ratio de Sharpe."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
        
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calcule le ratio de Sortino."""
        if len(returns) < 2:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        return np.sqrt(252) * returns.mean() / downside_returns.std()
        
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calcule le drawdown maximum."""
        rolling_max = equity.expanding().max()
        drawdowns = equity / rolling_max - 1
        return abs(drawdowns.min()) * 100
        
    def _calculate_calmar(self, returns: pd.Series, equity: pd.Series) -> float:
        """Calcule le ratio de Calmar."""
        max_dd = self._calculate_max_drawdown(equity)
        if max_dd == 0:
            return np.inf
        return (returns.mean() * 252) / (max_dd / 100)
        
    def _calculate_win_rate(self) -> float:
        """Calcule le taux de réussite."""
        if not self.trades:
            return 0.0
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        return winning_trades / len(self.trades) * 100
        
    def _calculate_profit_factor(self) -> float:
        """Calcule le facteur de profit."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else np.inf
        
    def _calculate_expectancy(self) -> float:
        """Calcule l'espérance de gain."""
        if not self.trades:
            return 0.0
        return np.mean([t.pnl for t in self.trades])
        
    def _calculate_max_consecutive(
        self,
        trades: List[Trade],
        winning: bool
    ) -> int:
        """Calcule le nombre maximum de trades consécutifs gagnants/perdants."""
        current_streak = 0
        max_streak = 0
        
        for trade in trades:
            if (winning and trade.pnl > 0) or (not winning and trade.pnl < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
        
    def _calculate_regime_metrics(self) -> Dict:
        """Calcule les métriques par régime de marché."""
        if not self.regime_detector:
            return {}
            
        regime_metrics = {}
        for trade in self.trades:
            if trade.regime is not None:
                if trade.regime not in regime_metrics:
                    regime_metrics[trade.regime] = {
                        'trades': [],
                        'pnl': 0.0,
                        'wins': 0,
                        'losses': 0
                    }
                    
                regime_metrics[trade.regime]['trades'].append(trade)
                regime_metrics[trade.regime]['pnl'] += trade.pnl
                
                if trade.pnl > 0:
                    regime_metrics[trade.regime]['wins'] += 1
                else:
                    regime_metrics[trade.regime]['losses'] += 1
                    
        # Calcul des statistiques par régime
        for regime in regime_metrics:
            trades = regime_metrics[regime]['trades']
            regime_metrics[regime].update({
                'win_rate': self._calculate_win_rate(trades),
                'avg_trade': np.mean([t.pnl for t in trades]),
                'sharpe': self._calculate_sharpe(
                    pd.Series([t.pnl for t in trades])
                )
            })
            
        return regime_metrics
        
    def _calculate_strategy_metrics(self) -> Dict:
        """Calcule les métriques par stratégie."""
        strategy_metrics = {}
        
        for trade in self.trades:
            if trade.strategy not in strategy_metrics:
                strategy_metrics[trade.strategy] = {
                    'trades': [],
                    'pnl': 0.0,
                    'wins': 0,
                    'losses': 0
                }
                
            strategy_metrics[trade.strategy]['trades'].append(trade)
            strategy_metrics[trade.strategy]['pnl'] += trade.pnl
            
            if trade.pnl > 0:
                strategy_metrics[trade.strategy]['wins'] += 1
            else:
                strategy_metrics[trade.strategy]['losses'] += 1
                
        # Calcul des statistiques par stratégie
        for strategy in strategy_metrics:
            trades = strategy_metrics[strategy]['trades']
            strategy_metrics[strategy].update({
                'win_rate': self._calculate_win_rate(trades),
                'avg_trade': np.mean([t.pnl for t in trades]),
                'sharpe': self._calculate_sharpe(
                    pd.Series([t.pnl for t in trades])
                )
            })
            
        return strategy_metrics
        
    def _calculate_timeframe_metrics(self) -> Dict:
        """Calcule les métriques par timeframe."""
        timeframe_metrics = {}
        
        for trade in self.trades:
            if trade.timeframe not in timeframe_metrics:
                timeframe_metrics[trade.timeframe] = {
                    'trades': [],
                    'pnl': 0.0,
                    'wins': 0,
                    'losses': 0
                }
                
            timeframe_metrics[trade.timeframe]['trades'].append(trade)
            timeframe_metrics[trade.timeframe]['pnl'] += trade.pnl
            
            if trade.pnl > 0:
                timeframe_metrics[trade.timeframe]['wins'] += 1
            else:
                timeframe_metrics[trade.timeframe]['losses'] += 1
                
        # Calcul des statistiques par timeframe
        for timeframe in timeframe_metrics:
            trades = timeframe_metrics[timeframe]['trades']
            timeframe_metrics[timeframe].update({
                'win_rate': self._calculate_win_rate(trades),
                'avg_trade': np.mean([t.pnl for t in trades]),
                'sharpe': self._calculate_sharpe(
                    pd.Series([t.pnl for t in trades])
                )
            })
            
        return timeframe_metrics
        
    def _calculate_session_metrics(self) -> Dict:
        """Calcule les métriques par session de trading."""
        session_metrics = {}
        
        for trade in self.trades:
            session = self._get_trade_session(trade)
            if session not in session_metrics:
                session_metrics[session] = {
                    'trades': [],
                    'pnl': 0.0,
                    'wins': 0,
                    'losses': 0
                }
                
            session_metrics[session]['trades'].append(trade)
            session_metrics[session]['pnl'] += trade.pnl
            
            if trade.pnl > 0:
                session_metrics[session]['wins'] += 1
            else:
                session_metrics[session]['losses'] += 1
                
        # Calcul des statistiques par session
        for session in session_metrics:
            trades = session_metrics[session]['trades']
            session_metrics[session].update({
                'win_rate': self._calculate_win_rate(trades),
                'avg_trade': np.mean([t.pnl for t in trades]),
                'sharpe': self._calculate_sharpe(
                    pd.Series([t.pnl for t in trades])
                )
            })
            
        return session_metrics
        
    def _get_trade_session(self, trade: Trade) -> str:
        """
        Détermine la session d'une transaction.
        
        Args:
            trade: Transaction
            
        Returns:
            Nom de la session
        """
        for session, times in self.sessions.items():
            tz = pytz.timezone(times['timezone'])
            local_time = trade.entry_time.astimezone(tz).time()
            start = datetime.strptime(times['start'], '%H:%M').time()
            end = datetime.strptime(times['end'], '%H:%M').time()
            
            if start <= local_time <= end:
                return session
                
        return 'unknown'
        
    def plot_results(self, result: BacktestResult) -> None:
        """
        Visualise les résultats du backtest.
        
        Args:
            result: Résultats du backtest
        """
        # Configuration du style
        plt.style.use('seaborn')
        
        # Création de la figure
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Courbe d'équité
        ax1 = fig.add_subplot(gs[0, :])
        result.equity_curve.plot(ax=ax1, label='Equity')
        ax1.set_title('Courbe d\'équité')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital')
        ax1.grid(True)
        
        # Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        drawdown = (result.equity_curve / result.equity_curve.expanding().max() - 1) * 100
        drawdown.plot(ax=ax2, color='red', label='Drawdown')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # Distribution des retours
        ax3 = fig.add_subplot(gs[1, 1])
        returns = result.equity_curve.pct_change().dropna()
        sns.histplot(returns, ax=ax3, kde=True)
        ax3.set_title('Distribution des retours')
        ax3.set_xlabel('Retour')
        ax3.set_ylabel('Fréquence')
        
        # Métriques par stratégie
        ax4 = fig.add_subplot(gs[2, 0])
        strategy_pnl = pd.DataFrame({
            s: [t.pnl for t in result.trades if t.strategy == s]
            for s in result.strategy_metrics.keys()
        })
        strategy_pnl.boxplot(ax=ax4)
        ax4.set_title('PnL par stratégie')
        ax4.set_xlabel('Stratégie')
        ax4.set_ylabel('PnL')
        plt.xticks(rotation=45)
        
        # Métriques par régime
        if result.regime_metrics:
            ax5 = fig.add_subplot(gs[2, 1])
            regime_pnl = pd.DataFrame({
                r: [t.pnl for t in result.trades if t.regime == r]
                for r in result.regime_metrics.keys()
            })
            regime_pnl.boxplot(ax=ax5)
            ax5.set_title('PnL par régime')
            ax5.set_xlabel('Régime')
            ax5.set_ylabel('PnL')
            
        plt.tight_layout()
        plt.show()
        
    def save_results(self, result: BacktestResult, path: str) -> None:
        """
        Sauvegarde les résultats du backtest.
        
        Args:
            result: Résultats du backtest
            path: Chemin de sauvegarde
        """
        # Conversion des résultats en format JSON
        output = {
            'metrics': result.metrics,
            'regime_metrics': result.regime_metrics,
            'strategy_metrics': result.strategy_metrics,
            'timeframe_metrics': result.timeframe_metrics,
            'session_metrics': result.session_metrics,
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position': t.position,
                    'pnl': t.pnl,
                    'fees': t.fees,
                    'slippage': t.slippage,
                    'regime': t.regime,
                    'strategy': t.strategy,
                    'timeframe': t.timeframe,
                    'metadata': t.metadata
                }
                for t in result.trades
            ],
            'equity_curve': result.equity_curve.to_dict()
        }
        
        # Sauvegarde
        with open(path, 'w') as f:
            json.dump(output, f, indent=4)
            
        logger.info(f"Résultats sauvegardés: {path}")
        
    def load_results(self, path: str) -> BacktestResult:
        """
        Charge les résultats d'un backtest.
        
        Args:
            path: Chemin des résultats
            
        Returns:
            Résultats du backtest
        """
        # Chargement
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Conversion des trades
        trades = []
        for t in data['trades']:
            trade = Trade(
                entry_time=datetime.fromisoformat(t['entry_time']),
                exit_time=datetime.fromisoformat(t['exit_time']) if t['exit_time'] else None,
                entry_price=t['entry_price'],
                exit_price=t['exit_price'],
                position=t['position'],
                pnl=t['pnl'],
                fees=t['fees'],
                slippage=t['slippage'],
                regime=t['regime'],
                strategy=t['strategy'],
                timeframe=t['timeframe'],
                metadata=t['metadata']
            )
            trades.append(trade)
            
        # Conversion de la courbe d'équité
        equity_curve = pd.Series(data['equity_curve'])
        
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=data['metrics'],
            regime_metrics=data['regime_metrics'],
            strategy_metrics=data['strategy_metrics'],
            timeframe_metrics=data['timeframe_metrics'],
            session_metrics=data['session_metrics']
        ) 