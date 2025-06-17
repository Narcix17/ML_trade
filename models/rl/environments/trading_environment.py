#!/usr/bin/env python3
"""
Shared Trading Environment for RL training and testing.
Eliminates duplicate code between training and testing scripts.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Unified Trading Environment for RL training and testing.
    """
    
    def __init__(self, data: pd.DataFrame, ml_model, config: dict, max_steps: int = 1000, mode: str = 'training'):
        """
        Initialize the trading environment.
        
        Args:
            data: Market data DataFrame
            ml_model: Trained ML model for signal generation
            config: Configuration dictionary
            max_steps: Maximum steps per episode
            mode: 'training' or 'testing'
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.ml_model = ml_model
        self.config = config
        self.max_steps = max_steps
        self.mode = mode
        
        # Trading parameters
        self.initial_balance = config.get('rl_environment', {}).get('initial_balance', 100000.0)
        self.max_position_size = config.get('rl_environment', {}).get('max_position_size', 0.1)
        self.transaction_cost = config.get('rl_environment', {}).get('transaction_cost', 0.0001)
        
        # Current state
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.pnl = 0.0
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: features + position + balance + pnl
        n_features = len(data.columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features + 3,),  # features + position + balance + pnl
            dtype=np.float32
        )
        
        # History for analysis
        self.action_history = []
        self.reward_history = []
        self.balance_history = [self.balance]
        self.position_history = [self.position]
        self.pnl_history = [self.pnl]
        self.price_history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.pnl = 0.0
        self.action_history = []
        self.reward_history = []
        self.balance_history = [self.balance]
        self.position_history = [self.position]
        self.pnl_history = [self.pnl]
        self.price_history = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute an action and return the new state."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}
        
        # Previous position
        old_position = self.position
        
        # Execute action
        if action == 1:  # Buy
            if self.position <= 0:
                self.position = 1
        elif action == 2:  # Sell
            if self.position >= 0:
                self.position = -1
        # action == 0: Hold (no change)
        
        # Calculate PnL
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # PnL based on position
        if self.position == 1:  # Long
            price_change = (next_price - current_price) / current_price
        elif self.position == -1:  # Short
            price_change = (current_price - next_price) / current_price
        else:  # Neutral
            price_change = 0
        
        # Transaction costs
        transaction_cost = 0
        if self.position != old_position:
            transaction_cost = self.transaction_cost
        
        # Update PnL and balance
        self.pnl += price_change * self.balance * self.max_position_size - transaction_cost
        self.balance += self.pnl
        
        # Limit values to prevent overflow
        self.balance = np.clip(self.balance, -1e6, 1e6)
        self.pnl = np.clip(self.pnl, -1e6, 1e6)
        
        # Calculate reward
        reward = self._calculate_reward(action, price_change, transaction_cost)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1 or self.current_step >= self.max_steps
        truncated = False
        
        # Update history
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.balance_history.append(self.balance)
        self.position_history.append(self.position)
        self.pnl_history.append(self.pnl)
        self.price_history.append(current_price)
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _get_observation(self) -> np.ndarray:
        """Return the current observation."""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])
        
        # Market features
        features = self.data.iloc[self.current_step].values.astype(np.float32)
        
        # Trading state with limits to prevent overflow
        balance_ratio = np.clip(self.balance / self.initial_balance, -10, 10)
        pnl_ratio = np.clip(self.pnl / self.initial_balance, -10, 10)
        
        trading_state = np.array([
            self.position,
            balance_ratio,
            pnl_ratio
        ], dtype=np.float32)
        
        # Check and clean NaN values
        observation = np.concatenate([features, trading_state])
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _calculate_reward(self, action: int, price_change: float, transaction_cost: float) -> float:
        """Calculate reward based on action and PnL."""
        # Base reward based on PnL
        base_reward = price_change * 100  # Amplify for learning
        
        # Transaction cost penalty
        transaction_penalty = -transaction_cost * 1000
        
        # ML signal bonus
        ml_signal_bonus = 0
        if self.current_step < len(self.data):
            try:
                # ML model prediction
                features = self.data.iloc[self.current_step:self.current_step+1]
                ml_prediction = self.ml_model.predict(features)
                ml_action = ml_prediction[0] if len(ml_prediction) > 0 else 0
                
                # Bonus if RL action matches ML signal
                if action == ml_action:
                    ml_signal_bonus = 0.1
                elif (action == 1 and ml_action == 2) or (action == 2 and ml_action == 1):
                    # Penalty for opposite action
                    ml_signal_bonus = -0.1
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")
        
        # Inactivity penalty
        inactivity_penalty = 0
        if len(self.action_history) > 10:
            recent_actions = self.action_history[-10:]
            if all(a == 0 for a in recent_actions):
                inactivity_penalty = -0.05
        
        total_reward = base_reward + transaction_penalty + ml_signal_bonus + inactivity_penalty
        
        return total_reward
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get trading metrics for analysis."""
        if not self.balance_history:
            return {}
        
        # Calculate metrics
        total_return = (self.balance_history[-1] - self.balance_history[0]) / self.balance_history[0]
        
        # Annualized return (approximate)
        days = len(self.balance_history) / 288  # 288 bars per day for M5
        annualized_return = (1 + total_return) ** (365 / days) - 1
        
        # Volatility
        returns = np.diff(self.balance_history) / self.balance_history[:-1]
        volatility = np.std(returns) * np.sqrt(288 * 365)  # Annualized
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% per year
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(self.balance_history)
        drawdown = (self.balance_history - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Total PnL
        total_pnl = self.pnl_history[-1]
        
        # Action distribution
        if self.action_history:
            action_counts = np.bincount(self.action_history, minlength=3)
            action_distribution = {
                'hold': action_counts[0] / len(self.action_history),
                'buy': action_counts[1] / len(self.action_history),
                'sell': action_counts[2] / len(self.action_history)
            }
        else:
            action_distribution = {'hold': 0, 'buy': 0, 'sell': 0}
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_pnl': total_pnl,
            'final_balance': self.balance_history[-1],
            'action_distribution': action_distribution,
            'total_trades': len(self.action_history),
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate based on profitable trades."""
        if not self.pnl_history or len(self.pnl_history) < 2:
            return 0.0
        
        # Count profitable periods
        pnl_changes = np.diff(self.pnl_history)
        winning_periods = np.sum(pnl_changes > 0)
        total_periods = len(pnl_changes)
        
        return winning_periods / total_periods if total_periods > 0 else 0.0 