#!/usr/bin/env python3
"""
Strategic model training with zone detection integration.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import des modules partagés
from utils.data_loading import get_mt5_connector, load_mt5_data
from utils.config import load_config
from utils.logging import setup_logger, get_logger

# Import our modules
from features.feature_engineering import FeatureEngineer
from features.zone_detection import ZoneDetector, ZoneType, MarketRegime
from models.ml_model import MLModel
from models.rl.rl_model import RLModel
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/strategic_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategicModelTrainer:
    """
    Training system that integrates zone detection with ML/RL models
    Generates strategic targets based on zone analysis
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config)
        self.zone_detector = ZoneDetector(
            window_size=self.config.get('zone_detection', {}).get('window_size', 5),
            min_amplitude_pct=self.config.get('zone_detection', {}).get('min_amplitude_pct', 0.25),
            bounce_lookback=self.config.get('zone_detection', {}).get('bounce_lookback', 50),
            validation_lookback=self.config.get('zone_detection', {}).get('validation_lookback', 20),
            confluence_threshold=self.config.get('zone_detection', {}).get('confluence_threshold', 2)
        )
        
        self.ml_model = MLModel(self.config)
        self.rl_model = RLModel()
        
        # Training parameters - read from config with fallbacks
        self.symbol = self.config.get('training', {}).get('symbol', 
                   self.config.get('trading', {}).get('symbol', 'EURUSD'))
        self.timeframe = self.config.get('training', {}).get('timeframes', ['M5'])[0]
        self.lookback_periods = self.config.get('training', {}).get('lookback_periods', 100)
        self.target_horizon = self.config.get('training', {}).get('target_horizon', 10)
        
        # Strategic targets storage
        self.strategic_targets = []
        
        logger.info(f"Initialized trainer with symbol: {self.symbol}, timeframe: {self.timeframe}")
        
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        
        # Login if credentials are provided
        if 'broker' in self.config:
            broker_config = self.config['broker']
            if broker_config.get('login') and broker_config.get('password'):
                if not mt5.login(
                    login=broker_config.get('login'),
                    password=broker_config.get('password'),
                    server=broker_config.get('server', 'MetaQuotes-Demo')
                ):
                    logger.error("MT5 login failed")
                    return False
        
        logger.info("Connected to MT5 successfully")
        return True
    
    def get_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data from MT5"""
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        mt5_timeframe = timeframe_map.get(self.timeframe, mt5.TIMEFRAME_M15)
        
        rates = mt5.copy_rates_range(self.symbol, mt5_timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logger.error(f"No data received for {self.symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Add volume if missing (MT5 sometimes doesn't provide volume for forex)
        if 'tick_volume' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['tick_volume']
        elif 'volume' not in df.columns:
            # Create synthetic volume based on price movement
            df['volume'] = abs(df['close'] - df['open']) * 1000000  # Synthetic volume
        
        logger.info(f"Retrieved {len(df)} bars of historical data")
        logger.info(f"Columns available: {list(df.columns)}")
        return df
    
    def generate_strategic_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate strategic targets based on zone analysis
        """
        logger.info("Generating strategic targets based on zone analysis...")
        
        # Update zones with current data
        self.zone_detector.update_zones(df)
        
        # Get zone summary
        zone_summary = self.zone_detector.get_zone_summary()
        logger.info(f"Zone Summary: {zone_summary}")
        
        # Initialize target columns
        df['target_zone_proximity'] = 0.0
        df['target_zone_strength'] = 0.0
        df['target_zone_score'] = 0.0
        df['target_entry_signal'] = 0.0
        df['target_exit_signal'] = 0.0
        df['target_direction'] = 0  # -1: short, 0: neutral, 1: long
        df['target_confidence'] = 0.0
        df['market_regime'] = 0.0
        
        # Process each data point
        for i in range(len(df)):
            current_df = df.iloc[:i+1]
            if len(current_df) < 50:  # Need minimum data for analysis
                continue
            
            # Update zones for this point
            self.zone_detector.update_zones(current_df)
            
            # Get current price and market regime
            current_price = current_df.iloc[-1]['close']
            regime = self.zone_detector.detect_market_regime(current_df)
            
            # Find nearest zones
            nearest_zones = self.zone_detector._find_nearest_zones(current_price, max_distance_pct=1.0)
            
            if nearest_zones:
                best_zone = nearest_zones[0]  # Closest zone
                
                # Calculate proximity (normalized)
                distance_pct = abs(current_price - best_zone.price) / current_price
                proximity_score = max(0, 1 - distance_pct * 100)  # Higher when closer
                
                # Entry signal based on zone type and proximity
                entry_signal = 0.0
                direction = 0
                
                if best_zone.zone_type == ZoneType.SWING_HIGH:
                    # Resistance zone - potential short entry
                    if proximity_score > 0.8 and best_zone.strength > 0.6:
                        entry_signal = proximity_score * best_zone.strength
                        direction = -1  # Short
                
                elif best_zone.zone_type == ZoneType.SWING_LOW:
                    # Support zone - potential long entry
                    if proximity_score > 0.8 and best_zone.strength > 0.6:
                        entry_signal = proximity_score * best_zone.strength
                        direction = 1  # Long
                
                elif best_zone.zone_type == ZoneType.PSYCHOLOGICAL:
                    # Psychological level - can work both ways
                    if proximity_score > 0.9:
                        # Determine direction based on recent price action
                        recent_change = (current_price - current_df.iloc[-5]['close']) / current_df.iloc[-5]['close']
                        if abs(recent_change) > 0.001:  # 0.1% movement
                            direction = 1 if recent_change > 0 else -1
                            entry_signal = proximity_score * best_zone.strength
                
                # Exit signal (opposite of entry)
                exit_signal = 0.0
                if entry_signal > 0:
                    # Exit when moving away from zone
                    exit_signal = 1 - proximity_score
                
                # Confidence based on zone score and market regime
                confidence = best_zone.score
                if regime == MarketRegime.RANGE:
                    # Higher confidence for range-bound markets
                    confidence *= 1.2
                elif regime == MarketRegime.TREND:
                    # Lower confidence for trending markets (zones may break)
                    confidence *= 0.8
                
                # Store targets
                df.iloc[i, df.columns.get_loc('target_zone_proximity')] = proximity_score
                df.iloc[i, df.columns.get_loc('target_zone_strength')] = best_zone.strength
                df.iloc[i, df.columns.get_loc('target_zone_score')] = best_zone.score
                df.iloc[i, df.columns.get_loc('target_entry_signal')] = entry_signal
                df.iloc[i, df.columns.get_loc('target_exit_signal')] = exit_signal
                df.iloc[i, df.columns.get_loc('target_direction')] = direction
                df.iloc[i, df.columns.get_loc('target_confidence')] = confidence
                df.iloc[i, df.columns.get_loc('market_regime')] = 1.0 if regime == MarketRegime.TREND else 0.0
        
        logger.info(f"Generated strategic targets for {len(df)} data points")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features and targets for training
        """
        logger.info("Preparing training data...")
        
        # Generate features
        features_df = self.feature_engineer.generate_features(df)
        
        # Generate strategic targets
        targets_df = self.generate_strategic_targets(df)
        
        # Combine features and targets
        training_df = pd.concat([features_df, targets_df[['target_zone_proximity', 'target_zone_strength', 
                                                          'target_zone_score', 'target_entry_signal', 
                                                          'target_exit_signal', 'target_direction', 
                                                          'target_confidence', 'market_regime']]], axis=1)
        
        # Remove rows with NaN values
        training_df = training_df.dropna()
        
        logger.info(f"Prepared training data: {len(training_df)} samples")
        return training_df
    
    def train_ml_model(self, training_df: pd.DataFrame) -> None:
        """
        Train ML model with strategic targets
        """
        logger.info("Training ML model with strategic targets...")
        
        # Prepare features (exclude target columns)
        target_columns = [col for col in training_df.columns if col.startswith('target_')]
        feature_columns = [col for col in training_df.columns if not col.startswith('target_')]
        
        X = training_df[feature_columns]
        y = training_df['target_direction']  # Main target
        
        # Train model
        self.ml_model.train(X, y)
        
        # Save model
        model_path = f"models/saved/strategic_ml_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(self.ml_model, model_path)
        logger.info(f"ML model saved to {model_path}")
        
        # Log feature importance
        if hasattr(self.ml_model.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.ml_model.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 feature importances:")
            logger.info(feature_importance.head(10))
    
    def train_rl_model(self, training_df: pd.DataFrame) -> None:
        """
        Train RL model with strategic targets
        """
        logger.info("Training RL model with strategic targets...")
        
        # Prepare features
        target_columns = [col for col in training_df.columns if col.startswith('target_')]
        feature_columns = [col for col in training_df.columns if not col.startswith('target_')]
        
        X = training_df[feature_columns]
        
        # Create custom reward function based on strategic targets
        def strategic_reward_function(state, action, next_state, reward):
            # Get strategic signals from state
            zone_proximity = state.get('target_zone_proximity', 0)
            zone_strength = state.get('target_zone_strength', 0)
            entry_signal = state.get('target_entry_signal', 0)
            confidence = state.get('target_confidence', 0)
            
            # Adjust reward based on strategic alignment
            strategic_bonus = 0
            
            if action == 1:  # Long action
                if entry_signal > 0.5 and confidence > 0.6:
                    strategic_bonus = entry_signal * confidence * 2
                elif entry_signal < 0.2:
                    strategic_bonus = -0.5  # Penalty for wrong timing
            
            elif action == -1:  # Short action
                if entry_signal > 0.5 and confidence > 0.6:
                    strategic_bonus = entry_signal * confidence * 2
                elif entry_signal < 0.2:
                    strategic_bonus = -0.5  # Penalty for wrong timing
            
            elif action == 0:  # Hold action
                if entry_signal < 0.3:  # Good to hold when no clear signal
                    strategic_bonus = 0.2
                else:
                    strategic_bonus = -0.3  # Penalty for missing opportunity
            
            return reward + strategic_bonus
        
        # Train RL model with strategic reward function
        self.rl_model.train(X, custom_reward_function=strategic_reward_function)
        
        # Save model
        model_path = f"models/saved/strategic_rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(self.rl_model, model_path)
        logger.info(f"RL model saved to {model_path}")
    
    def evaluate_models(self, training_df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance with strategic metrics
        """
        logger.info("Evaluating models...")
        
        # Prepare test data (last 20% of data)
        test_size = int(len(training_df) * 0.2)
        test_df = training_df.tail(test_size)
        
        target_columns = [col for col in training_df.columns if col.startswith('target_')]
        feature_columns = [col for col in training_df.columns if not col.startswith('target_')]
        
        X_test = test_df[feature_columns]
        y_test = test_df['target_direction']
        
        # ML model evaluation
        ml_predictions = self.ml_model.predict(X_test)
        ml_accuracy = np.mean(ml_predictions == y_test)
        
        # RL model evaluation
        rl_predictions = self.rl_model.predict(X_test)
        rl_accuracy = np.mean(rl_predictions == y_test)
        
        # Strategic alignment metrics
        strategic_alignment = {
            'ml_accuracy': ml_accuracy,
            'rl_accuracy': rl_accuracy,
            'avg_zone_strength': test_df['target_zone_strength'].mean(),
            'avg_confidence': test_df['target_confidence'].mean(),
            'entry_signals': (test_df['target_entry_signal'] > 0.5).sum(),
            'high_confidence_signals': (test_df['target_confidence'] > 0.7).sum()
        }
        
        logger.info("Model Evaluation Results:")
        for metric, value in strategic_alignment.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return strategic_alignment
    
    def run_training(self, days_back: int = 30) -> None:
        """
        Run complete training pipeline
        """
        logger.info("Starting strategic model training...")
        
        # Connect to MT5
        if not self.connect_mt5():
            return
        
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Fetching data from {start_date} to {end_date}")
            df = self.get_historical_data(start_date, end_date)
            
            if df.empty:
                logger.error("No data retrieved")
                return
            
            # Prepare training data
            training_df = self.prepare_training_data(df)
            
            if len(training_df) < 100:
                logger.error("Insufficient training data")
                return
            
            # Train models
            self.train_ml_model(training_df)
            self.train_rl_model(training_df)
            
            # Evaluate models
            evaluation_results = self.evaluate_models(training_df)
            
            # Save training summary
            summary = {
                'training_date': datetime.now().isoformat(),
                'data_period': f"{start_date} to {end_date}",
                'total_samples': len(training_df),
                'evaluation_results': evaluation_results,
                'zone_summary': self.zone_detector.get_zone_summary()
            }
            
            # Save summary to file
            import json
            with open(f'logs/training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Strategic model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
        
        finally:
            mt5.shutdown()

def main():
    """Main training function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train strategic ML/RL models')
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., GBPUSD, EURUSD)')
    parser.add_argument('--days-back', type=int, default=30, help='Number of days of historical data to use')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., M5, M15, H1)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = StrategicModelTrainer(config_path=args.config)
    
    # Override config values with command line arguments if provided
    if args.symbol:
        trainer.symbol = args.symbol
        logger.info(f"Using symbol from command line: {trainer.symbol}")
    
    if args.timeframe:
        trainer.timeframe = args.timeframe
        logger.info(f"Using timeframe from command line: {trainer.timeframe}")
    
    # Run training
    trainer.run_training(days_back=args.days_back)

if __name__ == "__main__":
    main() 