"""
Unit tests for live trading system functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import yaml
from unittest.mock import Mock, patch, MagicMock, call

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trading.live.live_trading import LiveTradingSystem, RiskManager, EntryPointDetector


class TestLiveTradingSystem(unittest.TestCase):
    """Test cases for LiveTradingSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.config_data = {
            'trading': {
                'symbol': 'EURUSD',
                'timeframe': 'M5',
                'lot_size': 0.01,
                'max_positions': 3
            },
            'training': {
                'symbol': 'EURUSD',
                'timeframes': ['M5'],
                'start_date': '2024-01-01',
                'end_date': '2024-12-31'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(self.config_data, tmp_file)
            self.config_path = tmp_file.name
        
        # Mock MT5
        self.mt5_patcher = patch('trading.live.live_trading.mt5')
        self.mock_mt5 = self.mt5_patcher.start()
        
        # Mock joblib
        self.joblib_patcher = patch('trading.live.live_trading.joblib')
        self.mock_joblib = self.joblib_patcher.start()
        
        # Mock PPO
        self.ppo_patcher = patch('trading.live.live_trading.PPO')
        self.mock_ppo = self.ppo_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_path):
            os.unlink(self.config_path)
        
        self.mt5_patcher.stop()
        self.joblib_patcher.stop()
        self.ppo_patcher.stop()
    
    @patch('trading.live.live_trading.FeatureMonitor')
    def test_initialization(self, mock_monitor):
        """Test LiveTradingSystem initialization."""
        # Mock model loading
        mock_ml_model = Mock()
        mock_rl_model = Mock()
        mock_feature_engineer = Mock()
        
        self.mock_joblib.load.side_effect = [mock_ml_model, mock_feature_engineer]
        self.mock_ppo.load.return_value = mock_rl_model
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            system = LiveTradingSystem(self.config_path)
        
        self.assertIsNotNone(system)
        self.assertEqual(system.config['trading']['symbol'], 'EURUSD')
        self.assertEqual(system.ml_model, mock_ml_model)
        self.assertEqual(system.rl_model, mock_rl_model)
    
    def test_connect_mt5_success(self):
        """Test successful MT5 connection."""
        # Mock MT5 initialization
        self.mock_mt5.initialize.return_value = True
        
        # Mock account info
        mock_account = Mock()
        mock_account.login = 12345
        mock_account.server = "TestServer"
        mock_account.balance = 10000.0
        mock_account.equity = 10050.0
        mock_account.margin = 100.0
        mock_account.margin_free = 9950.0
        mock_account.leverage = 100
        mock_account.currency = "USD"
        
        self.mock_mt5.account_info.return_value = mock_account
        
        system = LiveTradingSystem(self.config_path)
        result = system.connect_mt5()
        
        self.assertTrue(result)
        self.assertTrue(system.mt5_connected)
        self.assertEqual(system.account_info['login'], 12345)
    
    def test_connect_mt5_failure(self):
        """Test MT5 connection failure."""
        # Mock MT5 initialization failure
        self.mock_mt5.initialize.return_value = False
        self.mock_mt5.last_error.return_value = "Connection failed"
        
        system = LiveTradingSystem(self.config_path)
        result = system.connect_mt5()
        
        self.assertFalse(result)
        self.assertFalse(system.mt5_connected)
    
    def test_get_market_data_success(self):
        """Test successful market data retrieval."""
        # Mock MT5 rates
        mock_rates = np.array([
            (1640995200, 1.1234, 1.1240, 1.1230, 1.1235, 1000, 0, 0),
            (1640995260, 1.1235, 1.1245, 1.1232, 1.1242, 1200, 0, 0)
        ], dtype=[
            ('time', '<i8'), ('open', '<f8'), ('high', '<f8'), 
            ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'), 
            ('spread', '<i8'), ('real_volume', '<i8')
        ])
        
        self.mock_mt5.copy_rates_from_pos.return_value = mock_rates
        
        system = LiveTradingSystem(self.config_path)
        df = system.get_market_data('EURUSD', 'M5', bars=2)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
    
    def test_get_market_data_retry(self):
        """Test market data retrieval with retry mechanism."""
        # Mock first attempt failure, second success
        self.mock_mt5.copy_rates_from_pos.side_effect = [
            None,  # First attempt fails
            np.array([(1640995200, 1.1234, 1.1240, 1.1230, 1.1235, 1000, 0, 0)],
                    dtype=[('time', '<i8'), ('open', '<f8'), ('high', '<f8'), 
                          ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'), 
                          ('spread', '<i8'), ('real_volume', '<i8')])
        ]
        
        system = LiveTradingSystem(self.config_path)
        df = system.get_market_data('EURUSD', 'M5', bars=1, max_retries=2)
        
        self.assertIsNotNone(df)
        self.assertEqual(self.mock_mt5.copy_rates_from_pos.call_count, 2)
    
    def test_generate_features(self):
        """Test feature generation."""
        # Create test data
        df = pd.DataFrame({
            'open': [1.1234, 1.1235, 1.1236],
            'high': [1.1240, 1.1245, 1.1248],
            'low': [1.1230, 1.1232, 1.1234],
            'close': [1.1235, 1.1242, 1.1246],
            'volume': [1000, 1200, 1100]
        })
        
        system = LiveTradingSystem(self.config_path)
        features = system.generate_features(df)
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIn('returns', features.columns)
        self.assertIn('rsi', features.columns)
        self.assertIn('macd', features.columns)
    
    def test_get_ml_prediction_success(self):
        """Test successful ML prediction."""
        # Mock ML model
        mock_ml_model = Mock()
        mock_ml_model.predict.return_value = np.array([1])
        mock_ml_model.predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])
        
        # Create test features
        features = pd.DataFrame({
            'returns': [0.001],
            'returns_5': [0.002],
            'returns_10': [0.003],
            'volatility': [0.01],
            'volatility_5': [0.008],
            'sma_5': [1.1235],
            'sma_20': [1.1230],
            'sma_ratio': [1.0004],
            'rsi': [65.0],
            'macd': [0.0001],
            'macd_signal': [0.0002],
            'volume_ma': [1100],
            'volume_ratio': [1.1],
            'range': [0.0010],
            'range_ma': [0.0008]
        })
        
        system = LiveTradingSystem(self.config_path)
        system.ml_model = mock_ml_model
        
        result = system.get_ml_prediction(features)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['prediction'], 1)
        self.assertEqual(result['confidence'], 0.7)
    
    def test_get_ml_prediction_retry(self):
        """Test ML prediction with retry mechanism."""
        # Mock ML model that fails first, succeeds second
        mock_ml_model = Mock()
        mock_ml_model.predict.side_effect = [
            ValueError("Model error"),  # First attempt fails
            np.array([1])  # Second attempt succeeds
        ]
        mock_ml_model.predict_proba.side_effect = [
            ValueError("Model error"),  # First attempt fails
            np.array([[0.2, 0.7, 0.1]])  # Second attempt succeeds
        ]
        
        # Create test features
        features = pd.DataFrame({
            'returns': [0.001],
            'returns_5': [0.002],
            'returns_10': [0.003],
            'volatility': [0.01],
            'volatility_5': [0.008],
            'sma_5': [1.1235],
            'sma_20': [1.1230],
            'sma_ratio': [1.0004],
            'rsi': [65.0],
            'macd': [0.0001],
            'macd_signal': [0.0002],
            'volume_ma': [1100],
            'volume_ratio': [1.1],
            'range': [0.0010],
            'range_ma': [0.0008]
        })
        
        system = LiveTradingSystem(self.config_path)
        system.ml_model = mock_ml_model
        
        result = system.get_ml_prediction(features, max_retries=2)
        
        self.assertIsNotNone(result)
        self.assertEqual(mock_ml_model.predict.call_count, 2)
    
    def test_get_rl_action_with_model(self):
        """Test RL action with available model."""
        # Mock RL model
        mock_rl_model = Mock()
        mock_rl_model.predict.return_value = (np.array([1]), None)
        
        # Create test features
        features = pd.DataFrame({
            'returns': [0.001],
            'returns_5': [0.002],
            'returns_10': [0.003],
            'volatility': [0.01],
            'volatility_5': [0.008],
            'sma_5': [1.1235],
            'sma_20': [1.1230],
            'sma_ratio': [1.0004],
            'rsi': [65.0],
            'macd': [0.0001],
            'macd_signal': [0.0002],
            'volume_ma': [1100],
            'volume_ratio': [1.1],
            'range': [0.0010],
            'range_ma': [0.0008]
        })
        
        ml_prediction = {'prediction': 1, 'confidence': 0.8}
        
        system = LiveTradingSystem(self.config_path)
        system.rl_model = mock_rl_model
        
        result = system.get_rl_action(features, ml_prediction)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['action'], 1)
        self.assertEqual(result['source'], 'rl_model')
    
    def test_get_rl_action_fallback(self):
        """Test RL action fallback when model not available."""
        # Create test features
        features = pd.DataFrame({
            'returns': [0.001],
            'returns_5': [0.002],
            'returns_10': [0.003],
            'volatility': [0.01],
            'volatility_5': [0.008],
            'sma_5': [1.1235],
            'sma_20': [1.1230],
            'sma_ratio': [1.0004],
            'rsi': [65.0],
            'macd': [0.0001],
            'macd_signal': [0.0002],
            'volume_ma': [1100],
            'volume_ratio': [1.1],
            'range': [0.0010],
            'range_ma': [0.0008]
        })
        
        ml_prediction = {'prediction': 1, 'confidence': 0.8}
        
        system = LiveTradingSystem(self.config_path)
        system.rl_model = None  # No RL model available
        
        result = system.get_rl_action(features, ml_prediction)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['action'], 1)
        self.assertEqual(result['confidence'], 0.8)
        self.assertEqual(result['source'], 'ml_fallback')


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'trading': {
                'max_positions': 3,
                'max_daily_loss': 100,
                'max_daily_trades': 10
            }
        }
        self.risk_manager = RiskManager(self.config)
    
    def test_check_trading_allowed(self):
        """Test trading permission check."""
        # Test with no current positions
        result = self.risk_manager.check_trading_allowed()
        self.assertTrue(result)
        
        # Test with max positions reached
        self.risk_manager.current_positions = 3
        result = self.risk_manager.check_trading_allowed()
        self.assertFalse(result)
    
    def test_check_trading_conditions(self):
        """Test trading conditions validation."""
        # Test valid conditions
        result = self.risk_manager.check_trading_conditions('EURUSD', 'buy', 0.8)
        self.assertTrue(result)
        
        # Test low confidence
        result = self.risk_manager.check_trading_conditions('EURUSD', 'buy', 0.3)
        self.assertFalse(result)


class TestEntryPointDetector(unittest.TestCase):
    """Test cases for EntryPointDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = EntryPointDetector()
    
    def test_detect_entry_points(self):
        """Test entry point detection."""
        # Create test features
        features = pd.DataFrame({
            'returns': [0.001, 0.002, 0.003],
            'rsi': [65.0, 70.0, 75.0],
            'macd': [0.0001, 0.0002, 0.0003],
            'volume_ratio': [1.1, 1.2, 1.3]
        })
        
        ml_prediction = {'prediction': 1, 'confidence': 0.8}
        
        result = self.detector.detect_entry_points(features, ml_prediction, 0.8)
        
        self.assertIsNotNone(result)
        self.assertIn('signal_strength', result)
        self.assertIn('entry_decision', result)


if __name__ == '__main__':
    unittest.main() 