"""
Unit tests for ML model functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.ml_model import MLModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class TestMLModel(unittest.TestCase):
    """Test cases for MLModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ml': {
                'model': {
                    'type': 'xgboost',
                    'params': {
                        'max_depth': 3,
                        'learning_rate': 0.1,
                        'n_estimators': 100
                    }
                }
            },
            'features': {
                'alert_threshold': 0.7,
                'window_size': 1000
            }
        }
        
        # Create synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=15,
            n_classes=3,
            n_informative=10,
            random_state=42
        )
        
        self.features = pd.DataFrame(X, columns=[
            'returns', 'returns_5', 'returns_10', 'volatility', 'volatility_5',
            'sma_5', 'sma_20', 'sma_ratio', 'rsi', 'macd', 'macd_signal',
            'volume_ma', 'volume_ratio', 'range', 'range_ma'
        ])
        self.labels = pd.Series(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )
        
        self.model = MLModel(self.config)
    
    def test_initialization(self):
        """Test MLModel initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.model_type, 'xgboost')
        self.assertEqual(self.model.random_state, 42)
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.calibrator)
    
    def test_prepare_data(self):
        """Test data preparation functionality."""
        X_dict, y_dict = self.model.prepare_data(
            self.features, self.labels, test_size=0.2, val_size=0.1
        )
        
        # Check structure
        self.assertIn('train', X_dict)
        self.assertIn('val', X_dict)
        self.assertIn('test', X_dict)
        self.assertIn('train', y_dict)
        self.assertIn('val', y_dict)
        self.assertIn('test', y_dict)
        
        # Check sizes
        self.assertGreater(len(X_dict['train']), 0)
        self.assertGreater(len(X_dict['val']), 0)
        self.assertGreater(len(X_dict['test']), 0)
    
    def test_train_model(self):
        """Test model training."""
        # Mock the model to avoid actual training
        with patch('models.ml_model.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1, 2])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            
            metrics = self.model.train(self.X_train, self.y_train)
            
            # Verify training was called
            mock_model.fit.assert_called_once()
            
            # Check metrics structure
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
    
    def test_predict(self):
        """Test prediction functionality."""
        # Train a simple model first
        with patch('models.ml_model.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1, 2])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            
            self.model.train(self.X_train, self.y_train)
            
            # Test prediction
            predictions = self.model.predict(self.X_test)
            self.assertIsInstance(predictions, np.ndarray)
            self.assertEqual(len(predictions), len(self.X_test))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        # Train a simple model first
        with patch('models.ml_model.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1, 2])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            
            self.model.train(self.X_train, self.y_train)
            
            # Test probability prediction
            probas = self.model.predict_proba(self.X_test)
            self.assertIsInstance(probas, np.ndarray)
            self.assertEqual(probas.shape[0], len(self.X_test))
            self.assertEqual(probas.shape[1], 3)  # 3 classes
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Train a simple model first
        with patch('models.ml_model.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1, 2])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            
            self.model.train(self.X_train, self.y_train)
            
            # Test evaluation
            metrics = self.model.evaluate(self.X_test, self.y_test)
            
            # Check metrics
            self.assertIsInstance(metrics, dict)
            self.assertIn('accuracy', metrics)
            self.assertIn('precision_macro', metrics)
            self.assertIn('recall_macro', metrics)
            self.assertIn('f1_macro', metrics)
    
    def test_save_load_model(self):
        """Test model saving and loading."""
        # Train a simple model first
        with patch('models.ml_model.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1, 2])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            
            self.model.train(self.X_train, self.y_train)
            
            # Test saving
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                save_path = tmp_file.name
            
            try:
                self.model.save_model(save_path)
                self.assertTrue(os.path.exists(save_path))
                
                # Test loading
                new_model = MLModel(self.config)
                new_model.load_model(save_path)
                self.assertIsNotNone(new_model.model)
                
            finally:
                if os.path.exists(save_path):
                    os.unlink(save_path)
    
    def test_column_validation(self):
        """Test column validation in prediction."""
        # Train a simple model first
        with patch('models.ml_model.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1, 2])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            
            self.model.train(self.X_train, self.y_train)
            
            # Test with wrong columns
            wrong_features = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
            
            with self.assertRaises(ValueError):
                self.model.predict(wrong_features)
    
    def test_nan_handling(self):
        """Test NaN handling in features."""
        # Create features with NaN values
        features_with_nan = self.X_test.copy()
        features_with_nan.iloc[0, 0] = np.nan
        
        # Train a simple model first
        with patch('models.ml_model.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1, 2])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            
            self.model.train(self.X_train, self.y_train)
            
            # Test prediction with NaN should handle gracefully
            try:
                predictions = self.model.predict(features_with_nan)
                # If it doesn't raise an exception, it should handle NaN
                self.assertIsInstance(predictions, np.ndarray)
            except Exception as e:
                # If it raises an exception, it should be a meaningful one
                self.assertIn('NaN', str(e))


if __name__ == '__main__':
    unittest.main() 