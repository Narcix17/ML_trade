#!/usr/bin/env python3
"""
Shared testing utilities and test runner.
Eliminates duplicate code across test scripts.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.config.config_loader import load_config
from utils.data_loading.mt5_connector import load_mt5_data
from utils.logging.logger_setup import get_logger

logger = get_logger(__name__)

class TestRunner:
    """Shared test runner for all project components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize test runner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.test_results = {}
    
    def test_mt5_connection(self) -> bool:
        """
        Test MT5 connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info("🔌 Testing MT5 connection...")
        
        try:
            from utils.data_loading.mt5_connector import get_mt5_connector
            connector = get_mt5_connector(self.config)
            
            if connector.connect():
                # Test data loading
                symbol = self.config['training']['symbol']
                start_date = self.config['training']['start_date']
                end_date = self.config['training']['end_date']
                timeframe = self.config['training']['timeframes'][0]
                
                data = connector.load_data(symbol, start_date, end_date, timeframe)
                
                if not data.empty:
                    logger.success(f"✅ MT5 connection test passed - Loaded {len(data)} bars")
                    self.test_results['mt5_connection'] = {
                        'status': 'PASSED',
                        'data_count': len(data),
                        'symbol': symbol,
                        'timeframe': timeframe
                    }
                    return True
                else:
                    logger.error("❌ MT5 connection test failed - No data loaded")
                    self.test_results['mt5_connection'] = {
                        'status': 'FAILED',
                        'error': 'No data loaded'
                    }
                    return False
            else:
                logger.error("❌ MT5 connection test failed - Connection failed")
                self.test_results['mt5_connection'] = {
                    'status': 'FAILED',
                    'error': 'Connection failed'
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ MT5 connection test failed: {e}")
            self.test_results['mt5_connection'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_feature_engineering(self) -> bool:
        """
        Test feature engineering.
        
        Returns:
            True if test passed, False otherwise
        """
        logger.info("🔧 Testing feature engineering...")
        
        try:
            # Load test data
            symbol = self.config['testing']['symbol']
            start_date = self.config['testing']['start_date']
            end_date = self.config['testing']['end_date']
            timeframe = self.config['testing']['timeframes'][0]
            
            data = load_mt5_data(symbol, start_date, end_date, timeframe)
            
            if data.empty:
                logger.error("❌ Feature engineering test failed - No data available")
                self.test_results['feature_engineering'] = {
                    'status': 'FAILED',
                    'error': 'No data available'
                }
                return False
            
            # Generate features
            from features.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer(self.config)
            features = feature_engineer.generate_features(data)
            
            if features.empty:
                logger.error("❌ Feature engineering test failed - No features generated")
                self.test_results['feature_engineering'] = {
                    'status': 'FAILED',
                    'error': 'No features generated'
                }
                return False
            
            # Check for NaN values
            nan_count = features.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"⚠️ Feature engineering test - {nan_count} NaN values found")
            
            logger.success(f"✅ Feature engineering test passed - Generated {len(features.columns)} features")
            self.test_results['feature_engineering'] = {
                'status': 'PASSED',
                'feature_count': len(features.columns),
                'data_shape': features.shape,
                'nan_count': nan_count
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature engineering test failed: {e}")
            self.test_results['feature_engineering'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_ml_model(self) -> bool:
        """
        Test ML model loading and prediction.
        
        Returns:
            True if test passed, False otherwise
        """
        logger.info("🤖 Testing ML model...")
        
        try:
            # Load ML model
            model_type = self.config['ml']['model']['type']
            symbol = self.config['training']['symbol']
            timeframe = self.config['training']['timeframes'][0]
            
            model_dir = f"models/saved/{model_type}/{symbol}/{timeframe}"
            ml_model_path = f"{model_dir}/ml_model.joblib"
            
            if not os.path.exists(ml_model_path):
                logger.error(f"❌ ML model test failed - Model not found: {ml_model_path}")
                self.test_results['ml_model'] = {
                    'status': 'FAILED',
                    'error': f'Model not found: {ml_model_path}'
                }
                return False
            
            ml_model = joblib.load(ml_model_path)
            
            # Load test data
            symbol = self.config['testing']['symbol']
            start_date = self.config['testing']['start_date']
            end_date = self.config['testing']['end_date']
            timeframe = self.config['testing']['timeframes'][0]
            
            data = load_mt5_data(symbol, start_date, end_date, timeframe)
            
            if data.empty:
                logger.error("❌ ML model test failed - No test data available")
                self.test_results['ml_model'] = {
                    'status': 'FAILED',
                    'error': 'No test data available'
                }
                return False
            
            # Generate features
            from features.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer(self.config)
            features = feature_engineer.generate_features(data)
            
            if features.empty:
                logger.error("❌ ML model test failed - No features generated")
                self.test_results['ml_model'] = {
                    'status': 'FAILED',
                    'error': 'No features generated'
                }
                return False
            
            # Make prediction
            if hasattr(ml_model, 'columns_') and ml_model.columns_ is not None:
                # Use exact columns from trained model
                available_columns = [col for col in ml_model.columns_ if col in features.columns]
                if len(available_columns) < len(ml_model.columns_) * 0.8:
                    logger.error("❌ ML model test failed - Too many missing columns")
                    self.test_results['ml_model'] = {
                        'status': 'FAILED',
                        'error': 'Too many missing columns'
                    }
                    return False
                features_only = features[available_columns]
            else:
                # Fallback
                feature_columns = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                features_only = features[feature_columns]
            
            # Clean NaN values
            features_only = features_only.fillna(0)
            
            # Make prediction
            prediction = ml_model.predict(features_only.iloc[:1])
            
            logger.success(f"✅ ML model test passed - Prediction: {prediction[0]}")
            self.test_results['ml_model'] = {
                'status': 'PASSED',
                'model_path': ml_model_path,
                'prediction': int(prediction[0]),
                'feature_count': len(features_only.columns)
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ ML model test failed: {e}")
            self.test_results['ml_model'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def test_saved_feature_engineer(self) -> bool:
        """
        Test saved feature engineer.
        
        Returns:
            True if test passed, False otherwise
        """
        logger.info("🔧 Testing saved feature engineer...")
        
        try:
            # Load saved feature engineer
            symbol = self.config['training']['symbol']
            timeframe = self.config['training']['timeframes'][0]
            
            fe_dir = f"models/saved/feature_engineer/{symbol}/{timeframe}"
            fe_path = f"{fe_dir}/feature_engineer.joblib"
            
            if not os.path.exists(fe_path):
                logger.error(f"❌ Saved FE test failed - Feature engineer not found: {fe_path}")
                self.test_results['saved_feature_engineer'] = {
                    'status': 'FAILED',
                    'error': f'Feature engineer not found: {fe_path}'
                }
                return False
            
            feature_engineer = joblib.load(fe_path)
            
            # Load test data
            symbol = self.config['testing']['symbol']
            start_date = self.config['testing']['start_date']
            end_date = self.config['testing']['end_date']
            timeframe = self.config['testing']['timeframes'][0]
            
            data = load_mt5_data(symbol, start_date, end_date, timeframe)
            
            if data.empty:
                logger.error("❌ Saved FE test failed - No test data available")
                self.test_results['saved_feature_engineer'] = {
                    'status': 'FAILED',
                    'error': 'No test data available'
                }
                return False
            
            # Generate features using saved FE
            features = feature_engineer.generate_features(data)
            
            if features.empty:
                logger.error("❌ Saved FE test failed - No features generated")
                self.test_results['saved_feature_engineer'] = {
                    'status': 'FAILED',
                    'error': 'No features generated'
                }
                return False
            
            logger.success(f"✅ Saved FE test passed - Generated {len(features.columns)} features")
            self.test_results['saved_feature_engineer'] = {
                'status': 'PASSED',
                'feature_count': len(features.columns),
                'data_shape': features.shape
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ Saved FE test failed: {e}")
            self.test_results['saved_feature_engineer'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all available tests.
        
        Returns:
            Dictionary with all test results
        """
        logger.info("🧪 Running all tests...")
        
        tests = [
            self.test_mt5_connection,
            self.test_feature_engineering,
            self.test_ml_model,
            self.test_saved_feature_engineer
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed with exception: {e}")
        
        # Summary
        logger.info(f"📊 Test Summary: {passed}/{total} tests passed")
        
        if passed == total:
            logger.success("🎉 All tests passed!")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed")
        
        self.test_results['summary'] = {
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': passed / total * 100
        }
        
        return self.test_results
    
    def save_test_results(self, filename: Optional[str] = None) -> str:
        """
        Save test results to file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_results_{timestamp}.json"
        
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📁 Test results saved to: {output_path}")
        return output_path

def run_quick_test() -> bool:
    """
    Run a quick test of essential components.
    
    Returns:
        True if all essential tests pass, False otherwise
    """
    logger.info("⚡ Running quick test...")
    
    runner = TestRunner()
    
    # Essential tests only
    essential_tests = [
        runner.test_mt5_connection,
        runner.test_feature_engineering
    ]
    
    passed = 0
    for test_func in essential_tests:
        if test_func():
            passed += 1
    
    success = passed == len(essential_tests)
    
    if success:
        logger.success("✅ Quick test passed!")
    else:
        logger.error(f"❌ Quick test failed: {passed}/{len(essential_tests)} essential tests passed")
    
    return success

if __name__ == "__main__":
    # Run all tests when executed directly
    runner = TestRunner()
    results = runner.run_all_tests()
    runner.save_test_results() 