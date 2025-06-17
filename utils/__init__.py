"""
Shared utilities for the trading system.
Eliminates duplicate code across the project.
"""

from .data_loading.mt5_connector import MT5Connector, get_mt5_connector, load_mt5_data
from .config.config_loader import (
    load_config, reload_config, get_config_value, validate_config, 
    save_config, create_default_config, get_model_path, get_feature_engineer_path
)
from .logging.logger_setup import (
    setup_logger, get_logger, log_trading_action, log_model_prediction,
    log_data_loading, log_feature_generation, log_model_training,
    log_model_evaluation, log_error, log_warning, log_success
)
from .testing.test_runner import TestRunner, run_quick_test

__all__ = [
    # Data loading
    'MT5Connector',
    'get_mt5_connector', 
    'load_mt5_data',
    
    # Configuration
    'load_config',
    'reload_config',
    'get_config_value',
    'validate_config',
    'save_config',
    'create_default_config',
    'get_model_path',
    'get_feature_engineer_path',
    
    # Logging
    'setup_logger',
    'get_logger',
    'log_trading_action',
    'log_model_prediction',
    'log_data_loading',
    'log_feature_generation',
    'log_model_training',
    'log_model_evaluation',
    'log_error',
    'log_warning',
    'log_success',
    
    # Testing
    'TestRunner',
    'run_quick_test'
] 