#!/usr/bin/env python3
"""
Shared configuration loading utilities.
Eliminates duplicate code across the project.
"""

import yaml
import os
from typing import Dict, Any, Optional
from loguru import logger

# Global config cache
_config_cache = None

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration file with caching.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    # Return cached config if available
    if _config_cache is not None:
        return _config_cache
    
    # Load config from file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Cache the config
        _config_cache = config
        
        logger.info(f"✅ Configuration loaded from: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def reload_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Force reload configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    # Clear cache
    _config_cache = None
    
    # Load fresh config
    return load_config(config_path)

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'ml.model.type')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = [
        'training',
        'testing', 
        'ml',
        'features',
        'rl_environment'
    ]
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate training section
    training_required = ['symbol', 'timeframes', 'start_date', 'end_date']
    for key in training_required:
        if key not in config['training']:
            logger.error(f"Missing training configuration: {key}")
            return False
    
    # Validate testing section
    testing_required = ['symbol', 'timeframes', 'start_date', 'end_date']
    for key in testing_required:
        if key not in config['testing']:
            logger.error(f"Missing testing configuration: {key}")
            return False
    
    # Validate ML section
    if 'model' not in config['ml']:
        logger.error("Missing ML model configuration")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True

def save_config(config: Dict[str, Any], config_path: str = 'config.yaml') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise

def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration template.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'training': {
            'symbol': 'EURUSD',
            'timeframes': ['M5'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        },
        'testing': {
            'symbol': 'EURUSD',
            'timeframes': ['M5'],
            'start_date': '2025-01-01',
            'end_date': '2025-12-31'
        },
        'ml': {
            'model': {
                'type': 'xgboost',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            },
            'features': {
                'technical_indicators': True,
                'price_features': True,
                'volume_features': True
            }
        },
        'features': {
            'technical_indicators': {
                'sma': [10, 20, 50],
                'ema': [10, 20, 50],
                'rsi': [14],
                'macd': [12, 26, 9],
                'bollinger_bands': [20, 2]
            }
        },
        'rl_environment': {
            'initial_balance': 100000.0,
            'max_position_size': 0.1,
            'transaction_cost': 0.0001
        },
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10
        },
        'mt5': {
            'login': None,
            'password': None,
            'server': None
        }
    }

def get_model_path(config: Dict[str, Any], model_type: str, symbol: str, timeframe: str) -> str:
    """
    Generate model path based on configuration.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model (ml, rl, etc.)
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        Model path
    """
    if model_type == 'ml':
        ml_type = get_config_value(config, 'ml.model.type', 'xgboost')
        return f"models/saved/{ml_type}/{symbol}/{timeframe}"
    elif model_type == 'rl':
        return f"models/ppo_smoteenn"
    else:
        return f"models/{model_type}/{symbol}/{timeframe}"

def get_feature_engineer_path(config: Dict[str, Any], symbol: str, timeframe: str) -> str:
    """
    Generate feature engineer path based on configuration.
    
    Args:
        config: Configuration dictionary
        symbol: Trading symbol
        timeframe: Timeframe
        
    Returns:
        Feature engineer path
    """
    return f"models/saved/feature_engineer/{symbol}/{timeframe}" 