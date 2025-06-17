#!/usr/bin/env python3
"""
Shared utilities for loading ML models and data.
Eliminates duplicate code between training and testing scripts.
"""

import joblib
import pandas as pd
import numpy as np
import os
import yaml
from typing import Tuple, Optional
from loguru import logger

def load_smoteenn_model(config: dict):
    """
    Load the trained ML model with SMOTEENN.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loaded ML model
    """
    model_type = config['ml']['model']['type']
    symbol = config['training']['symbol']
    timeframe = config['training']['timeframes'][0]
    
    model_dir = f"models/saved/{model_type}/{symbol}/{timeframe}"
    ml_model_path = f"{model_dir}/ml_model.joblib"
    
    if not os.path.exists(ml_model_path):
        raise FileNotFoundError(f"ML model not found: {ml_model_path}")
    
    logger.info(f"Loading ML model from: {ml_model_path}")
    ml_model = joblib.load(ml_model_path)
    
    return ml_model

def load_training_data(config: dict) -> pd.DataFrame:
    """
    Load training data for RL.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with features
    """
    symbol = config['training']['symbol']
    timeframe = config['training']['timeframes'][0]
    start_date = config['training']['start_date']
    end_date = config['training']['end_date']
    
    # Load data from MT5 using the training module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from training.main import load_mt5_data
    
    df = load_mt5_data(symbol, start_date, end_date, timeframe)
    
    if df.empty:
        raise ValueError("Unable to load data")
    
    # Generate features
    from features.feature_engineering import FeatureEngineer
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.generate_features(df)
    
    # Clean NaN values
    features = features.fillna(0)
    
    logger.info(f"Training data loaded: {features.shape}")
    
    return features

def load_test_data(config: dict) -> pd.DataFrame:
    """
    Load test data for RL evaluation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with features
    """
    symbol = config['testing']['symbol']
    timeframe = config['testing']['timeframes'][0]
    start_date = config['testing']['start_date']
    end_date = config['testing']['end_date']
    
    # Load data from MT5 using the training module
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from training.main import load_mt5_data
    
    df = load_mt5_data(symbol, start_date, end_date, timeframe)
    
    if df.empty:
        raise ValueError("Unable to load data")
    
    # Generate features
    from features.feature_engineering import FeatureEngineer
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.generate_features(df)
    
    # Clean NaN values
    features = features.fillna(0)
    
    logger.info(f"Test data loaded: {features.shape}")
    
    return features

def load_rl_model(config: dict):
    """
    Load the trained RL model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loaded RL model
    """
    from stable_baselines3 import PPO
    
    rl_model_path = "models/ppo_smoteenn/ppo_smoteenn_final.zip"
    
    if not os.path.exists(rl_model_path):
        raise FileNotFoundError(f"RL model not found: {rl_model_path}")
    
    logger.info(f"Loading RL model from: {rl_model_path}")
    rl_model = PPO.load(rl_model_path)
    
    return rl_model

def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_results(results: dict, filename: str, output_dir: str = 'reports') -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        filename: Output filename
        output_dir: Output directory
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")

def create_results_filename(prefix: str, extension: str = 'json') -> str:
    """
    Create a standardized filename for results.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        Formatted filename
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}" 