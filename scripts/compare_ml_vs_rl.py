#!/usr/bin/env python3
"""
Comparison of ML vs RL performance using shared components.
Eliminates duplicate code by using shared utilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import yaml
import joblib
import os
import json
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import shared components
from utils.data_loading import load_mt5_data
from utils.config import load_config, save_results, create_results_filename
from utils.logging import setup_logger, get_logger
from models.rl.utils.data_loader import load_smoteenn_model, load_test_data
from models.rl.utils.visualization import plot_comparison, create_simple_comparison_plot
from models.rl.environments.trading_environment import TradingEnvironment

def evaluate_ml_model(ml_model, test_data, config):
    """Evaluate ML model performance using shared components."""
    logger.info("🤖 Evaluating ML SMOTEENN model...")
    
    # Use exact columns from trained model
    if ml_model.columns_ is not None:
        available_columns = [col for col in ml_model.columns_ if col in test_data.columns]
        missing_columns = [col for col in ml_model.columns_ if col not in test_data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info(f"Using {len(available_columns)} available columns out of {len(ml_model.columns_)}")
        
        if len(available_columns) < len(ml_model.columns_) * 0.8:
            logger.error("Too many missing columns, cannot make prediction")
            return {
                'total_predictions': 0,
                'prediction_distribution': {'hold': 0.33, 'buy': 0.33, 'sell': 0.34},
                'prediction_counts': {0: 0, 1: 0, 2: 0}
            }, np.array([])
        
        features_only = test_data[available_columns]
    else:
        feature_columns = [col for col in test_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        features_only = test_data[feature_columns]
        logger.warning("No training column information, using fallback")
    
    # ML predictions
    try:
        ml_predictions = ml_model.predict(features_only)
    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return {
            'total_predictions': 0,
            'prediction_distribution': {'hold': 0.33, 'buy': 0.33, 'sell': 0.34},
            'prediction_counts': {0: 0, 1: 0, 2: 0}
        }, np.array([])
    
    # Calculate metrics
    unique, counts = np.unique(ml_predictions, return_counts=True)
    prediction_dist = dict(zip(unique, counts))
    
    total_predictions = len(ml_predictions)
    ml_metrics = {
        'total_predictions': total_predictions,
        'prediction_distribution': {
            'hold': prediction_dist.get(0, 0) / total_predictions,
            'buy': prediction_dist.get(1, 0) / total_predictions,
            'sell': prediction_dist.get(2, 0) / total_predictions
        },
        'prediction_counts': prediction_dist
    }
    
    logger.info(f"📊 ML prediction distribution:")
    for action, prop in ml_metrics['prediction_distribution'].items():
        logger.info(f"  {action}: {prop:.2%}")
    
    return ml_metrics, ml_predictions

def simulate_ml_trading(ml_predictions, test_data, config):
    """Simulate ML-based trading using shared environment."""
    logger.info("💰 Simulating ML trading...")
    
    # Create ML environment
    ml_model = load_smoteenn_model(config)
    env = TradingEnvironment(
        data=test_data,
        ml_model=ml_model,
        config=config,
        max_steps=len(test_data),
        mode='testing'
    )
    
    # Simulate trading with ML predictions
    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    
    while not done and not truncated and step < len(ml_predictions):
        # Use ML prediction as action
        action = ml_predictions[step]
        obs, reward, done, truncated, _ = env.step(action)
        step += 1
    
    # Get metrics
    metrics = env.get_metrics()
    
    return env.balance_history, env.pnl_history, env.position_history, metrics

def load_rl_results():
    """Load RL test results."""
    rl_results_path = "reports/rl_smoteenn_test_results.json"
    
    if not os.path.exists(rl_results_path):
        raise FileNotFoundError(f"RL results not found: {rl_results_path}")
    
    with open(rl_results_path, 'r') as f:
        rl_results = json.load(f)
    
    return rl_results

def main():
    """Main comparison function using shared components."""
    
    # Load configuration
    config = load_config()
    
    logger.info("🔄 STARTING ML vs RL COMPARISON WITH SHARED COMPONENTS")
    logger.info(f"💱 Symbol: {config['testing']['symbol']}")
    logger.info(f"⏰ Timeframe: {config['testing']['timeframes'][0]}")
    
    # Load ML model
    ml_model = load_smoteenn_model(config)
    
    # Load test data
    test_data = load_test_data(config)
    
    # Evaluate ML model
    ml_metrics, ml_predictions = evaluate_ml_model(ml_model, test_data, config)
    
    # Simulate ML trading
    ml_balance_history, ml_pnl_history, ml_position_history, ml_trading_metrics = simulate_ml_trading(
        ml_predictions, test_data, config
    )
    
    # Load RL results
    try:
        rl_results = load_rl_results()
        rl_metrics = rl_results.get('metrics', {})
        rl_balance_history = rl_results.get('balance_history', [])
        
        logger.info("📊 RL Results loaded from file")
    except FileNotFoundError:
        logger.warning("RL results file not found, using placeholder data")
        rl_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'action_distribution': {'hold': 0.33, 'buy': 0.33, 'sell': 0.34}
        }
        rl_balance_history = [100000] * len(ml_balance_history)
    
    # Display comparison
    logger.info("📊 PERFORMANCE COMPARISON:")
    logger.info(f"   ML Total Return: {ml_trading_metrics['total_return']*100:.2f}%")
    logger.info(f"   RL Total Return: {rl_metrics.get('total_return', 0)*100:.2f}%")
    logger.info(f"   ML Sharpe Ratio: {ml_trading_metrics['sharpe_ratio']:.2f}")
    logger.info(f"   RL Sharpe Ratio: {rl_metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"   ML Max Drawdown: {ml_trading_metrics['max_drawdown']*100:.2f}%")
    logger.info(f"   RL Max Drawdown: {rl_metrics.get('max_drawdown', 0)*100:.2f}%")
    
    # Create comparison plot
    plot_path = f"reports/ml_vs_rl_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_comparison(
        ml_metrics=ml_trading_metrics,
        rl_metrics=rl_metrics,
        ml_balance_history=ml_balance_history,
        rl_balance_history=rl_balance_history,
        save_path=plot_path
    )
    
    # Save comparison results
    comparison_results = {
        'comparison_date': datetime.now().isoformat(),
        'symbol': config['testing']['symbol'],
        'timeframe': config['testing']['timeframes'][0],
        'ml_metrics': ml_trading_metrics,
        'rl_metrics': rl_metrics,
        'plot_path': plot_path,
        'improvement': {
            'return_improvement': (rl_metrics.get('total_return', 0) - ml_trading_metrics['total_return']) * 100,
            'sharpe_improvement': rl_metrics.get('sharpe_ratio', 0) - ml_trading_metrics['sharpe_ratio'],
            'drawdown_improvement': ml_trading_metrics['max_drawdown'] - rl_metrics.get('max_drawdown', 0)
        }
    }
    
    results_filename = create_results_filename('ml_vs_rl_comparison')
    save_results(comparison_results, results_filename)
    
    logger.info("🎉 ML vs RL comparison completed successfully!")
    logger.info(f"📊 Results saved to: reports/{results_filename}")
    logger.info(f"📈 Plot saved to: {plot_path}")

if __name__ == "__main__":
    main() 