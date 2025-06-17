#!/usr/bin/env python3
"""
RL Testing Script using shared components.
Eliminates duplicate code by using shared utilities and environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import yaml
import joblib
from datetime import datetime
from loguru import logger
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

# Import shared components
from models.rl.environments.trading_environment import TradingEnvironment
from models.rl.utils.data_loader import load_smoteenn_model, load_test_data, load_rl_model, load_config, save_results, create_results_filename
from models.rl.utils.visualization import plot_trading_results

def main():
    """Main RL testing function using shared components."""
    
    # Load configuration
    config = load_config()
    
    logger.info("🧪 STARTING RL TESTING WITH SHARED COMPONENTS")
    logger.info(f"💱 Symbol: {config['testing']['symbol']}")
    logger.info(f"⏰ Timeframe: {config['testing']['timeframes'][0]}")
    logger.info(f"📅 Period: {config['testing']['start_date']} to {config['testing']['end_date']}")
    
    # Load ML model SMOTEENN
    ml_model = load_smoteenn_model(config)
    
    # Load test data
    test_data = load_test_data(config)
    
    # Load RL model
    rl_model = load_rl_model(config)
    
    # Create testing environment
    env = TradingEnvironment(
        data=test_data,
        ml_model=ml_model,
        config=config,
        max_steps=len(test_data),
        mode='testing'
    )
    
    # Test the model
    logger.info("🎯 Testing RL model...")
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    while not done and not truncated:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
    
    # Get metrics
    metrics = env.get_metrics()
    
    # Display results
    logger.info("📊 RL Testing Results:")
    logger.info(f"   Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"   Annualized Return: {metrics['annualized_return']*100:.2f}%")
    logger.info(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"   Total PnL: ${metrics['total_pnl']:.2f}")
    logger.info(f"   Final Balance: ${metrics['final_balance']:.2f}")
    logger.info(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
    logger.info(f"   Total Trades: {metrics['total_trades']}")
    
    logger.info("📈 Action Distribution:")
    for action, percentage in metrics['action_distribution'].items():
        logger.info(f"   {action.capitalize()}: {percentage*100:.1f}%")
    
    # Create visualization
    plot_path = f"reports/rl_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_trading_results(
        balance_history=env.balance_history,
        pnl_history=env.pnl_history,
        action_history=env.action_history,
        price_history=env.price_history,
        metrics=metrics,
        save_path=plot_path
    )
    
    # Save results
    test_results = {
        'test_date': datetime.now().isoformat(),
        'symbol': config['testing']['symbol'],
        'timeframe': config['testing']['timeframes'][0],
        'metrics': metrics,
        'plot_path': plot_path
    }
    
    results_filename = create_results_filename('rl_test_results')
    save_results(test_results, results_filename)
    
    logger.info("🎉 RL testing completed successfully!")
    logger.info(f"📊 Results saved to: reports/{results_filename}")
    logger.info(f"📈 Plot saved to: {plot_path}")

if __name__ == "__main__":
    main() 