#!/usr/bin/env python3
"""
RL Training Script using shared components.
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch

# Import shared components
from models.rl.environments.trading_environment import TradingEnvironment
from models.rl.utils.data_loader import load_smoteenn_model, load_training_data, load_config, save_results, create_results_filename

def main():
    """Main RL training function using shared components."""
    
    # Load configuration
    config = load_config()
    
    logger.info("🚀 STARTING RL TRAINING WITH SHARED COMPONENTS")
    logger.info(f"💱 Symbol: {config['training']['symbol']}")
    logger.info(f"⏰ Timeframe: {config['training']['timeframes'][0]}")
    logger.info(f"📅 Period: {config['training']['start_date']} to {config['training']['end_date']}")
    
    # Load ML model SMOTEENN
    ml_model = load_smoteenn_model(config)
    
    # Load training data
    training_data = load_training_data(config)
    
    # Create training environment
    env = TradingEnvironment(
        data=training_data,
        ml_model=ml_model,
        config=config,
        max_steps=1000,
        mode='training'
    )
    
    # Wrap environment for vectorization
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Normalize observations
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # PPO parameters
    ppo_params = {
        'learning_rate': float(config['ppo']['learning_rate']),
        'n_steps': config['ppo']['n_steps'],
        'batch_size': config['ppo']['batch_size'],
        'n_epochs': config['ppo']['n_epochs'],
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        'normalize_advantage': True,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
        'tensorboard_log': None,
        'policy_kwargs': None,
        'verbose': 1,
        'seed': None,
        'device': 'auto',
        '_init_setup_model': True
    }
    
    # Create PPO model
    model = PPO("MlpPolicy", env, **ppo_params)
    
    # Callbacks
    eval_env = TradingEnvironment(
        data=training_data.tail(1000),  # Use last 1000 samples for evaluation
        ml_model=ml_model,
        config=config,
        max_steps=1000,
        mode='training'
    )
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo_smoteenn/",
        log_path="models/ppo_smoteenn/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/ppo_smoteenn/",
        name_prefix="ppo_smoteenn"
    )
    
    # Training
    logger.info("🎯 Starting PPO training...")
    total_timesteps = 100000
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = "models/ppo_smoteenn/ppo_smoteenn_final.zip"
    model.save(final_model_path)
    logger.info(f"✅ Final model saved to: {final_model_path}")
    
    # Save vectorized environment
    env.save("models/ppo_smoteenn/vec_normalize.pkl")
    logger.info("✅ Vectorized environment saved")
    
    # Training results
    training_results = {
        'training_date': datetime.now().isoformat(),
        'symbol': config['training']['symbol'],
        'timeframe': config['training']['timeframes'][0],
        'total_timesteps': total_timesteps,
        'model_path': final_model_path,
        'ppo_params': ppo_params
    }
    
    # Save results
    results_filename = create_results_filename('rl_training_results')
    save_results(training_results, results_filename)
    
    logger.info("🎉 RL training completed successfully!")
    logger.info(f"📊 Results saved to: reports/{results_filename}")

if __name__ == "__main__":
    main() 