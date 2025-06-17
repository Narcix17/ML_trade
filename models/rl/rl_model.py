import numpy as np
import pandas as pd
from typing import List, Optional, Callable
import joblib
import os
from stable_baselines3 import PPO
import logging

logger = logging.getLogger(__name__)

class RLModel:
    """
    Wrapper for RL models (PPO) with strategic features
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained PPO model"""
        try:
            self.model = PPO.load(model_path)
            self.model_path = model_path
            self.is_trained = True
            logger.info(f"Loaded RL model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            return False
    
    def train(self, X: pd.DataFrame, custom_reward_function: Optional[Callable] = None) -> None:
        """
        Train RL model with strategic features
        For now, this is a placeholder - in a full implementation,
        we would create a custom environment with strategic rewards
        """
        logger.info("RL training with strategic features - placeholder implementation")
        
        # Convert features to numpy array
        X_array = X.values.astype(np.float32)
        
        # Simple training simulation (in real implementation, this would use gym environment)
        # For now, we'll just mark as trained and use a simple prediction method
        self.is_trained = True
        logger.info("RL model marked as trained (placeholder)")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict actions using the RL model
        Returns: array of actions (-1: short, 0: hold, 1: long)
        """
        if not self.is_trained or self.model is None:
            # Return random actions if no model
            return np.random.choice([-1, 0, 1], size=len(X))
        
        try:
            # Convert to numpy array
            X_array = X.values.astype(np.float32)
            
            # Get predictions from PPO model
            actions = []
            for i in range(len(X_array)):
                action, _ = self.model.predict(X_array[i], deterministic=True)
                # Convert PPO actions (0,1,2) to trading actions (-1,0,1)
                if action == 0:
                    actions.append(0)  # Hold
                elif action == 1:
                    actions.append(1)  # Long
                elif action == 2:
                    actions.append(-1)  # Short
                else:
                    actions.append(0)  # Default to hold
            
            return np.array(actions)
            
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            # Return random actions as fallback
            return np.random.choice([-1, 0, 1], size=len(X))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict action probabilities
        Returns: array of probability distributions for each action
        """
        if not self.is_trained or self.model is None:
            # Return uniform probabilities if no model
            return np.full((len(X), 3), 1/3)
        
        try:
            X_array = X.values.astype(np.float32)
            probabilities = []
            
            for i in range(len(X_array)):
                # Get action probabilities from PPO model
                action_probs = self.model.policy.get_distribution(X_array[i]).distribution.probs.detach().numpy()
                probabilities.append(action_probs)
            
            return np.array(probabilities)
            
        except Exception as e:
            logger.error(f"RL probability prediction failed: {e}")
            return np.full((len(X), 3), 1/3)
    
    def save_model(self, path: str) -> bool:
        """Save the trained model"""
        if not self.is_trained or self.model is None:
            logger.warning("No trained model to save")
            return False
        
        try:
            self.model.save(path)
            logger.info(f"RL model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save RL model: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the model"""
        info = {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'model_type': 'PPO' if self.model else None
        }
        
        if self.model:
            info['policy_type'] = type(self.model.policy).__name__
            info['learning_rate'] = self.model.learning_rate
        
        return info 