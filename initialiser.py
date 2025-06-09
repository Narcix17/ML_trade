"""
Initializer module for managing caches and saved states.
This module provides functions to initialize, save, and load various cached states
including model states, feature monitors, and other persistent data.
"""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "monitoring": {
        "features": {
            "alert_threshold": 0.7,
            "check_interval": 3600,
            "drift_thresholds": {
                "correlation": 1.0,
                "default": 0.5,
                "mahalanobis": 1.0,
                "mean_shift": 0.3,
                "missing_rate": 0.1,
                "std_shift": 0.3
            },
            "window_size": 1000
        },
        "alerts": {
            "email": {
                "enabled": True,
                "recipients": ["recipient@email.com"],
                "sender": "your-email@gmail.com",
                "smtp_port": 587,
                "smtp_server": "smtp.gmail.com"
            },
            "telegram": {
                "enabled": False,
                "bot_token": "",
                "chat_id": ""
            }
        },
        "drift_detection": {
            "check_interval": 3600,
            "methods": [
                {
                    "name": "ks_test",
                    "threshold": 0.05,
                    "window": 1000
                },
                {
                    "name": "hellinger",
                    "threshold": 0.1,
                    "window": 1000
                },
                {
                    "name": "jensen_shannon",
                    "threshold": 0.1,
                    "window": 1000
                }
            ]
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/trading.log",
            "max_size": 10485760,
            "backup_count": 5
        },
        "performance": {
            "window_size": 1000,
            "update_frequency": 100,
            "metrics": {
                "accuracy": {"threshold": 0.8, "degradation_threshold": 0.1},
                "precision": {"threshold": 0.7, "degradation_threshold": 0.1},
                "recall": {"threshold": 0.7, "degradation_threshold": 0.1},
                "f1": {"threshold": 0.7, "degradation_threshold": 0.1}
            }
        },
        "system": {
            "cpu": {"check_interval": 60, "threshold": 0.8},
            "memory": {"check_interval": 60, "threshold": 0.8},
            "disk": {"check_interval": 300, "threshold": 0.9}
        }
    }
}

class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir (str): Directory where cache files will be stored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Define cache file paths
        self.model_cache_path = self.cache_dir / "model_state.pkl"
        self.feature_monitor_cache_path = self.cache_dir / "feature_monitor_state.pkl"
        self.config_cache_path = self.cache_dir / "config.json"
        self.metrics_cache_path = self.cache_dir / "metrics.json"
        
        # Initialize with default config if none exists
        if not self.config_cache_path.exists():
            self.save_config(DEFAULT_CONFIG)
            logger.info("Initialized with default configuration")
            
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration, merging with defaults if needed."""
        try:
            if self.config_cache_path.exists():
                with open(self.config_cache_path, 'r') as f:
                    config = json.load(f)
                # Ensure monitoring section exists
                if 'monitoring' not in config:
                    config['monitoring'] = DEFAULT_CONFIG['monitoring']
                    self.save_config(config)
                    logger.info("Added missing monitoring section to config")
                return config
            return DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return DEFAULT_CONFIG
            
    def save_model_state(self, model: Any) -> None:
        """Save model state to cache."""
        try:
            with open(self.model_cache_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model state saved to {self.model_cache_path}")
        except Exception as e:
            logger.error(f"Error saving model state: {str(e)}")
            
    def load_model_state(self) -> Optional[Any]:
        """Load model state from cache."""
        try:
            if self.model_cache_path.exists():
                with open(self.model_cache_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model state loaded from {self.model_cache_path}")
                return model
            logger.warning("No model state cache found")
            return None
        except Exception as e:
            logger.error(f"Error loading model state: {str(e)}")
            return None
            
    def save_feature_monitor_state(self, monitor: Any) -> None:
        """Save feature monitor state to cache."""
        try:
            with open(self.feature_monitor_cache_path, 'wb') as f:
                pickle.dump(monitor, f)
            logger.info(f"Feature monitor state saved to {self.feature_monitor_cache_path}")
        except Exception as e:
            logger.error(f"Error saving feature monitor state: {str(e)}")
            
    def load_feature_monitor_state(self) -> Optional[Any]:
        """Load feature monitor state from cache."""
        try:
            if self.feature_monitor_cache_path.exists():
                with open(self.feature_monitor_cache_path, 'rb') as f:
                    monitor = pickle.load(f)
                logger.info(f"Feature monitor state loaded from {self.feature_monitor_cache_path}")
                return monitor
            logger.warning("No feature monitor state cache found")
            return None
        except Exception as e:
            logger.error(f"Error loading feature monitor state: {str(e)}")
            return None
            
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to cache."""
        try:
            # Ensure monitoring section exists
            if 'monitoring' not in config:
                config['monitoring'] = DEFAULT_CONFIG['monitoring']
            with open(self.config_cache_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_cache_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to cache."""
        try:
            with open(self.metrics_cache_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.metrics_cache_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            
    def load_metrics(self) -> Optional[Dict[str, Any]]:
        """Load metrics from cache."""
        try:
            if self.metrics_cache_path.exists():
                with open(self.metrics_cache_path, 'r') as f:
                    metrics = json.load(f)
                logger.info(f"Metrics loaded from {self.metrics_cache_path}")
                return metrics
            logger.warning("No metrics cache found")
            return None
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
            return None
            
    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*"):
                cache_file.unlink()
            logger.info("All cache files cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached files."""
        cache_info = {
            "cache_directory": str(self.cache_dir),
            "files": {}
        }
        
        for cache_file in self.cache_dir.glob("*"):
            cache_info["files"][cache_file.name] = {
                "size": cache_file.stat().st_size,
                "last_modified": cache_file.stat().st_mtime
            }
            
        return cache_info

def initialize_cache(cache_dir: str = "cache") -> CacheManager:
    """
    Initialize the cache system.
    
    Args:
        cache_dir (str): Directory where cache files will be stored
        
    Returns:
        CacheManager: Initialized cache manager instance
    """
    logger.info(f"Initializing cache system in directory: {cache_dir}")
    return CacheManager(cache_dir)

# Example usage:
if __name__ == "__main__":
    # Initialize cache manager
    cache_manager = initialize_cache()
    
    # Get current config (will use defaults if none exists)
    config = cache_manager.get_config()
    print("Current configuration:", json.dumps(config, indent=2))
    
    # Get cache information
    cache_info = cache_manager.get_cache_info()
    print("Cache information:", json.dumps(cache_info, indent=2)) 