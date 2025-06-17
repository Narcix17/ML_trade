"""
Configuration validator with schema validation.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Configuration validator with schema validation."""
    
    def __init__(self):
        """Initialize the validator."""
        self.schema = self._get_schema()
        self.errors = []
        self.warnings = []
    
    def _get_schema(self) -> Dict[str, Any]:
        """Get the configuration schema."""
        return {
            'trading': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'symbol': {'type': 'string', 'required': True},
                    'timeframe': {'type': 'string', 'required': True, 'allowed': ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']},
                    'lot_size': {'type': 'float', 'required': True, 'min': 0.01, 'max': 100.0},
                    'max_positions': {'type': 'integer', 'required': True, 'min': 1, 'max': 10},
                    'sessions': {
                        'type': 'dict',
                        'required': False,
                        'schema': {
                            'asia': {'type': 'dict', 'schema': {'start': 'string', 'end': 'string'}},
                            'eu': {'type': 'dict', 'schema': {'start': 'string', 'end': 'string'}},
                            'us': {'type': 'dict', 'schema': {'start': 'string', 'end': 'string'}},
                            'overlap': {'type': 'dict', 'schema': {'start': 'string', 'end': 'string', 'timezone': 'string'}}
                        }
                    }
                }
            },
            'training': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'symbol': {'type': 'string', 'required': True},
                    'timeframes': {'type': 'list', 'required': True, 'schema': {'type': 'string'}},
                    'start_date': {'type': 'string', 'required': True},
                    'end_date': {'type': 'string', 'required': True},
                    'lookback_periods': {'type': 'integer', 'required': True, 'min': 10, 'max': 1000},
                    'target_horizon': {'type': 'integer', 'required': True, 'min': 1, 'max': 50},
                    'validation_split': {'type': 'float', 'required': True, 'min': 0.1, 'max': 0.5},
                    'test_split': {'type': 'float', 'required': True, 'min': 0.1, 'max': 0.5}
                }
            },
            'testing': {
                'type': 'dict',
                'required': False,
                'schema': {
                    'symbol': {'type': 'string', 'required': True},
                    'timeframes': {'type': 'list', 'required': True, 'schema': {'type': 'string'}},
                    'start_date': {'type': 'string', 'required': True},
                    'end_date': {'type': 'string', 'required': True}
                }
            },
            'ml': {
                'type': 'dict',
                'required': False,
                'schema': {
                    'model': {
                        'type': 'dict',
                        'schema': {
                            'type': {'type': 'string', 'allowed': ['xgboost', 'random_forest', 'lightgbm']},
                            'params': {'type': 'dict'}
                        }
                    }
                }
            },
            'rl': {
                'type': 'dict',
                'required': False,
                'schema': {
                    'algorithm': {'type': 'string', 'allowed': ['ppo', 'a2c', 'dqn']},
                    'env_params': {'type': 'dict'},
                    'training_params': {'type': 'dict'}
                }
            },
            'features': {
                'type': 'dict',
                'required': False,
                'schema': {
                    'alert_threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
                    'window_size': {'type': 'integer', 'min': 100, 'max': 10000}
                }
            },
            'logging': {
                'type': 'dict',
                'required': False,
                'schema': {
                    'level': {'type': 'string', 'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR']},
                    'file': {'type': 'string'},
                    'format': {'type': 'string'}
                }
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        try:
            self._validate_dict(config, self.schema, 'root')
            self._validate_cross_references(config)
            self._validate_paths(config)
            
            if self.errors:
                logger.error(f"Configuration validation failed with {len(self.errors)} errors:")
                for error in self.errors:
                    logger.error(f"  - {error}")
                return False
            
            if self.warnings:
                logger.warning(f"Configuration validation completed with {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    logger.warning(f"  - {warning}")
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any], path: str):
        """Validate dictionary against schema."""
        if not isinstance(data, dict):
            self.errors.append(f"{path}: Expected dict, got {type(data).__name__}")
            return
        
        # Check required fields
        if 'required' in schema and schema['required']:
            for field in schema.get('schema', {}):
                if schema['schema'][field].get('required', False):
                    if field not in data:
                        self.errors.append(f"{path}.{field}: Required field missing")
        
        # Validate each field
        for key, value in data.items():
            field_path = f"{path}.{key}"
            
            if key in schema.get('schema', {}):
                field_schema = schema['schema'][key]
                self._validate_field(value, field_schema, field_path)
            else:
                self.warnings.append(f"{field_path}: Unknown field")
    
    def _validate_field(self, value: Any, schema: Dict[str, Any], path: str):
        """Validate a single field."""
        # Type validation
        if 'type' in schema:
            if not self._check_type(value, schema['type']):
                self.errors.append(f"{path}: Expected {schema['type']}, got {type(value).__name__}")
                return
        
        # Required validation
        if schema.get('required', False) and value is None:
            self.errors.append(f"{path}: Required field is None")
            return
        
        # Min/Max validation
        if 'min' in schema and value is not None:
            if isinstance(value, (int, float)) and value < schema['min']:
                self.errors.append(f"{path}: Value {value} is below minimum {schema['min']}")
        
        if 'max' in schema and value is not None:
            if isinstance(value, (int, float)) and value > schema['max']:
                self.errors.append(f"{path}: Value {value} is above maximum {schema['max']}")
        
        # Allowed values validation
        if 'allowed' in schema and value not in schema['allowed']:
            self.errors.append(f"{path}: Value '{value}' not in allowed values {schema['allowed']}")
        
        # Nested validation
        if 'schema' in schema:
            if schema['type'] == 'dict':
                self._validate_dict(value, schema, path)
            elif schema['type'] == 'list':
                self._validate_list(value, schema, path)
    
    def _validate_list(self, data: List[Any], schema: Dict[str, Any], path: str):
        """Validate list against schema."""
        if not isinstance(data, list):
            self.errors.append(f"{path}: Expected list, got {type(data).__name__}")
            return
        
        if 'schema' in schema:
            for i, item in enumerate(data):
                self._validate_field(item, schema['schema'], path)
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'dict': dict,
            'list': list
        }
        
        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])
        
        return True
    
    def _validate_cross_references(self, config: Dict[str, Any]):
        """Validate cross-references between sections."""
        # Check symbol consistency
        trading_symbol = config.get('trading', {}).get('symbol')
        training_symbol = config.get('training', {}).get('symbol')
        testing_symbol = config.get('testing', {}).get('symbol')
        
        symbols = [s for s in [trading_symbol, training_symbol, testing_symbol] if s]
        if len(set(symbols)) > 1:
            self.warnings.append(f"Symbol inconsistency: trading={trading_symbol}, training={training_symbol}, testing={testing_symbol}")
        
        # Check timeframe consistency
        trading_tf = config.get('trading', {}).get('timeframe')
        training_tfs = config.get('training', {}).get('timeframes', [])
        
        if trading_tf and training_tfs and trading_tf not in training_tfs:
            self.warnings.append(f"Trading timeframe {trading_tf} not in training timeframes {training_tfs}")
        
        # Check date consistency
        training_start = config.get('training', {}).get('start_date')
        training_end = config.get('training', {}).get('end_date')
        testing_start = config.get('testing', {}).get('start_date')
        testing_end = config.get('testing', {}).get('end_date')
        
        if training_start and training_end and training_start >= training_end:
            self.errors.append("Training start_date must be before end_date")
        
        if testing_start and testing_end and testing_start >= testing_end:
            self.errors.append("Testing start_date must be before end_date")
        
        if training_end and testing_start and training_end >= testing_start:
            self.warnings.append("Training and testing periods overlap")
    
    def _validate_paths(self, config: Dict[str, Any]):
        """Validate file and directory paths."""
        # Check if model paths exist
        symbol = config.get('training', {}).get('symbol', 'EURUSD')
        timeframe = config.get('training', {}).get('timeframes', ['M5'])[0]
        
        model_path = f"models/saved/xgboost/{symbol}/{timeframe}/ml_model.joblib"
        if not os.path.exists(model_path):
            self.warnings.append(f"ML model not found: {model_path}")
        
        # Check if logs directory exists
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            self.warnings.append(f"Logs directory not found: {logs_dir}")
    
    def validate_config_file(self, config_path: str) -> bool:
        """
        Validate configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return self.validate_config(config)
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            logger.error(f"Configuration file error: {e}")
            return False
    
    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings."""
        return self.warnings


def validate_config(config_path: str = "config.yaml") -> bool:
    """
    Convenience function to validate configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator()
    return validator.validate_config_file(config_path)


if __name__ == '__main__':
    # Test the validator
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    if validate_config(config_path):
        print("✅ Configuration is valid")
        sys.exit(0)
    else:
        print("❌ Configuration is invalid")
        sys.exit(1) 