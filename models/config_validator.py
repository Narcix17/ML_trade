from typing import Dict, Any, List
import yaml
from loguru import logger
import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationError:
    section: str
    field: str
    message: str

class ConfigValidator:
    """Validates the trading system configuration file."""
    
    REQUIRED_SECTIONS = [
        "broker", "trading", "features", "ml", "monitoring", 
        "backtesting", "execution"
    ]
    
    REQUIRED_FIELDS = {
        "broker": ["name", "account_type", "login", "server", "symbols", "timeframes"],
        "trading": ["risk_management", "stops", "filters", "sessions"],
        "ml": ["model", "training", "features"],
        "monitoring": ["drift_detection", "performance", "system", "logging"],
        "backtesting": ["general", "execution", "risk", "analysis"],
        "execution": ["order", "position", "risk", "monitoring"]
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the validator with the path to the config file."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.errors: List[ValidationError] = []
        
    def load_config(self) -> bool:
        """Load and parse the YAML configuration file."""
        try:
            if not os.path.exists(self.config_path):
                self.errors.append(ValidationError(
                    section="root",
                    field="config_path",
                    message=f"Configuration file not found: {self.config_path}"
                ))
                return False
                
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            self.errors.append(ValidationError(
                section="root",
                field="yaml_parse",
                message=f"Failed to parse YAML: {str(e)}"
            ))
            return False
            
    def validate(self) -> bool:
        """Run all validation checks on the configuration."""
        if not self.load_config():
            return False
            
        self._validate_required_sections()
        self._validate_required_fields()
        self._validate_broker_config()
        self._validate_ml_config()
        self._validate_monitoring_config()
        self._validate_backtesting_config()
        self._validate_execution_config()
        
        return len(self.errors) == 0
        
    def _validate_required_sections(self):
        """Check that all required sections are present."""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                self.errors.append(ValidationError(
                    section="root",
                    field=section,
                    message=f"Missing required section: {section}"
                ))
                
    def _validate_required_fields(self):
        """Check that all required fields are present in each section."""
        for section, fields in self.REQUIRED_FIELDS.items():
            if section not in self.config:
                continue
            for field in fields:
                if field not in self.config[section]:
                    self.errors.append(ValidationError(
                        section=section,
                        field=field,
                        message=f"Missing required field: {field}"
                    ))
                    
    def _validate_broker_config(self):
        """Validate broker-specific configuration."""
        if "broker" not in self.config:
            return
            
        broker = self.config["broker"]
        
        # Validate account credentials
        if broker.get("account_type") == "demo" and not broker.get("password"):
            logger.warning("Demo account password is empty - this is fine for testing")
            
        # Validate symbols
        if "symbols" in broker:
            for symbol in broker["symbols"]:
                required_symbol_fields = ["name", "type", "exchange", "point", 
                                       "lot_step", "min_lot", "max_lot", "margin_required"]
                for field in required_symbol_fields:
                    if field not in symbol:
                        self.errors.append(ValidationError(
                            section="broker.symbols",
                            field=field,
                            message=f"Missing required field in symbol {symbol.get('name', 'unknown')}: {field}"
                        ))
                        
    def _validate_ml_config(self):
        """Validate machine learning configuration."""
        if "ml" not in self.config:
            return
            
        ml = self.config["ml"]
        
        # Validate model type
        if "model" in ml and "type" in ml["model"]:
            if ml["model"]["type"] not in ["xgboost", "lightgbm"]:
                self.errors.append(ValidationError(
                    section="ml.model",
                    field="type",
                    message=f"Invalid model type: {ml['model']['type']}. Must be 'xgboost' or 'lightgbm'"
                ))
                
        # Validate training parameters
        if "training" in ml:
            training = ml["training"]
            if "train_test_split" in training:
                split = training["train_test_split"]
                if not 0 < split < 1:
                    self.errors.append(ValidationError(
                        section="ml.training",
                        field="train_test_split",
                        message=f"Invalid train_test_split: {split}. Must be between 0 and 1"
                    ))
                    
    def _validate_monitoring_config(self):
        """Validate monitoring configuration."""
        if "monitoring" not in self.config:
            return
            
        monitoring = self.config["monitoring"]
        
        # Validate drift detection methods
        if "drift_detection" in monitoring and "methods" in monitoring["drift_detection"]:
            valid_methods = ["ks_test", "hellinger", "jensen_shannon"]
            for method in monitoring["drift_detection"]["methods"]:
                if method["name"] not in valid_methods:
                    self.errors.append(ValidationError(
                        section="monitoring.drift_detection",
                        field="methods",
                        message=f"Invalid drift detection method: {method['name']}"
                    ))
                    
    def _validate_backtesting_config(self):
        """Validate backtesting configuration."""
        if "backtesting" not in self.config:
            return
            
        backtest = self.config["backtesting"]
        
        # Validate date range
        if "general" in backtest:
            general = backtest["general"]
            try:
                start_date = datetime.strptime(general["start_date"], "%Y-%m-%d")
                end_date = datetime.strptime(general["end_date"], "%Y-%m-%d")
                if start_date >= end_date:
                    self.errors.append(ValidationError(
                        section="backtesting.general",
                        field="date_range",
                        message="start_date must be before end_date"
                    ))
            except ValueError as e:
                self.errors.append(ValidationError(
                    section="backtesting.general",
                    field="date_format",
                    message=f"Invalid date format: {str(e)}"
                ))
                
    def _validate_execution_config(self):
        """Validate execution configuration."""
        if "execution" not in self.config:
            return
            
        execution = self.config["execution"]
        
        # Validate order type
        if "order" in execution and "type" in execution["order"]:
            if execution["order"]["type"] not in ["market", "limit"]:
                self.errors.append(ValidationError(
                    section="execution.order",
                    field="type",
                    message=f"Invalid order type: {execution['order']['type']}"
                ))
                
    def get_errors(self) -> List[ValidationError]:
        """Return all validation errors."""
        return self.errors
        
    def print_errors(self):
        """Print all validation errors in a readable format."""
        if not self.errors:
            logger.info("Configuration validation successful!")
            return
            
        logger.error("Configuration validation failed with the following errors:")
        for error in self.errors:
            logger.error(f"[{error.section}] {error.field}: {error.message}")
            
def validate_config(config_path: str = "config.yaml") -> bool:
    """Convenience function to validate the configuration file."""
    validator = ConfigValidator(config_path)
    is_valid = validator.validate()
    if not is_valid:
        validator.print_errors()
    return is_valid 