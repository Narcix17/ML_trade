#!/usr/bin/env python3
"""
Shared logging configuration utilities.
Eliminates duplicate code across the project.
"""

import os
import sys
from datetime import datetime
from loguru import logger
from typing import Optional, Dict, Any

def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days",
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
) -> None:
    """
    Setup project-wide logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (None for console only)
        rotation: Log rotation policy
        retention: Log retention policy
        format: Log message format
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=format,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format=format,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"🔧 Logger configured - Level: {log_level}")

def get_logger(name: str = __name__):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)

def log_function_call(func_name: str, args: Dict[str, Any] = None, kwargs: Dict[str, Any] = None):
    """
    Decorator to log function calls.
    
    Args:
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"🔍 Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"✅ {func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"❌ {func_name} failed: {e}")
                raise
        return wrapper
    return decorator

def log_performance(func_name: str):
    """
    Decorator to log function performance.
    
    Args:
        func_name: Function name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.debug(f"⏱️ Starting {func_name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"⚡ {func_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.error(f"💥 {func_name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

def log_trading_action(action: str, symbol: str, volume: float, price: float, **kwargs):
    """
    Log trading actions with consistent format.
    
    Args:
        action: Trading action (BUY, SELL, HOLD)
        symbol: Trading symbol
        volume: Trade volume
        price: Trade price
        **kwargs: Additional trading parameters
    """
    logger.info(f"💰 {action} {volume} {symbol} @ {price:.5f} | {kwargs}")

def log_model_prediction(model_name: str, prediction: Any, confidence: float = None, **kwargs):
    """
    Log model predictions with consistent format.
    
    Args:
        model_name: Name of the model
        prediction: Model prediction
        confidence: Prediction confidence
        **kwargs: Additional prediction parameters
    """
    confidence_str = f" (confidence: {confidence:.3f})" if confidence is not None else ""
    logger.info(f"🤖 {model_name} prediction: {prediction}{confidence_str} | {kwargs}")

def log_data_loading(symbol: str, timeframe: str, start_date: str, end_date: str, data_count: int):
    """
    Log data loading operations with consistent format.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        start_date: Start date
        end_date: End date
        data_count: Number of data points loaded
    """
    logger.info(f"📊 Loaded {data_count} bars for {symbol} ({timeframe}) from {start_date} to {end_date}")

def log_feature_generation(feature_count: int, symbol: str, timeframe: str):
    """
    Log feature generation operations with consistent format.
    
    Args:
        feature_count: Number of features generated
        symbol: Trading symbol
        timeframe: Data timeframe
    """
    logger.info(f"🔧 Generated {feature_count} features for {symbol} ({timeframe})")

def log_model_training(model_name: str, symbol: str, timeframe: str, metrics: Dict[str, Any]):
    """
    Log model training operations with consistent format.
    
    Args:
        model_name: Name of the model
        symbol: Trading symbol
        timeframe: Data timeframe
        metrics: Training metrics
    """
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"🎯 {model_name} training completed for {symbol} ({timeframe}) | {metrics_str}")

def log_model_evaluation(model_name: str, symbol: str, timeframe: str, metrics: Dict[str, Any]):
    """
    Log model evaluation operations with consistent format.
    
    Args:
        model_name: Name of the model
        symbol: Trading symbol
        timeframe: Data timeframe
        metrics: Evaluation metrics
    """
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"📈 {model_name} evaluation for {symbol} ({timeframe}) | {metrics_str}")

def log_error(error: Exception, context: str = "", **kwargs):
    """
    Log errors with consistent format.
    
    Args:
        error: Exception object
        context: Error context
        **kwargs: Additional error parameters
    """
    context_str = f" in {context}" if context else ""
    kwargs_str = f" | {kwargs}" if kwargs else ""
    logger.error(f"❌ Error{context_str}: {error}{kwargs_str}")

def log_warning(message: str, context: str = "", **kwargs):
    """
    Log warnings with consistent format.
    
    Args:
        message: Warning message
        context: Warning context
        **kwargs: Additional warning parameters
    """
    context_str = f" in {context}" if context else ""
    kwargs_str = f" | {kwargs}" if kwargs else ""
    logger.warning(f"⚠️ Warning{context_str}: {message}{kwargs_str}")

def log_success(message: str, context: str = "", **kwargs):
    """
    Log success messages with consistent format.
    
    Args:
        message: Success message
        context: Success context
        **kwargs: Additional success parameters
    """
    context_str = f" in {context}" if context else ""
    kwargs_str = f" | {kwargs}" if kwargs else ""
    logger.success(f"✅ Success{context_str}: {message}{kwargs_str}")

# Initialize default logger
setup_logger() 