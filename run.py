#!/usr/bin/env python3
"""
Main entry point for the trading system.
Provides unified access to all system components.
"""

import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from utils.logging import setup_logger, get_logger
from utils.config import load_config

logger = get_logger(__name__)

def run_live_trading():
    """Run live trading system."""
    logger.info("🚀 Starting live trading system...")
    from trading.live.live_trading import LiveTradingSystem
    
    trading_system = LiveTradingSystem()
    if trading_system.connect_mt5():
        try:
            trading_system.run_live_trading()
        except KeyboardInterrupt:
            logger.info("🛑 Live trading stopped by user")
        finally:
            trading_system.disconnect()
    else:
        logger.error("❌ Failed to connect to MT5")

def run_training():
    """Run ML model training."""
    logger.info("🎯 Starting ML model training...")
    from training.main import main as train_main
    
    train_main()

def run_strategic_training():
    """Run strategic model training."""
    logger.info("🎯 Starting strategic model training...")
    from training.train_strategic_model import StrategicModelTrainer
    
    trainer = StrategicModelTrainer()
    trainer.run_training(days_back=30)

def run_rl_training():
    """Run RL model training."""
    logger.info("🎯 Starting RL model training...")
    from models.rl.training.train_rl import main as rl_train_main
    
    rl_train_main()

def run_rl_testing():
    """Run RL model testing."""
    logger.info("🧪 Starting RL model testing...")
    from models.rl.testing.test_rl import main as rl_test_main
    
    rl_test_main()

def run_comparison():
    """Run ML vs RL comparison."""
    logger.info("📊 Starting ML vs RL comparison...")
    from scripts.compare_ml_vs_rl import main as compare_main
    
    compare_main()

def run_tests():
    """Run system tests."""
    logger.info("🧪 Starting system tests...")
    from scripts.run_tests import main as tests_main
    
    tests_main()

def run_monitoring():
    """Run system monitoring."""
    logger.info("📈 Starting system monitoring...")
    from scripts.monitoring import main as monitoring_main
    
    monitoring_main()

def run_system_status():
    """Run system status check."""
    logger.info("🔍 Starting system status check...")
    from scripts.system_status import main as status_main
    
    status_main()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading System - Main Entry Point')
    parser.add_argument('command', choices=[
        'live-trading',
        'training',
        'strategic-training',
        'rl-training',
        'rl-testing',
        'comparison',
        'tests',
        'monitoring',
        'status',
        'help'
    ], help='Command to execute')
    
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level=args.log_level)
    
    logger.info("🎯 TRADING SYSTEM - MAIN ENTRY POINT")
    logger.info(f"📋 Command: {args.command}")
    logger.info(f"📁 Config: {args.config}")
    logger.info(f"🔧 Log Level: {args.log_level}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("✅ Configuration loaded successfully")
        
        # Execute command
        if args.command == 'live-trading':
            run_live_trading()
        elif args.command == 'training':
            run_training()
        elif args.command == 'strategic-training':
            run_strategic_training()
        elif args.command == 'rl-training':
            run_rl_training()
        elif args.command == 'rl-testing':
            run_rl_testing()
        elif args.command == 'comparison':
            run_comparison()
        elif args.command == 'tests':
            run_tests()
        elif args.command == 'monitoring':
            run_monitoring()
        elif args.command == 'status':
            run_system_status()
        elif args.command == 'help':
            parser.print_help()
        
        logger.info("✅ Command completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 