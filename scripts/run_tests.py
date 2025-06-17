#!/usr/bin/env python3
"""
Unified test runner for the trading system.
Uses shared utilities to eliminate duplicate code.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.testing import TestRunner, run_quick_test
from utils.logging import setup_logger, get_logger
from utils.config import load_config
import argparse

logger = get_logger(__name__)

def main():
    """Main test runner function."""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run trading system tests')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--save-results', action='store_true', help='Save test results to file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level=args.log_level)
    
    logger.info("🧪 STARTING TRADING SYSTEM TESTS")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"📋 Configuration loaded from: {args.config}")
        
        if args.quick:
            # Run quick test
            logger.info("⚡ Running quick test...")
            success = run_quick_test()
            
            if success:
                logger.success("✅ Quick test completed successfully!")
                return 0
            else:
                logger.error("❌ Quick test failed!")
                return 1
        else:
            # Run all tests
            logger.info("🔍 Running comprehensive test suite...")
            
            runner = TestRunner(config)
            results = runner.run_all_tests()
            
            # Save results if requested
            if args.save_results:
                output_file = runner.save_test_results()
                logger.info(f"📁 Test results saved to: {output_file}")
            
            # Check overall success
            summary = results.get('summary', {})
            success_rate = summary.get('success_rate', 0)
            
            if success_rate == 100:
                logger.success("🎉 All tests passed!")
                return 0
            else:
                logger.warning(f"⚠️ {summary.get('failed_tests', 0)} tests failed")
                return 1
                
    except Exception as e:
        logger.error(f"❌ Test runner failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 