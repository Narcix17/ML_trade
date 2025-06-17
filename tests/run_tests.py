#!/usr/bin/env python3
"""
Comprehensive test runner for the trading system.
"""

import unittest
import sys
import os
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests(verbose=False, pattern=None):
    """
    Run all tests in the test suite.
    
    Args:
        verbose: Enable verbose output
        pattern: Pattern to match test files
    
    Returns:
        TestResult object
    """
    # Discover tests
    test_loader = unittest.TestLoader()
    
    if pattern:
        test_loader.testNamePatterns = [pattern]
    
    # Find all test files
    test_dir = Path(__file__).parent
    test_suite = test_loader.discover(
        start_dir=str(test_dir),
        pattern='test_*.py',
        top_level_dir=str(project_root)
    )
    
    # Run tests
    test_runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout
    )
    
    print("🧪 Running Trading System Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    result = test_runner.run(test_suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Time: {end_time - start_time:.2f} seconds")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    return result

def run_specific_test(test_name, verbose=False):
    """
    Run a specific test.
    
    Args:
        test_name: Name of the test to run
        verbose: Enable verbose output
    
    Returns:
        TestResult object
    """
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromName(test_name)
    
    test_runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout
    )
    
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 50)
    
    start_time = time.time()
    result = test_runner.run(test_suite)
    end_time = time.time()
    
    print(f"\n⏱️  Time: {end_time - start_time:.2f} seconds")
    
    return result

def run_coverage_tests():
    """
    Run tests with coverage reporting.
    """
    try:
        import coverage
        
        # Start coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        result = run_all_tests(verbose=True)
        
        # Stop coverage
        cov.stop()
        cov.save()
        
        # Generate report
        print("\n📊 Coverage Report")
        print("=" * 50)
        cov.report()
        
        # Generate HTML report
        cov.html_report(directory='htmlcov')
        print(f"\n📁 HTML coverage report generated in: htmlcov/")
        
        return result
        
    except ImportError:
        print("❌ Coverage not installed. Install with: pip install coverage")
        return run_all_tests(verbose=True)

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run trading system tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true', help='Run with coverage')
    parser.add_argument('--test', '-t', help='Run specific test')
    parser.add_argument('--pattern', '-p', help='Pattern to match test files')
    
    args = parser.parse_args()
    
    try:
        if args.coverage:
            result = run_coverage_tests()
        elif args.test:
            result = run_specific_test(args.test, args.verbose)
        else:
            result = run_all_tests(args.verbose, args.pattern)
        
        # Exit with appropriate code
        if result.failures or result.errors:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 