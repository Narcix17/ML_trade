"""
Performance profiler and optimizer for the trading system.
"""

import time
import cProfile
import pstats
import io
import functools
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import psutil
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for trading system components."""
    
    def __init__(self, output_dir: str = "reports/performance"):
        """
        Initialize the profiler.
        
        Args:
            output_dir: Directory to save performance reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiles = {}
        self.metrics = {}
        self.start_times = {}
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile a function.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise e
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                profiler.disable()
                
                # Save profile stats
                stats = pstats.Stats(profiler)
                stats_file = self.output_dir / f"{func.__name__}_profile.txt"
                stats.dump_stats(str(stats_file))
                
                # Record metrics
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                self.metrics[func.__name__] = {
                    'execution_time': execution_time,
                    'memory_used': memory_used,
                    'success': success,
                    'calls': stats.total_calls,
                    'profile_file': str(stats_file)
                }
                
                logger.debug(f"Profiled {func.__name__}: {execution_time:.4f}s, {memory_used:.2f}MB")
            
            return result
        
        return wrapper
    
    def start_timer(self, name: str):
        """Start a timer for a named operation."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        End a timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        del self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = {}
        
        self.metrics[name]['execution_time'] = elapsed
        logger.debug(f"Timer {name}: {elapsed:.4f}s")
        
        return elapsed
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report.
        
        Returns:
            Performance report dictionary
        """
        report = {
            'summary': {
                'total_functions': len(self.metrics),
                'total_execution_time': sum(m.get('execution_time', 0) for m in self.metrics.values()),
                'total_memory_used': sum(m.get('memory_used', 0) for m in self.metrics.values()),
                'successful_calls': sum(1 for m in self.metrics.values() if m.get('success', True))
            },
            'functions': self.metrics,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Find slow functions
        slow_threshold = 1.0  # seconds
        slow_functions = [
            name for name, metrics in self.metrics.items()
            if metrics.get('execution_time', 0) > slow_threshold
        ]
        
        if slow_functions:
            recommendations.append(f"Consider optimizing slow functions: {', '.join(slow_functions)}")
        
        # Find memory-intensive functions
        memory_threshold = 100  # MB
        memory_functions = [
            name for name, metrics in self.metrics.items()
            if metrics.get('memory_used', 0) > memory_threshold
        ]
        
        if memory_functions:
            recommendations.append(f"Consider memory optimization for: {', '.join(memory_functions)}")
        
        # Find frequently called functions
        call_threshold = 1000
        frequent_functions = [
            name for name, metrics in self.metrics.items()
            if metrics.get('calls', 0) > call_threshold
        ]
        
        if frequent_functions:
            recommendations.append(f"Consider caching for frequently called functions: {', '.join(frequent_functions)}")
        
        return recommendations
    
    def save_report(self, filename: str = None):
        """
        Save performance report to file.
        
        Args:
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        import json
        with open(report_path, 'w') as f:
            json.dump(self.get_performance_report(), f, indent=2)
        
        logger.info(f"Performance report saved to: {report_path}")


class DataOptimizer:
    """Data optimization utilities."""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"DataFrame optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB ({reduction:.1f}% reduction)")
        
        return df
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
        """
        Split DataFrame into chunks for processing.
        
        Args:
            df: Input DataFrame
            chunk_size: Size of each chunk
            
        Returns:
            List of DataFrame chunks
        """
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    @staticmethod
    def parallel_apply(df: pd.DataFrame, func: Callable, n_jobs: int = -1) -> pd.DataFrame:
        """
        Apply function to DataFrame in parallel.
        
        Args:
            df: Input DataFrame
            func: Function to apply
            n_jobs: Number of jobs (-1 for all cores)
            
        Returns:
            DataFrame with applied function
        """
        try:
            from joblib import Parallel, delayed
            
            chunks = DataOptimizer.chunk_dataframe(df)
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(chunk) for chunk in chunks
            )
            
            return pd.concat(results, ignore_index=True)
            
        except ImportError:
            logger.warning("joblib not available, falling back to sequential processing")
            return func(df)


class CacheManager:
    """Simple caching manager for expensive operations."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
    
    def get(self, key: str) -> Any:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_size:
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 0
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': sum(self.access_count.values()) / max(len(self.cache), 1),
            'keys': list(self.cache.keys())
        }


# Global profiler instance
profiler = PerformanceProfiler()

# Global cache manager
cache_manager = CacheManager()


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function using the global profiler.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    return profiler.profile_function(func)


def optimize_performance():
    """
    Run performance optimization analysis.
    """
    logger.info("🔧 Starting performance optimization analysis...")
    
    # Get system information
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
    
    logger.info(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    
    # Generate and save performance report
    report = profiler.get_performance_report()
    profiler.save_report()
    
    # Print summary
    summary = report['summary']
    logger.info(f"Performance Summary:")
    logger.info(f"  - Functions profiled: {summary['total_functions']}")
    logger.info(f"  - Total execution time: {summary['total_execution_time']:.2f}s")
    logger.info(f"  - Total memory used: {summary['total_memory_used']:.2f}MB")
    logger.info(f"  - Successful calls: {summary['successful_calls']}")
    
    # Print recommendations
    if report['recommendations']:
        logger.info("Recommendations:")
        for rec in report['recommendations']:
            logger.info(f"  - {rec}")
    
    return report


if __name__ == '__main__':
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    @profile_function
    def example_function():
        """Example function to profile."""
        time.sleep(0.1)
        return "done"
    
    # Run example
    for _ in range(5):
        example_function()
    
    # Generate report
    optimize_performance() 