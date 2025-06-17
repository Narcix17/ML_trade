#!/usr/bin/env python3
"""
Shared visualization utilities for RL results.
Eliminates duplicate plotting code between training and testing scripts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime

def plot_trading_results(
    balance_history: List[float],
    pnl_history: List[float],
    action_history: List[int],
    price_history: List[float],
    metrics: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive trading results visualization.
    
    Args:
        balance_history: Account balance over time
        pnl_history: PnL over time
        action_history: Actions taken over time
        price_history: Price over time
        metrics: Trading metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('RL Trading Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Balance curve
    axes[0].plot(balance_history, label='Account Balance', color='blue', linewidth=2)
    axes[0].axhline(y=balance_history[0], color='red', linestyle='--', alpha=0.7, label='Initial Balance')
    axes[0].set_title('Account Balance Over Time')
    axes[0].set_ylabel('Balance ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. PnL curve
    axes[1].plot(pnl_history, label='PnL', color='green', linewidth=2)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('Profit and Loss Over Time')
    axes[1].set_ylabel('PnL ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Price and actions
    axes[2].plot(price_history, label='Price', color='purple', linewidth=1, alpha=0.7)
    axes[2].set_title('Price Movement')
    axes[2].set_ylabel('Price')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add action markers
    if action_history:
        buy_indices = [i for i, a in enumerate(action_history) if a == 1]
        sell_indices = [i for i, a in enumerate(action_history) if a == 2]
        
        if buy_indices:
            axes[2].scatter(buy_indices, [price_history[i] for i in buy_indices], 
                          color='green', marker='^', s=50, label='Buy', alpha=0.7)
        if sell_indices:
            axes[2].scatter(sell_indices, [price_history[i] for i in sell_indices], 
                          color='red', marker='v', s=50, label='Sell', alpha=0.7)
        axes[2].legend()
    
    # 4. Action distribution
    if action_history:
        action_counts = np.bincount(action_history, minlength=3)
        action_labels = ['Hold', 'Buy', 'Sell']
        colors = ['gray', 'green', 'red']
        
        axes[3].bar(action_labels, action_counts, color=colors, alpha=0.7)
        axes[3].set_title('Action Distribution')
        axes[3].set_ylabel('Count')
        
        # Add percentage labels
        total_actions = sum(action_counts)
        for i, (label, count) in enumerate(zip(action_labels, action_counts)):
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            axes[3].text(i, count + max(action_counts) * 0.01, 
                        f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_comparison(
    ml_metrics: Dict,
    rl_metrics: Dict,
    ml_balance_history: List[float],
    rl_balance_history: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Create comparison plot between ML and RL results.
    
    Args:
        ml_metrics: ML model metrics
        rl_metrics: RL model metrics
        ml_balance_history: ML balance history
        rl_balance_history: RL balance history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ML vs RL Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Balance comparison
    axes[0, 0].plot(ml_balance_history, label='ML Strategy', color='blue', linewidth=2)
    axes[0, 0].plot(rl_balance_history, label='RL Strategy', color='red', linewidth=2)
    axes[0, 0].axhline(y=ml_balance_history[0], color='gray', linestyle='--', alpha=0.7, label='Initial Balance')
    axes[0, 0].set_title('Account Balance Comparison')
    axes[0, 0].set_ylabel('Balance ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Performance metrics comparison
    metrics_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
    ml_values = [
        ml_metrics.get('total_return', 0) * 100,
        ml_metrics.get('sharpe_ratio', 0),
        abs(ml_metrics.get('max_drawdown', 0)) * 100
    ]
    rl_values = [
        rl_metrics.get('total_return', 0) * 100,
        rl_metrics.get('sharpe_ratio', 0),
        abs(rl_metrics.get('max_drawdown', 0)) * 100
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, ml_values, width, label='ML Strategy', color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, rl_values, width, label='RL Strategy', color='red', alpha=0.7)
    axes[0, 1].set_title('Performance Metrics Comparison')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Action distribution comparison
    ml_actions = ml_metrics.get('action_distribution', {})
    rl_actions = rl_metrics.get('action_distribution', {})
    
    action_types = ['Hold', 'Buy', 'Sell']
    ml_action_values = [
        ml_actions.get('hold', 0) * 100,
        ml_actions.get('buy', 0) * 100,
        ml_actions.get('sell', 0) * 100
    ]
    rl_action_values = [
        rl_actions.get('hold', 0) * 100,
        rl_actions.get('buy', 0) * 100,
        rl_actions.get('sell', 0) * 100
    ]
    
    axes[1, 0].bar(x - width/2, ml_action_values, width, label='ML Strategy', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, rl_action_values, width, label='RL Strategy', color='red', alpha=0.7)
    axes[1, 0].set_title('Action Distribution Comparison')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(action_types)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    summary_data = [
        ['Total Return', f"{ml_metrics.get('total_return', 0)*100:.2f}%", f"{rl_metrics.get('total_return', 0)*100:.2f}%"],
        ['Sharpe Ratio', f"{ml_metrics.get('sharpe_ratio', 0):.2f}", f"{rl_metrics.get('sharpe_ratio', 0):.2f}"],
        ['Max Drawdown', f"{abs(ml_metrics.get('max_drawdown', 0))*100:.2f}%", f"{abs(rl_metrics.get('max_drawdown', 0))*100:.2f}%"],
        ['Win Rate', f"{ml_metrics.get('win_rate', 0)*100:.1f}%", f"{rl_metrics.get('win_rate', 0)*100:.1f}%"],
        ['Total Trades', f"{ml_metrics.get('total_trades', 0)}", f"{rl_metrics.get('total_trades', 0)}"]
    ]
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table = axes[1, 1].table(
        cellText=summary_data,
        colLabels=['Metric', 'ML Strategy', 'RL Strategy'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

def create_simple_comparison_plot(ml_metrics: Dict, rl_metrics: Dict, save_path: Optional[str] = None) -> None:
    """
    Create a simple comparison plot for quick analysis.
    
    Args:
        ml_metrics: ML model metrics
        rl_metrics: RL model metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ML vs RL Quick Comparison', fontsize=14, fontweight='bold')
    
    # Performance metrics
    metrics = ['Total Return', 'Sharpe Ratio']
    ml_values = [
        ml_metrics.get('total_return', 0) * 100,
        ml_metrics.get('sharpe_ratio', 0)
    ]
    rl_values = [
        rl_metrics.get('total_return', 0) * 100,
        rl_metrics.get('sharpe_ratio', 0)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, ml_values, width, label='ML Strategy', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, rl_values, width, label='RL Strategy', color='red', alpha=0.7)
    axes[0].set_title('Performance Metrics')
    axes[0].set_ylabel('Value')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Action distribution
    ml_actions = ml_metrics.get('action_distribution', {})
    rl_actions = rl_metrics.get('action_distribution', {})
    
    action_types = ['Hold', 'Buy', 'Sell']
    ml_action_values = [
        ml_actions.get('hold', 0) * 100,
        ml_actions.get('buy', 0) * 100,
        ml_actions.get('sell', 0) * 100
    ]
    rl_action_values = [
        rl_actions.get('hold', 0) * 100,
        rl_actions.get('buy', 0) * 100,
        rl_actions.get('sell', 0) * 100
    ]
    
    axes[1].bar(x - width/2, ml_action_values, width, label='ML Strategy', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, rl_action_values, width, label='RL Strategy', color='red', alpha=0.7)
    axes[1].set_title('Action Distribution')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(action_types)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Simple comparison plot saved to: {save_path}")
    
    plt.show() 