import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy import stats
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analysis.financial_metrics import FinancialMetrics
from utils.data_fetcher import StockDataFetcher


class StatsVisualizer:
    """Visualization class for financial statistics."""
    
    def __init__(self):
        """Initialize the visualizer with required components."""
        self.metrics = FinancialMetrics()
        self.fetcher = StockDataFetcher()
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'plots')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_returns_distribution(self, symbol: str, period: str = "2y") -> str:
        """
        Create distribution plot of returns with skewness and kurtosis annotations.
        
        Args:
            symbol: Stock symbol
            period: Time period for data
            
        Returns:
            Path to saved plot
        """
        try:
            # Fetch data
            df = self.fetcher.get_stock_data(symbol, period=period)
            returns = self.metrics.calculate_returns(df['Close'])
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot returns distribution
            sns.histplot(returns, kde=True, stat='density', alpha=0.7)
            
            # Add normal distribution for comparison
            x = np.linspace(returns.min(), returns.max(), 100)
            plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
                    'r--', label='Normal Distribution', linewidth=2)
            
            # Calculate statistics
            skew = self.metrics.calculate_skewness(returns)
            kurt = self.metrics.calculate_kurtosis(returns)
            
            # Add annotations
            plt.title(f'Returns Distribution for {symbol}\nSkewness: {skew:.3f}, Kurtosis: {kurt:.3f}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Log Returns', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            save_path = os.path.join(self.output_dir, f'returns_dist_{symbol}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"Error creating returns distribution plot for {symbol}: {str(e)}")
            return None
    
    def plot_volatility_trend(self, symbol: str, period: str = "2y", window: int = 30) -> str:
        """
        Create rolling volatility plot.
        
        Args:
            symbol: Stock symbol
            period: Time period for data
            window: Rolling window size in days
            
        Returns:
            Path to saved plot
        """
        try:
            # Fetch data
            df = self.fetcher.get_stock_data(symbol, period=period)
            returns = self.metrics.calculate_returns(df['Close'])
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(rolling_vol.index, rolling_vol, 
                    label=f'{window}-Day Rolling Volatility', linewidth=2)
            
            # Add overall HV line
            hv = self.metrics.calculate_historical_volatility(returns)
            plt.axhline(y=hv, color='r', linestyle='--', linewidth=2,
                       label=f'Historical Volatility: {hv:.2f}%')
            
            plt.title(f'Historical Volatility Analysis - {symbol}', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Annualized Volatility (%)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            save_path = os.path.join(self.output_dir, f'volatility_{symbol}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"Error creating volatility plot for {symbol}: {str(e)}")
            return None
    
    def plot_comparative_stats(self, symbols: List[str], period: str = "2y") -> str:
        """
        Create comparative bar plots for multiple stocks' statistics.
        
        Args:
            symbols: List of stock symbols
            period: Time period for data
            
        Returns:
            Path to saved plot
        """
        try:
            # Collect statistics for all symbols
            stats_data = {}
            for symbol in symbols:
                df = self.fetcher.get_stock_data(symbol, period=period)
                stats_data[symbol] = self.metrics.analyze_stock(df)
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Extract metrics
            symbols_list = list(stats_data.keys())
            hv = [stats_data[s]['historical_volatility'] for s in symbols_list]
            skew = [stats_data[s]['skewness'] for s in symbols_list]
            kurt = [stats_data[s]['kurtosis'] for s in symbols_list]
            
            # Plot each metric
            ax1.bar(symbols_list, hv, alpha=0.7)
            ax1.set_title('Historical Volatility (%)', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(symbols_list, skew, alpha=0.7)
            ax2.set_title('Skewness', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            ax3.bar(symbols_list, kurt, alpha=0.7)
            ax3.set_title('Kurtosis', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.output_dir, 'comparative_stats.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"Error creating comparative plot: {str(e)}")
            return None