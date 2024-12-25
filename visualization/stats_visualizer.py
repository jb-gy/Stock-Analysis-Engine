import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from analysis.financial_metrics import FinancialMetrics
from utils.data_fetcher import StockDataFetcher

class StatsVisualizer:
    """Visualization class for financial statistics."""
    
    def __init__(self):
        """Initialize the visualizer with required components."""
        self.metrics = FinancialMetrics()
        self.fetcher = StockDataFetcher()
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_returns_distribution(self, symbol: str, period: str = "2y") -> str:
        """
        Create distribution plot of returns with skewness and kurtosis annotations.
        
        Args:
            symbol: Stock symbol
            period: Time period for data
            
        Returns:
            Path to saved plot
        """
        # Fetch data
        df = self.fetcher.get_stock_data(symbol, period=period)
        returns = self.metrics.calculate_returns(df['Close'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot returns distribution
        sns.histplot(returns, kde=True, stat='density')
        
        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
                'r--', label='Normal Distribution')
        
        # Calculate statistics
        skew = self.metrics.calculate_skewness(returns)
        kurt = self.metrics.calculate_kurtosis(returns)
        
        # Add annotations
        plt.title(f'Returns Distribution for {symbol}\nSkewness: {skew:.3f}, Kurtosis: {kurt:.3f}')
        plt.xlabel('Log Returns')
        plt.ylabel('Density')
        plt.legend()
        
        # Save plot
        save_path = f'/home/ubuntu/stock-analysis-engine/output/plots/returns_dist_{symbol}.png'
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
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
        # Fetch data
        df = self.fetcher.get_stock_data(symbol, period=period)
        returns = self.metrics.calculate_returns(df['Close'])
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_vol.index, rolling_vol, label=f'{window}-Day Rolling Volatility')
        
        # Add overall HV line
        hv = self.metrics.calculate_historical_volatility(returns)
        plt.axhline(y=hv, color='r', linestyle='--', 
                   label=f'Historical Volatility: {hv:.2f}%')
        
        plt.title(f'Historical Volatility Analysis - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility (%)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        save_path = f'/home/ubuntu/stock-analysis-engine/output/plots/volatility_{symbol}.png'
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    
    def plot_comparative_stats(self, symbols: List[str], period: str = "2y") -> str:
        """
        Create comparative bar plots for multiple stocks' statistics.
        
        Args:
            symbols: List of stock symbols
            period: Time period for data
            
        Returns:
            Path to saved plot
        """
        # Collect statistics for all symbols
        stats = {}
        for symbol in symbols:
            df = self.fetcher.get_stock_data(symbol, period=period)
            stats[symbol] = self.metrics.analyze_stock(df)
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Extract metrics
        symbols_list = list(stats.keys())
        hv = [stats[s]['historical_volatility'] for s in symbols_list]
        skew = [stats[s]['skewness'] for s in symbols_list]
        kurt = [stats[s]['kurtosis'] for s in symbols_list]
        
        # Plot each metric
        ax1.bar(symbols_list, hv)
        ax1.set_title('Historical Volatility (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(symbols_list, skew)
        ax2.set_title('Skewness')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.bar(symbols_list, kurt)
        ax3.set_title('Kurtosis')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = f'/home/ubuntu/stock-analysis-engine/output/plots/comparative_stats.png'
        plt.savefig(save_path)
        plt.close()
        
        return save_path
