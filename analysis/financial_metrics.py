import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


class FinancialMetrics:
    """Calculate various financial statistics from stock data."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns from price series."""
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def calculate_historical_volatility(returns: pd.Series, window: int = 252) -> float:
        """
        Calculate historical volatility (annualized).
        
        Args:
            returns: Series of log returns
            window: Number of trading days to use (default: 252 for annual)
            
        Returns:
            Annualized volatility as a percentage
        """
        return returns.std() * np.sqrt(window) * 100
    
    @staticmethod
    def calculate_skewness(returns: pd.Series) -> float:
        """Calculate skewness of returns distribution."""
        return stats.skew(returns)
    
    @staticmethod
    def calculate_kurtosis(returns: pd.Series) -> float:
        """Calculate excess kurtosis of returns distribution."""
        return stats.kurtosis(returns)
    
    def analyze_stock(self, df: pd.DataFrame, window: int = 252) -> Dict[str, float]:
        """
        Analyze stock data and return key statistics.
        
        Args:
            df: DataFrame with stock data (must have 'Close' column)
            window: Window size for volatility calculation
            
        Returns:
            Dictionary containing calculated statistics
        """
        # Calculate log returns
        returns = self.calculate_returns(df['Close'])
        
        # Calculate statistics
        stats_dict = {
            'historical_volatility': self.calculate_historical_volatility(returns, window),
            'skewness': self.calculate_skewness(returns),
            'kurtosis': self.calculate_kurtosis(returns),
            'mean_return': returns.mean() * window * 100,  # Annualized mean return
            'std_return': returns.std() * np.sqrt(window) * 100,  # Annualized std of returns
            'sharpe_ratio': (returns.mean() * window) / (returns.std() * np.sqrt(window))  # Assuming risk-free rate = 0
        }
        
        return stats_dict