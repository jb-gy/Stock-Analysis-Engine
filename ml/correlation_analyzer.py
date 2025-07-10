import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import coint
import warnings
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_fetcher import StockDataFetcher
from analysis.financial_metrics import FinancialMetrics


class CorrelationAnalyzer:
    """Advanced correlation and regression analysis for multiple stocks."""
    
    def __init__(self):
        """Initialize the correlation analyzer."""
        self.fetcher = StockDataFetcher()
        self.metrics = FinancialMetrics()
        self.scaler = StandardScaler()
        
        # Create output directory for ML plots
        self.output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'ml_plots')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fetch_aligned_data(self, symbols: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch and align stock data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            period: Time period for data
            
        Returns:
            DataFrame with aligned closing prices
        """
        aligned_data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                df = self.fetcher.get_stock_data(symbol, period=period)
                aligned_data[symbol] = df['Close']
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
                continue
        
        # Drop rows with any NaN values to ensure alignment
        aligned_data = aligned_data.dropna()
        
        if aligned_data.empty:
            raise ValueError("No aligned data available for the given symbols")
        
        return aligned_data
    
    def calculate_returns_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns for all stocks."""
        returns = pd.DataFrame()
        
        for column in price_data.columns:
            returns[column] = self.metrics.calculate_returns(price_data[column])
        
        return returns.dropna()
    
    def correlation_analysis(self, symbols: List[str], period: str = "2y") -> Dict:
        """
        Comprehensive correlation analysis between stocks.
        
        Args:
            symbols: List of stock symbols
            period: Time period for analysis
            
        Returns:
            Dictionary containing correlation results
        """
        print(f"🔍 Performing correlation analysis for {len(symbols)} stocks...")
        
        # Fetch aligned data
        price_data = self.fetch_aligned_data(symbols, period)
        returns_data = self.calculate_returns_matrix(price_data)
        
        # Calculate correlation matrices
        price_corr = price_data.corr()
        returns_corr = returns_data.corr()
        
        # Calculate rolling correlation (30-day window)
        rolling_corr = {}
        if len(symbols) >= 2:
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    if symbol1 in returns_data.columns and symbol2 in returns_data.columns:
                        rolling_corr[f"{symbol1}_{symbol2}"] = returns_data[symbol1].rolling(30).corr(returns_data[symbol2])
        
        results = {
            'price_correlation': price_corr,
            'returns_correlation': returns_corr,
            'rolling_correlations': rolling_corr,
            'price_data': price_data,
            'returns_data': returns_data
        }
        
        return results
    
    def regression_analysis(self, symbols: List[str], period: str = "2y") -> Dict:
        """
        Perform regression analysis between stocks (pairwise).
        
        Args:
            symbols: List of stock symbols
            period: Time period for analysis
            
        Returns:
            Dictionary containing regression results
        """
        print(f"📈 Performing regression analysis for {len(symbols)} stocks...")
        
        # Get correlation analysis results
        corr_results = self.correlation_analysis(symbols, period)
        returns_data = corr_results['returns_data']
        
        regression_results = {}
        
        # Perform pairwise regression analysis
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 in returns_data.columns and symbol2 in returns_data.columns:
                    try:
                        # Prepare data
                        X = returns_data[symbol1].values.reshape(-1, 1)
                        y = returns_data[symbol2].values
                        
                        # Remove NaN values
                        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                        X_clean = X[mask]
                        y_clean = y[mask]
                        
                        if len(X_clean) < 10:  # Need minimum data points
                            continue
                        
                        # Fit regression model
                        model = LinearRegression()
                        model.fit(X_clean, y_clean)
                        
                        # Make predictions
                        y_pred = model.predict(X_clean)
                        
                        # Calculate metrics
                        r2 = r2_score(y_clean, y_pred)
                        pearson_corr, pearson_p = pearsonr(X_clean.flatten(), y_clean)
                        
                        # Beta calculation (slope)
                        beta = model.coef_[0]
                        alpha = model.intercept_
                        
                        # Statistical significance
                        n = len(X_clean)
                        significance = "Significant" if pearson_p < 0.05 else "Not Significant"
                        
                        regression_results[f"{symbol1}_vs_{symbol2}"] = {
                            'beta': beta,
                            'alpha': alpha,
                            'r_squared': r2,
                            'correlation': pearson_corr,
                            'p_value': pearson_p,
                            'significance': significance,
                            'sample_size': n,
                            'model': model,
                            'X_data': X_clean,
                            'y_data': y_clean,
                            'predictions': y_pred
                        }
                        
                    except Exception as e:
                        print(f"Warning: Regression failed for {symbol1} vs {symbol2}: {e}")
                        continue
        
        return {**corr_results, 'regression_results': regression_results}
    
    def cointegration_test(self, symbols: List[str], period: str = "2y") -> Dict:
        """
        Test for cointegration between stock pairs.
        
        Args:
            symbols: List of stock symbols
            period: Time period for analysis
            
        Returns:
            Dictionary containing cointegration test results
        """
        print(f"🔬 Testing cointegration for {len(symbols)} stocks...")
        
        price_data = self.fetch_aligned_data(symbols, period)
        cointegration_results = {}
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                if symbol1 in price_data.columns and symbol2 in price_data.columns:
                    try:
                        # Perform Engle-Granger cointegration test
                        score, p_value, _ = coint(price_data[symbol1], price_data[symbol2])
                        
                        is_cointegrated = p_value < 0.05
                        relationship = "Cointegrated" if is_cointegrated else "Not Cointegrated"
                        
                        cointegration_results[f"{symbol1}_{symbol2}"] = {
                            'test_statistic': score,
                            'p_value': p_value,
                            'is_cointegrated': is_cointegrated,
                            'relationship': relationship
                        }
                        
                    except Exception as e:
                        print(f"Warning: Cointegration test failed for {symbol1}-{symbol2}: {e}")
                        continue
        
        return cointegration_results
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, title: str, symbols: List[str]) -> str:
        """Create correlation heatmap visualization."""
        plt.figure(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title(f'{title}\n({len(symbols)} Stocks)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = title.lower().replace(' ', '_').replace('\n', '_')
        save_path = os.path.join(self.output_dir, f'{filename}_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_regression_analysis(self, regression_results: Dict, symbols: List[str]) -> str:
        """Create regression analysis visualization."""
        n_pairs = len(regression_results)
        if n_pairs == 0:
            return None
        
        # Calculate subplot dimensions
        cols = min(3, n_pairs)
        rows = (n_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_pairs == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_pairs > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (pair_name, results) in enumerate(regression_results.items()):
            ax = axes[idx] if n_pairs > 1 else axes[0]
            
            # Extract data
            X = results['X_data'].flatten()
            y = results['y_data']
            y_pred = results['predictions']
            
            # Scatter plot
            ax.scatter(X, y, alpha=0.6, s=20)
            ax.plot(X, y_pred, 'r-', linewidth=2)
            
            # Labels and title
            symbol1, symbol2 = pair_name.split('_vs_')
            ax.set_xlabel(f'{symbol1} Returns')
            ax.set_ylabel(f'{symbol2} Returns')
            ax.set_title(f'{symbol1} vs {symbol2}\n'
                        f'β: {results["beta"]:.3f}, R²: {results["r_squared"]:.3f}\n'
                        f'Corr: {results["correlation"]:.3f} ({results["significance"]})')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, 'regression_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_rolling_correlations(self, rolling_correlations: Dict, symbols: List[str]) -> str:
        """Plot rolling correlation over time."""
        if not rolling_correlations:
            return None
        
        plt.figure(figsize=(14, 8))
        
        for pair_name, rolling_corr in rolling_correlations.items():
            symbol1, symbol2 = pair_name.split('_')
            plt.plot(rolling_corr.index, rolling_corr.values, 
                    label=f'{symbol1}-{symbol2}', linewidth=2, alpha=0.8)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='High Correlation')
        plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='High Negative Correlation')
        
        plt.title('Rolling 30-Day Correlations Between Stocks', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Correlation Coefficient')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, 'rolling_correlations.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def comprehensive_analysis(self, symbols: List[str], period: str = "2y") -> Dict:
        """
        Perform comprehensive correlation and regression analysis.
        
        Args:
            symbols: List of stock symbols
            period: Time period for analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n🤖 MACHINE LEARNING ANALYSIS")
        print(f"{'='*60}")
        
        # Perform all analyses
        regression_results = self.regression_analysis(symbols, period)
        cointegration_results = self.cointegration_test(symbols, period)
        
        # Create visualizations
        print(f"\n📊 Creating ML visualizations...")
        
        # Correlation heatmaps
        price_heatmap = self.plot_correlation_heatmap(
            regression_results['price_correlation'], 
            'Price Correlation Matrix', 
            symbols
        )
        
        returns_heatmap = self.plot_correlation_heatmap(
            regression_results['returns_correlation'], 
            'Returns Correlation Matrix', 
            symbols
        )
        
        # Regression plots
        regression_plot = self.plot_regression_analysis(
            regression_results['regression_results'], 
            symbols
        )
        
        # Rolling correlation plots
        rolling_plot = self.plot_rolling_correlations(
            regression_results['rolling_correlations'], 
            symbols
        )
        
        # Compile all results
        complete_results = {
            **regression_results,
            'cointegration_results': cointegration_results,
            'visualizations': {
                'price_correlation_heatmap': price_heatmap,
                'returns_correlation_heatmap': returns_heatmap,
                'regression_analysis_plot': regression_plot,
                'rolling_correlations_plot': rolling_plot
            }
        }
        
        # Print summary
        self.print_analysis_summary(complete_results, symbols)
        
        return complete_results
    
    def print_analysis_summary(self, results: Dict, symbols: List[str]):
        """Print a comprehensive summary of the analysis."""
        print(f"\n🎯 CORRELATION & REGRESSION SUMMARY")
        print(f"{'─'*70}")
        
        # Regression results summary
        if 'regression_results' in results and results['regression_results']:
            print(f"\n📈 REGRESSION ANALYSIS:")
            print(f"{'Pair':<15} {'Beta':<8} {'R²':<8} {'Correlation':<12} {'Significance':<15}")
            print(f"{'─'*70}")
            
            for pair_name, reg_results in results['regression_results'].items():
                symbol1, symbol2 = pair_name.split('_vs_')
                pair_display = f"{symbol1}-{symbol2}"
                print(f"{pair_display:<15} {reg_results['beta']:<8.3f} "
                      f"{reg_results['r_squared']:<8.3f} {reg_results['correlation']:<12.3f} "
                      f"{reg_results['significance']:<15}")
        
        # Cointegration results
        if 'cointegration_results' in results and results['cointegration_results']:
            print(f"\n🔬 COINTEGRATION ANALYSIS:")
            print(f"{'Pair':<15} {'Test Stat':<12} {'P-Value':<10} {'Relationship':<15}")
            print(f"{'─'*60}")
            
            for pair_name, coint_results in results['cointegration_results'].items():
                symbol1, symbol2 = pair_name.split('_')
                pair_display = f"{symbol1}-{symbol2}"
                print(f"{pair_display:<15} {coint_results['test_statistic']:<12.3f} "
                      f"{coint_results['p_value']:<10.4f} {coint_results['relationship']:<15}")
        
        # Overall correlation insights
        if 'returns_correlation' in results:
            corr_matrix = results['returns_correlation']
            # Get upper triangle correlations (excluding diagonal)
            upper_tri = np.triu(corr_matrix.values, k=1)
            correlations = upper_tri[upper_tri != 0]
            
            if len(correlations) > 0:
                print(f"\n🔍 CORRELATION INSIGHTS:")
                print(f"Average correlation: {np.mean(correlations):.3f}")
                print(f"Highest correlation: {np.max(correlations):.3f}")
                print(f"Lowest correlation: {np.min(correlations):.3f}")
                print(f"High correlations (>0.7): {np.sum(correlations > 0.7)} pairs")
                print(f"Negative correlations (<0): {np.sum(correlations < 0)} pairs")