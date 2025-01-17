from utils.data_fetcher import StockDataFetcher
from financial_metrics import FinancialMetrics
import pandas as pd

def test_financial_metrics(symbols: list = ["AAPL", "MSFT", "GOOGL"]):
    """
    TODO
    Test financial metrics calculation for multiple stocks.
    
    Args:
        symbols: List of stock symbols to test
    """
    # Initialize our classes
    fetcher = StockDataFetcher()
    metrics = FinancialMetrics()
    
    print("\nFinancial Metrics Test Results")
    print("=" * 50)
    
    for symbol in symbols:
        try:
            print(f"\nAnalyzing {symbol}:")
            print("-" * 30)
            
            # Fetch 2 years of data
            df = fetcher.get_stock_data(symbol, period="2y")
            
            # Calculate metrics
            stats = metrics.analyze_stock(df)
            
            # Display results
            print(f"Historical Volatility: {stats['historical_volatility']:.2f}%")
            print(f"Skewness: {stats['skewness']:.3f}")
            print(f"Kurtosis: {stats['kurtosis']:.3f}")
            print(f"Annualized Mean Return: {stats['mean_return']:.2f}%")
            print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")

if __name__ == "__main__":
    test_financial_metrics()
