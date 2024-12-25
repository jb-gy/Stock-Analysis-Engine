import pandas as pd
from data_fetcher import StockDataFetcher

def validate_stock_data(symbol: str = "AAPL", sample_size: int = 5) -> None:
    """
    Validate and display a sample of stock data to verify format and content.
    
    Args:
        symbol: Stock symbol to validate (default: AAPL)
        sample_size: Number of rows to display (default: 5)
    """
    try:
        # Initialize data fetcher
        fetcher = StockDataFetcher()
        
        # Get data - using 2 years of data for robust statistical calculations
        df = fetcher.get_stock_data(symbol, period="2y")
        
        # Display basic information
        print(f"\nDataset Info for {symbol}:")
        print("-" * 50)
        print(f"Total records: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nColumns available: {', '.join(df.columns)}")
        print("\nData Types:")
        print(df.dtypes)
        
        # Display sample
        print(f"\nFirst {sample_size} rows of data:")
        print("-" * 50)
        print(df.head(sample_size))
        
        # Basic data quality checks
        print("\nData Quality Checks:")
        print("-" * 50)
        print(f"Missing values:\n{df.isnull().sum()}")
        
        return df
        
    except Exception as e:
        print(f"Error validating data: {str(e)}")
        return None

if __name__ == "__main__":
    # Test with Apple stock
    validate_stock_data("AAPL")
