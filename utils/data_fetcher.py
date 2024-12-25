import yfinance as yf
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

class StockDataFetcher:
    """Utility class to fetch and process stock data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.cache = {}  # Simple cache to store fetched data
        
    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for the given symbol.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data fetching (optional)
            end_date: End date for data fetching (optional)
            period: Period to fetch if start/end dates not provided (default: '1y')
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        try:
            # Create a unique key for caching
            cache_key = f"{symbol}_{start_date}_{end_date}_{period}"
            
            # Return cached data if available
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Initialize ticker
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period=period)
            
            # Basic validation
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Cache the result
            self.cache[cache_key] = df
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
            
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest closing price for a symbol."""
        df = self.get_stock_data(symbol, period="1d")
        return df['Close'].iloc[-1]
