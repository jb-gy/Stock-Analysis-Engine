# Stock Analysis Engine

A Python-based financial analysis tool that provides visualized statistics for any publicly traded stock using day-end data.

## Features

- **Statistical Analysis**: Calculate and visualize key financial metrics:
  - Historical Volatility (HV)
  - Skewness (measure of return distribution asymmetry)
  - Kurtosis (measure of tail heaviness)
  - Additional metrics: Sharpe Ratio, annualized returns

- **Visualization Suite**:
  - Returns distribution plots with normal distribution comparison
  - Rolling volatility analysis
  - Comparative statistics across multiple stocks

- **Data Handling**:
  - Automated day-end data fetching from Yahoo Finance
  - Efficient data caching for repeated analyses
  - Support for any publicly traded stock symbol

## Design Decisions

### Architecture

The project follows a modular design with three main components:

1. **Data Layer** (`src/utils/`):
   - `data_fetcher.py`: Handles data retrieval and caching
   - `data_validator.py`: Ensures data quality and format

2. **Analysis Layer** (`src/analysis/`):
   - `financial_metrics.py`: Implements statistical calculations
   - Core metrics calculated using scipy and numpy for reliability

3. **Visualization Layer** (`src/visualization/`):
   - `stats_visualizer.py`: Creates statistical visualizations
   - Uses matplotlib and seaborn for professional-quality plots

### Technical Choices

- **Yahoo Finance API**: Chosen for reliable day-end data access and comprehensive market coverage
- **Pandas & NumPy**: Efficient data manipulation and numerical computations
- **SciPy**: Robust statistical calculations
- **Matplotlib & Seaborn**: Professional-quality visualizations

## How It Works

### Data Collection

1. The `StockDataFetcher` class fetches historical data:
   ```python
   fetcher = StockDataFetcher()
   data = fetcher.get_stock_data("AAPL", period="2y")
   ```

### Statistical Analysis

2. The `FinancialMetrics` class calculates key statistics:
   ```python
   metrics = FinancialMetrics()
   stats = metrics.analyze_stock(data)
   ```

   - Historical Volatility: Annualized standard deviation of log returns
   - Skewness: Measures asymmetry in returns distribution
   - Kurtosis: Indicates tail heaviness compared to normal distribution

### Visualization

3. The `StatsVisualizer` class creates insightful plots:
   ```python
   visualizer = StatsVisualizer()
   visualizer.plot_returns_distribution("AAPL")
   visualizer.plot_volatility_trend("AAPL")
   visualizer.plot_comparative_stats(["AAPL", "MSFT", "GOOGL"])
   ```

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-analysis-engine
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run example analysis:
   ```python
   from visualization.stats_visualizer import StatsVisualizer
   
   visualizer = StatsVisualizer()
   visualizer.plot_returns_distribution("AAPL")  # Analyze Apple stock
   ```

## Output

The tool generates three types of visualizations:

1. **Returns Distribution Plot**:
   - Shows the distribution of stock returns
   - Overlays normal distribution for comparison
   - Displays skewness and kurtosis values

2. **Volatility Trend Plot**:
   - Shows rolling historical volatility
   - Includes overall HV reference line
   - Helps identify volatility regimes

3. **Comparative Statistics Plot**:
   - Compares HV, skewness, and kurtosis across stocks
   - Facilitates multi-stock analysis
   - Useful for portfolio considerations

## Dependencies

- pandas
- numpy
- scipy
- matplotlib
- seaborn
- yfinance

## Notes

- Historical data is sourced from Yahoo Finance
- Default analysis period is 2 years for robust statistical calculations
- All visualizations are saved in the `output/plots` directory
