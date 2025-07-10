# 📈 Stock Analysis Engine with Machine Learning

A comprehensive Python-based financial analysis platform that combines traditional statistical analysis with advanced machine learning techniques for in-depth stock market insights.

## 🌟 Features

### 📊 Traditional Financial Analysis
- **Statistical Metrics**: Historical Volatility, Skewness, Kurtosis, Sharpe Ratio
- **Return Analysis**: Logarithmic returns calculation and distribution analysis
- **Visualization Suite**: Professional-quality charts with matplotlib and seaborn
- **Multi-Stock Comparison**: Side-by-side statistical comparison

### 🤖 Machine Learning Capabilities
- **Regression Analysis**: Pairwise regression models with beta coefficients and R² values
- **Correlation Analysis**: Static and rolling correlation matrices with heatmap visualizations
- **Cointegration Testing**: Engle-Granger tests for long-term relationship identification
- **Predictive Modeling**: Foundation for price prediction and portfolio optimization

### 🎨 Advanced Visualizations
- Returns distribution plots with normal distribution overlay
- Rolling volatility trend analysis
- Correlation heatmaps with statistical significance
- Regression scatter plots with trend lines
- Time series correlation evolution

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection (for fetching stock data)

### Installation

1. **Clone or Download the Project**
   ```bash
   git clone <your-repository-url>
   cd stock-analysis-engine
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
stock-analysis-engine/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── main.py                      # Main application entry point
├── __init__.py                  # Package initialization
│
├── analysis/                    # Core financial analysis
│   ├── __init__.py
│   ├── financial_metrics.py    # Statistical calculations
│   └── test_metrics.py         # Testing module
│
├── utils/                       # Data handling utilities
│   ├── __init__.py
│   ├── data_fetcher.py         # Yahoo Finance data retrieval
│   └── data_validator.py       # Data quality validation
│
├── visualization/               # Charting and plotting
│   ├── __init__.py
│   ├── stats_visualizer.py     # Traditional financial charts
│   └── test_visualizer.py      # Visualization testing
│
├── ml/                         # Machine Learning module
│   ├── __init__.py
│   └── correlation_analyzer.py # ML correlation & regression analysis
│
└── output/                     # Generated files
    ├── plots/                  # Traditional analysis charts
    └── ml_plots/              # Machine learning visualizations
```

## 🎯 Usage Guide

### Interactive Menu Options

When you run `python main.py`, you'll see:

```
🚀 Stock Analysis Engine
============================================================

What would you like to do?
1. Analyze a single stock
2. Compare multiple stocks with ML correlation analysis
3. Quick demo with Apple (AAPL)
4. ML Demo: Tech stocks correlation analysis (AAPL, MSFT, GOOGL, TSLA)
```

### Option 1: Single Stock Analysis
- Enter any stock ticker symbol (e.g., AAPL, TSLA, MSFT)
- Choose time period (1y, 2y, 5y)
- Get comprehensive statistical analysis and visualizations

**Sample Output:**
```
📈 FINANCIAL STATISTICS for AAPL:
──────────────────────────────────────────────────
Historical Volatility: 28.45%
Skewness: -0.156
Kurtosis: 4.231
Annualized Mean Return: 15.23%
Sharpe Ratio: 0.535

✅ Returns distribution saved: output/plots/returns_dist_AAPL.png
✅ Volatility trend saved: output/plots/volatility_AAPL.png
```

### Option 2: Multi-Stock ML Analysis
- Compare multiple stocks with advanced ML algorithms
- Get regression analysis, correlation matrices, and cointegration tests
- Includes both traditional and ML-powered insights

**Sample ML Output:**
```
🤖 REGRESSION ANALYSIS:
Pair            Beta     R²       Correlation  Significance   
──────────────────────────────────────────────────────────────
AAPL-MSFT      0.847    0.423    0.651        Significant    
AAPL-GOOGL     0.734    0.356    0.597        Significant    

🔬 COINTEGRATION ANALYSIS:
Pair            Test Stat    P-Value    Relationship   
────────────────────────────────────────────────────
AAPL-MSFT      -3.245       0.0234     Cointegrated   
```

### Option 3: Quick Demo
- Instant analysis of Apple (AAPL) stock
- Perfect for testing the system

### Option 4: ML Demo
- Demonstrates full ML capabilities
- Analyzes tech stock correlations (AAPL, MSFT, GOOGL, TSLA)
- Shows all ML features in action

## 📊 Generated Outputs

### Traditional Analysis (`output/plots/`)
- `returns_dist_{SYMBOL}.png` - Returns distribution with skewness/kurtosis
- `volatility_{SYMBOL}.png` - Rolling volatility analysis
- `comparative_stats.png` - Multi-stock comparison charts

### ML Analysis (`output/ml_plots/`)
- `price_correlation_matrix_heatmap.png` - Price correlation heatmap
- `returns_correlation_matrix_heatmap.png` - Returns correlation heatmap
- `regression_analysis.png` - Pairwise regression plots
- `rolling_correlations.png` - Time series correlation evolution

## 🧮 Statistical Metrics Explained

### Traditional Metrics
- **Historical Volatility**: Annualized standard deviation of returns (risk measure)
- **Skewness**: Asymmetry of return distribution (negative = more left tail risk)
- **Kurtosis**: Tail heaviness compared to normal distribution (higher = more extreme events)
- **Sharpe Ratio**: Risk-adjusted return measure (return per unit of risk)

### ML Metrics
- **Beta Coefficient**: Stock's sensitivity to market movements (β > 1 = more volatile than market)
- **R-Squared**: Percentage of variance explained by the model (0-1 scale)
- **Correlation**: Linear relationship strength between two stocks (-1 to +1)
- **Cointegration**: Long-term equilibrium relationship between stock prices

## 🔧 Technical Implementation

### Data Source
- **Yahoo Finance API** via `yfinance` library
- Real-time and historical stock data
- Reliable day-end pricing information

### Core Libraries
- **pandas & numpy**: Data manipulation and numerical computations
- **scipy**: Advanced statistical calculations
- **scikit-learn**: Machine learning algorithms
- **statsmodels**: Econometric analysis and cointegration testing
- **matplotlib & seaborn**: Professional visualization

### Machine Learning Algorithms
- **Linear Regression**: For beta calculation and relationship modeling
- **Pearson/Spearman Correlation**: For relationship strength measurement
- **Engle-Granger Test**: For cointegration analysis
- **Rolling Window Analysis**: For time-varying correlation patterns

## 🛠 Development and Extension

### Adding New ML Models

1. Create new file in `ml/` directory
2. Import base classes from existing modules
3. Follow the pattern established in `correlation_analyzer.py`

Example structure:
```python
# ml/your_new_model.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_fetcher import StockDataFetcher
from analysis.financial_metrics import FinancialMetrics

class YourNewModel:
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.metrics = FinancialMetrics()
    
    def your_analysis_method(self, symbols, period):
        # Your ML implementation here
        pass
```

### Extending Visualizations

Add new plot methods to `visualization/stats_visualizer.py` or create specialized visualization classes in the `ml/` module.

## 📋 Dependencies

### Core Requirements
```
yfinance>=0.2.0          # Stock data fetching
pandas>=1.5.0            # Data manipulation
numpy>=1.24.0            # Numerical computing
scipy>=1.10.0            # Statistical functions
matplotlib>=3.6.0        # Plotting library
seaborn>=0.12.0          # Statistical visualization
scikit-learn>=1.3.0      # Machine learning
statsmodels>=0.14.0      # Econometric analysis
```

### Optional (for future enhancements)
```
tensorflow>=2.13.0       # Deep learning
torch>=2.0.0             # PyTorch for neural networks
transformers>=4.21.0     # NLP for news analysis
```



## 🚀 Future Enhancements

### Planned ML Features
1. **LSTM Price Prediction**: Neural networks for price forecasting
2. **Portfolio Optimization**: Modern Portfolio Theory implementation
3. **Sentiment Analysis**: News and social media impact analysis
4. **RAG Integration**: Natural language query interface

### Advanced Analytics
1. **Options Pricing Models**: Black-Scholes implementation
2. **Risk Management**: VaR and CVaR calculations
3. **Backtesting Framework**: Strategy performance evaluation
4. **Real-time Data Streaming**: Live market analysis

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your amazing feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing free financial data
- The Python scientific computing community
- Contributors to pandas, numpy, scikit-learn, and other open-source libraries


**Happy Analyzing! 📈🤖**

*Transform your investment decisions with data-driven insights and machine learning intelligence.*