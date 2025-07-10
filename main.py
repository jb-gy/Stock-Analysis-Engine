# try:
#     choice = input("\nEnter your choice (1-4): ").strip()
    
#     if choice == "1":
#         symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
#         period = input("Enter period (1y, 2y, 5y) [default: 2y]: ").strip() or "2y"
#         analyze_single_stock(symbol, period)
        
#     elif choice == "2":
#         symbols_input = input(f"Enter symbols separated by commas [default: {', '.join(default_stocks)}]: ").strip()
#         if symbols_input:
#             symbols = [s.strip().upper() for s in symbols_input.split(",")]
#         else:
#             symbols = default_stocks
#         period = input("Enter period (1y, 2y, 5y) [default: 2y]: ").strip() or "2y"
#         compare_stocks(symbols, period)
        
#     elif choice == "3":
#         print("\n🍎 Running quick demo with Apple (AAPL)...")
#         analyze_single_stock("AAPL", "2y")
        
#     elif choice == "4":
#         print("\n🤖 Running ML Demo: Tech stocks correlation analysis...")
#         tech_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA"]
#         compare_stocks(tech_stocks, "2y")
        
#     else:
#         print("❌ Invalid choice. Running demo...")
#         analyze_single_stock("AAPL", "1y")#!/usr/bin/env python3
# """
# Stock Analysis Engine - Main Entry Point

# This script demonstrates the full capabilities of the stock analysis engine.
# """

import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from analysis.financial_metrics import FinancialMetrics
from utils.data_fetcher import StockDataFetcher
from visualization.stats_visualizer import StatsVisualizer
from ml.correlation_analyzer import CorrelationAnalyzer


def analyze_single_stock(symbol: str, period: str = "2y"):
    """Analyze a single stock with full statistics and visualizations."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {symbol.upper()}")
    print(f"{'='*60}")
    
    try:
        # Initialize components
        fetcher = StockDataFetcher()
        metrics = FinancialMetrics()
        visualizer = StatsVisualizer()
        
        # Fetch data
        print(f"📊 Fetching {period} of data for {symbol}...")
        df = fetcher.get_stock_data(symbol, period=period)
        print(f"✅ Retrieved {len(df)} trading days of data")
        
        # Calculate metrics
        print(f"🧮 Calculating financial metrics...")
        stats = metrics.analyze_stock(df)
        
        # Display results
        print(f"\n📈 FINANCIAL STATISTICS for {symbol}:")
        print(f"{'─'*50}")
        print(f"Historical Volatility: {stats['historical_volatility']:.2f}%")
        print(f"Skewness: {stats['skewness']:.3f}")
        print(f"Kurtosis: {stats['kurtosis']:.3f}")
        print(f"Annualized Mean Return: {stats['mean_return']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        
        # Returns distribution plot
        dist_plot = visualizer.plot_returns_distribution(symbol, period)
        if dist_plot:
            print(f"✅ Returns distribution saved: {dist_plot}")
        
        # Volatility trend plot
        vol_plot = visualizer.plot_volatility_trend(symbol, period)
        if vol_plot:
            print(f"✅ Volatility trend saved: {vol_plot}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        return None


def compare_stocks(symbols: list, period: str = "2y"):
    """Compare multiple stocks with statistical analysis and ML correlation models."""
    print(f"\n{'='*60}")
    print(f"COMPARATIVE ANALYSIS WITH MACHINE LEARNING")
    print(f"{'='*60}")
    
    try:
        visualizer = StatsVisualizer()
        ml_analyzer = CorrelationAnalyzer()
        
        # Analyze each stock individually
        all_stats = {}
        for symbol in symbols:
            stats = analyze_single_stock(symbol, period)
            if stats:
                all_stats[symbol] = stats
        
        # Create traditional comparative visualization
        if len(all_stats) > 1:
            print(f"\n🔄 Creating traditional comparative analysis...")
            comp_plot = visualizer.plot_comparative_stats(list(all_stats.keys()), period)
            if comp_plot:
                print(f"✅ Comparative analysis saved: {comp_plot}")
            
            # NEW: ML-powered correlation and regression analysis
            if len(all_stats) >= 2:
                ml_results = ml_analyzer.comprehensive_analysis(list(all_stats.keys()), period)
                
                # Display ML visualization paths
                if 'visualizations' in ml_results:
                    viz = ml_results['visualizations']
                    print(f"\n🤖 MACHINE LEARNING VISUALIZATIONS:")
                    if viz['price_correlation_heatmap']:
                        print(f"✅ Price correlation heatmap: {viz['price_correlation_heatmap']}")
                    if viz['returns_correlation_heatmap']:
                        print(f"✅ Returns correlation heatmap: {viz['returns_correlation_heatmap']}")
                    if viz['regression_analysis_plot']:
                        print(f"✅ Regression analysis plots: {viz['regression_analysis_plot']}")
                    if viz['rolling_correlations_plot']:
                        print(f"✅ Rolling correlations plot: {viz['rolling_correlations_plot']}")
            
            # Traditional summary comparison
            print(f"\n📊 TRADITIONAL COMPARATIVE SUMMARY:")
            print(f"{'─'*70}")
            print(f"{'Stock':<8} {'Vol%':<8} {'Skew':<8} {'Kurt':<8} {'Return%':<10} {'Sharpe':<8}")
            print(f"{'─'*70}")
            
            for symbol, stats in all_stats.items():
                print(f"{symbol:<8} {stats['historical_volatility']:<8.2f} "
                      f"{stats['skewness']:<8.3f} {stats['kurtosis']:<8.3f} "
                      f"{stats['mean_return']:<10.2f} {stats['sharpe_ratio']:<8.3f}")
        
    except Exception as e:
        print(f"❌ Error in comparative analysis: {str(e)}")


def main():
    """Main function - demonstrates full capabilities."""
    print("🚀 Stock Analysis Engine")
    print("=" * 60)
    
    # Default stocks to analyze
    default_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    print("\nWhat would you like to do?")
    print("1. Analyze a single stock")
    print("2. Compare multiple stocks with ML correlation analysis")
    print("3. Quick demo with Apple (AAPL)")
    print("4. ML Demo: Tech stocks correlation analysis (AAPL, MSFT, GOOGL, TSLA)")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
            period = input("Enter period (1y, 2y, 5y) [default: 2y]: ").strip() or "2y"
            analyze_single_stock(symbol, period)
            
        elif choice == "2":
            symbols_input = input(f"Enter symbols separated by commas [default: {', '.join(default_stocks)}]: ").strip()
            if symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(",")]
            else:
                symbols = default_stocks
            period = input("Enter period (1y, 2y, 5y) [default: 2y]: ").strip() or "2y"
            compare_stocks(symbols, period)
            
        elif choice == "3":
            print("\n🍎 Running quick demo with Apple (AAPL)...")
            analyze_single_stock("AAPL", "2y")
            
        else:
            print("❌ Invalid choice. Running demo...")
            analyze_single_stock("AAPL", "1y")
            
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    
    print(f"\n🎉 Analysis complete! Check these folders for results:")
    print(f"   📊 Traditional plots: 'output/plots'")
    print(f"   🤖 ML analysis plots: 'output/ml_plots'")


if __name__ == "__main__":
    main()