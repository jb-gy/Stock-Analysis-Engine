from visualization.stats_visualizer import StatsVisualizer

def test_visualizations():
    """Test the visualization functions with sample stocks."""
    visualizer = StatsVisualizer()
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    print("\nGenerating visualizations...")
    print("=" * 50)
    
    # Test individual stock visualizations
    for symbol in symbols:
        try:
            print(f"\nCreating visualizations for {symbol}:")
            print("-" * 30)
            
            # Generate and save distribution plot
            dist_plot = visualizer.plot_returns_distribution(symbol)
            print(f"Returns distribution plot saved: {dist_plot}")
            
            # Generate and save volatility plot
            vol_plot = visualizer.plot_volatility_trend(symbol)
            print(f"Volatility trend plot saved: {vol_plot}")
            
        except Exception as e:
            print(f"Error creating visualizations for {symbol}: {str(e)}")
    
    # Test comparative visualization
    try:
        comp_plot = visualizer.plot_comparative_stats(symbols)
        print(f"\nComparative statistics plot saved: {comp_plot}")
    except Exception as e:
        print(f"Error creating comparative plot: {str(e)}")

if __name__ == "__main__":
    test_visualizations()
