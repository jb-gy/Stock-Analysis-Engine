import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
            if dist_plot:
                print(f"Returns distribution plot saved: {dist_plot}")
            else:
                print(f"Failed to create returns distribution plot for {symbol}")
            
            # Generate and save volatility plot
            vol_plot = visualizer.plot_volatility_trend(symbol)
            if vol_plot:
                print(f"Volatility trend plot saved: {vol_plot}")
            else:
                print(f"Failed to create volatility plot for {symbol}")
            
        except Exception as e:
            print(f"Error creating visualizations for {symbol}: {str(e)}")
    
    # Test comparative visualization
    try:
        print("\nCreating comparative visualization:")
        print("-" * 30)
        comp_plot = visualizer.plot_comparative_stats(symbols)
        if comp_plot:
            print(f"Comparative statistics plot saved: {comp_plot}")
        else:
            print("Failed to create comparative plot")
    except Exception as e:
        print(f"Error creating comparative plot: {str(e)}")


if __name__ == "__main__":
    test_visualizations()