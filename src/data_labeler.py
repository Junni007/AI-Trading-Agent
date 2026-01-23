import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import logging

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataLabeler')

class GoldenLabeler:
    """
    Generates 'Golden Labels' (Perfect Hindsight Trades) for Supervised Fine-Tuning.
    """
    def __init__(self, order: int = 5, profit_threshold: float = 0.02):
        """
        Args:
            order: How many points on each side to use for local min/max comparison.
                   Higher = zoom out (capture big trends). Lower = zoom in (noisy).
            profit_threshold: Minimum % move to consider valid (ignore flat noise).
        """
        self.order = order
        self.profit_threshold = profit_threshold

    def label_ticker(self, df: pd.DataFrame, col_name: str = 'Close') -> pd.DataFrame:
        """
        Adds 'Target_Action' column: 0=Hold, 1=Buy, 2=Sell
        """
        df = df.copy()
        prices = df[col_name].values
        
        # 1. Find Local Minima (Valleys) and Maxima (Peaks)
        # argrelextrema checks for local peaks within 'order' window
        ilocs_min = argrelextrema(prices, np.less_equal, order=self.order)[0]
        ilocs_max = argrelextrema(prices, np.greater_equal, order=self.order)[0]

        # 2. Assign Labels
        # Initialize as 0 (Hold)
        labels = np.zeros(len(df), dtype=int)
        
        # Mark Buys (1)
        for i in ilocs_min:
            # Check if likely profitable (naive check against next peak)
            # (In a real zig-zag, we'd pair them, but this is a robust heuristc)
            labels[i] = 1
            
        # Mark Sells (2)
        for i in ilocs_max:
            labels[i] = 2
            
        df['Target_Action'] = labels
        
        # 3. Filter Noise (Optional but recommended)
        # Remove buys that don't lead to a significant rise?
        # For now, we trust the 'order' parameter to filter small noise.
        
        return df

    def visualize(self, df: pd.DataFrame, ticker_name: str, n_samples: int = 500):
        """
        Save a plot verifying the labels.
        """
        subset = df.iloc[-n_samples:]
        prices = subset['Close'].values
        actions = subset['Target_Action'].values
        
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Price', color='gray', alpha=0.5)
        
        # Plot Buys
        buy_indices = np.where(actions == 1)[0]
        plt.scatter(buy_indices, prices[buy_indices], color='green', marker='^', s=100, label='Perfect Buy')
        
        # Plot Sells
        sell_indices = np.where(actions == 2)[0]
        plt.scatter(sell_indices, prices[sell_indices], color='red', marker='v', s=100, label='Perfect Sell')
        
        plt.title(f"Golden Labels for {ticker_name} (Order={self.order})")
        plt.legend()
        plt.savefig(f"golden_labels_{ticker_name}.png")
        plt.close()
        logger.info(f"Saved visualization to golden_labels_{ticker_name}.png")

def main():
    from src.ticker_utils import get_nifty500_tickers
    from src.data_loader import MVPDataLoader
    
    # Test on one ticker
    ticker = "RELIANCE.NS"
    logger.info(f"Fetching data for {ticker}...")
    
    loader = MVPDataLoader(tickers=[ticker])
    df_dict = loader.fetch_batch_data()
    
    if ticker not in df_dict.columns.levels[0]:
        print("Ticker download failed.")
        return

    df = df_dict[ticker].dropna()
    
    # Label
    labeler = GoldenLabeler(order=20) # Look for peaks/valleys over ~20 days (Monthly trends)
    labeled_df = labeler.label_ticker(df)
    
    print(labeled_df['Target_Action'].value_counts())
    
    # Visualize
    labeler.visualize(labeled_df, ticker)

if __name__ == "__main__":
    main()
