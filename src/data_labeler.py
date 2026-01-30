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
    Supports multiple horizons (Short Term vs Long Term).
    """
    def __init__(self, orders: list[int] | None = None, profit_threshold: float = 0.02):
        """
        Args:
            orders: List of window sizes for local min/max comparison.
                    5 = Short Term (Weekly trends)
                    20 = Long Term (Monthly trends)
            profit_threshold: Minimum % move to consider valid (ignore flat noise).
        """
        self.orders = orders if orders is not None else [5, 20]
        if not self.orders:
             raise ValueError("orders list cannot be empty. Please provide at least one order window.")
        self.profit_threshold = profit_threshold

    def label_ticker(self, df: pd.DataFrame, col_name: str = 'Close') -> pd.DataFrame:
        """
        Adds 'Target_Action_{order}' columns: 0=Hold, 1=Buy, 2=Sell
        Also adds a default 'Target_Action' which matches the first order.
        """
        df = df.copy()
        prices = df[col_name].values
        
        for order in self.orders:
            # 1. Find Local Minima (Valleys) and Maxima (Peaks)
            ilocs_min = argrelextrema(prices, np.less_equal, order=order)[0]
            ilocs_max = argrelextrema(prices, np.greater_equal, order=order)[0]

            # 2. Assign Labels
            labels = np.zeros(len(df), dtype=int)
            
            # Mark Buys (1)
            for i in ilocs_min:
                labels[i] = 1
                
            # Mark Sells (2)
            for i in ilocs_max:
                labels[i] = 2
                
            col_label = f"Target_Action_{order}"
            df[col_label] = labels
            
        # Default target (first one) for backward compatibility
        df['Target_Action'] = df[f"Target_Action_{self.orders[0]}"]
        
        return df

    def visualize(self, df: pd.DataFrame, ticker_name: str, target_col: str = 'Target_Action', n_samples: int = 500, save_path: str = None):
        """
        Save a plot verifying the labels.
        """
        subset = df.iloc[-n_samples:]
        prices = subset['Close'].values
        actions = subset[target_col].values
        
        plt.figure(figsize=(12, 6))
        plt.plot(prices, label='Price', color='gray', alpha=0.5)
        
        # Plot Buys
        buy_indices = np.where(actions == 1)[0]
        plt.scatter(buy_indices, prices[buy_indices], color='green', marker='^', s=100, label='Buy')
        
        # Plot Sells
        sell_indices = np.where(actions == 2)[0]
        plt.scatter(sell_indices, prices[sell_indices], color='red', marker='v', s=100, label='Sell')
        
        plt.title(f"Labels: {target_col} for {ticker_name}")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.savefig(f"golden_labels_{ticker_name}.png")
            
        plt.close()

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
    labeler = GoldenLabeler(orders=[5, 20])
    labeled_df = labeler.label_ticker(df)
    
    print("Short Term (5):")
    print(labeled_df['Target_Action_5'].value_counts())
    print("Long Term (20):")
    print(labeled_df['Target_Action_20'].value_counts())
    
    # Visualize
    labeler.visualize(labeled_df, ticker, target_col='Target_Action_5', save_path=f"golden_labels_{ticker}_short.png")
    labeler.visualize(labeled_df, ticker, target_col='Target_Action_20', save_path=f"golden_labels_{ticker}_long.png")

if __name__ == "__main__":
    main()
