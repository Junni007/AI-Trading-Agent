import numpy as np
from src.data_loader import MVPDataLoader
from src.ticker_utils import get_extended_tickers

def inspect_data():
    print("Inspecting Data...")
    # Load small subset of tickers (e.g. 5)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"] 
    loader = MVPDataLoader(tickers=tickers, window_size=50)
    splits = loader.get_data_splits()
    
    X_train, y_train = splits['train']
    
    print(f"\nDataset Shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    
    # Check Class Balance
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nClass Distribution (Train):")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} ({c/len(y_train)*100:.2f}%)")
        
    # Check Feature Stats
    # X_train shape: (N, 50, 7) -> (Sample, Time, Feat)
    # We flatten N/Time to check feature distributions
    print(f"\nFeature Statistics (Expected ~0.0 mean, ~1.0 std):")
    feat_names = ['RSI', 'MACD', 'MACD_Signal', 'Log_Return', 'Trend_Signal', 'ATR', 'Vol_Change']
    
    for i, name in enumerate(feat_names):
        try:
            feat_vals = X_train[:, :, i].flatten()
            print(f"{name}: Mean={feat_vals.mean():.4f}, Std={feat_vals.std():.4f}, Min={feat_vals.min():.4f}, Max={feat_vals.max():.4f}")
        except IndexError:
            # Handle if data loading failed or features mismatch
            pass
        
    # Check for NaNs
    if np.isnan(X_train).any():
        print("\n❌ WARNING: NaNs found in X_train!")
    else:
        print("\n✅ No NaNs in X_train.")

if __name__ == "__main__":
    inspect_data()
