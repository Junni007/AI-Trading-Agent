import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

from src.data_loader import MVPDataLoader
from src.data_labeler import GoldenLabeler
from src.train_ppo_optimized import VectorizedTradingEnv  # Re-use for feature engineering logic

logger = logging.getLogger('SFTDataset')

class GoldenDataset(Dataset):
    """
    PyTorch Dataset for Supervised Fine-Tuning (SFT).
    Maps Market Sequences -> Golden Labels (Perfect Hindsight Actions).
    """
    def __init__(self, tickers: List[str], window_size: int = 50, outlier_filter: bool = True):
        self.window_size = window_size
        self.samples = []  # List of (sequence_tensor, label_int, context_tensor)
        
        # 1. Load Data
        loader = MVPDataLoader(tickers=tickers)
        data_dict = loader.fetch_batch_data()
        
        # 2. Labeler
        labeler = GoldenLabeler(order=20)
        
        if isinstance(data_dict, pd.DataFrame) and isinstance(data_dict.columns, pd.MultiIndex):
            # It's a MultiIndex DataFrame (Ticker, Field)
            tickers_in_data = data_dict.columns.get_level_values(0).unique()
            for ticker in tickers_in_data:
                try:
                    df = data_dict[ticker].copy().dropna()
                    if df.empty:
                        continue
                        
                    # Add Golden Labels
                    df = labeler.label_ticker(df)
                    
                    # Check if we have enough signals
                    if df['Target_Action'].sum() == 0:
                        continue
                        
                    self._process_ticker(df)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
        else:
            # Fallback for single ticker or flat dict (unlikely with MVPDataLoader)
            for ticker, df in data_dict.items():
                if df.empty:
                    continue
                try:
                    df = df.dropna()
                    df = labeler.label_ticker(df)
                    if df['Target_Action'].sum() == 0: continue
                    self._process_ticker(df)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                
        logger.info(f"Created SFT Dataset with {len(self.samples)} samples across {len(tickers)} tickers.")
        
        # Calculate Class Weights (for unbalanced loss)
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels)
        total = len(labels)
        self.class_weights = torch.tensor([total / c if c > 0 else 0 for c in counts], dtype=torch.float32)
        logger.info(f"Class Counts: {counts}")
        logger.info(f"Class Weights: {self.class_weights}")

    def _process_ticker(self, df: pd.DataFrame):
        """
        Convert labeled DataFrame into sequences.
        Re-uses VectorizedTradingEnv logic for feature consistency.
        """
        # Ensure we have all columns required by VectorizedTradingEnv
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            logger.warning("Skipping ticker due to missing columns")
            return

        # Create a dummy env just to use its feature engineering
        # (A bit inefficient but guarantees consistency)
        env = VectorizedTradingEnv(df, n_envs=1, window_size=self.window_size)
        
        features = env.features  # Tensor (T, Features)
        labels = df['Target_Action'].values
        
        # Align lengths (Env trims NaNs)
        # VectorizedTradingEnv drops initial NaNs, so we need to match indices
        # Features length is shorter than df length depending on NaN dropping
        # But VectorizedTradingEnv uses the Cleaned DF. 
        # Let's inspect env.features length vs labels length
        
        # Actually, VectorizedTradingEnv computes features from the provided DF.
        # It drops NaNs at the start.
        # We need to ensure labels align with features.
        
        # Simpler approach: Re-implement feature engineering explicitly or trust index alignment
        # Since Env changes the data processing (normalization), we should probably 
        # manually build the sequences here to be safe, OR trust that env.features corresponds 
        # to the END of the dataframe.
        
        # Let's rely on the fact that Env.features corresponds to the rows where data is valid.
        # Env.features rows = len(df) - nan_rows. 
        # Labels should be sliced from the end.
        
        n_features = len(env.features)
        labels = labels[-n_features:] 
        
        # Generate samples
        # Valid start index: 0
        # Valid end index: n_features - window_size
        
        for i in range(0, n_features - self.window_size):
            # Sequence
            # Shape: (Window, Features)
            seq = features[i : i + self.window_size]
            
            # Label
            # The label is the action we should take at the END of the window
            # i.e., at time step (i + window_size - 1)
            target = labels[i + self.window_size - 1]
            
            # Context (Position, Balance)
            # For SFT, we assume NEUTRAL context (No Position, Full Balance)
            # to teach the pure "Entry" signal.
            # We could inject Random Pos/Balance to teach Exits?
            # Phase 1: Teach Entries. Assume Neutral Context.
            
            # Broadcast neutral context: Position=0, Balance=1.0
            # Matches src/train_ppo_optimized.py _get_obs broadcasting
            # Shape (Window, 2)
            context = torch.tensor([0.0, 1.0], device=env.device).expand(self.window_size, 2)
            
            # Full Observation: (Window, Features+2)
            full_obs = torch.cat([seq, context], dim=-1)
            
            self.samples.append((full_obs.cpu(), int(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main():
    # Verification
    from src.ticker_utils import get_nifty500_tickers
    
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    dataset = GoldenDataset(tickers)
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    batch = next(iter(loader))
    obs, label = batch
    
    print(f"Observation Shape: {obs.shape}") # Should be (32, 50, 7)
    print(f"Label Shape: {label.shape}")
    print(f"Sample Labels: {label[:10]}")

if __name__ == "__main__":
    main()
