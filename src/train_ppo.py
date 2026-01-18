"""
Signal.Engine - PPO Training Script
Trains the TradingAgent (PPO) on historical Nifty 500 data.
"""
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import logging

from src.data_loader import MVPDataLoader
from src.env import TradingEnv
from src.agent import TradingAgent
from src.ticker_utils import get_nifty500_tickers

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PPOTrainer')

def prepare_training_data(num_tickers: int = 50):
    """
    Loads historical data for a subset of Nifty 500 tickers.
    Returns a concatenated DataFrame suitable for the TradingEnv.
    """
    logger.info(f"Loading training data for {num_tickers} tickers...")
    
    # Get tickers
    all_tickers = get_nifty500_tickers()[:num_tickers]
    
    # Use MVPDataLoader to get data
    loader = MVPDataLoader(tickers=all_tickers)
    full_df = loader.fetch_batch_data()
    
    if full_df.empty:
        raise ValueError("Failed to download any data!")
    
    # For RL, we typically train on a single combined "market" or pick one ticker.
    # Let's pick the first available ticker as a demo.
    # A more advanced version would train across all tickers (multi-env).
    
    is_multi_index = isinstance(full_df.columns, pd.MultiIndex)
    
    if is_multi_index:
        # Pick first available ticker
        available = [t for t in all_tickers if t in full_df.columns.get_level_values(0)]
        if not available:
            raise ValueError("No tickers available in downloaded data.")
        selected_ticker = available[0]
        df = full_df[selected_ticker].copy()
        logger.info(f"Selected ticker for RL training: {selected_ticker}")
    else:
        df = full_df.copy()
    
    # Clean
    df.dropna(inplace=True)
    
    # Filter to training period (2019-2022)
    train_df = df[(df.index >= '2019-01-01') & (df.index <= '2022-12-31')]
    
    logger.info(f"Training data: {len(train_df)} rows from {train_df.index.min()} to {train_df.index.max()}")
    
    return train_df

def main():
    # Hyperparameters - Optimized for better learning
    NUM_TICKERS = 20  # Increased for more diverse training data
    NUM_EPOCHS = 200  # Reduced epochs, better rollout quality instead
    
    # 1. Prepare Data
    import pandas as pd
    train_df = prepare_training_data(NUM_TICKERS)
    
    if len(train_df) < 100:
        logger.error("Not enough training data. Exiting.")
        return
    
    # 2. Create Environment
    env = TradingEnv(train_df, initial_balance=10000, window_size=50)
    
    # 3. Create Agent - Tuned hyperparameters
    agent = TradingAgent(env, lr=3e-4, gamma=0.99, clip_eps=0.1)  # Higher LR, lower clip
    
    # 4. Setup Callbacks
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='ppo-{epoch:03d}-{reward:.2f}',
        save_top_k=3,
        monitor='reward',
        mode='max',
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='reward',
        patience=50,
        mode='max',
        verbose=True
    )
    
    # 5. Train - Optimized configuration
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='auto',  # Uses GPU if available
        precision='16-mixed',  # Mixed precision for ~2x speedup
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    logger.info("Starting PPO Training...")
    trainer.fit(agent)
    
    # 6. Save Best
    best_path = checkpoint_callback.best_model_path
    if best_path:
        # Copy to standard name for RLExpert
        import shutil
        final_path = os.path.join(checkpoint_dir, "best_ppo.ckpt")
        shutil.copy(best_path, final_path)
        logger.info(f"✅ Training complete. Best model saved to: {final_path}")
    else:
        logger.warning("No best checkpoint found.")

if __name__ == "__main__":
    main()
