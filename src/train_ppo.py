"""
Signal.Engine - PPO Training Script
Standard training script for the TradingAgent on historical data.
For GPU-optimized training with vectorized environments, use train_ppo_optimized.py
"""
import os
import warnings
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import logging

# Suppress gym deprecation warning (we're using gymnasium, but some deps import old gym)
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

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
    
    # For RL, we pick one ticker for cleaner reward signal
    is_multi_index = isinstance(full_df.columns, pd.MultiIndex)
    
    if is_multi_index:
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
    # Hyperparameters
    NUM_TICKERS = 20
    NUM_EPOCHS = 100
    ROLLOUT_STEPS = 512  # Steps per rollout
    
    # 1. Prepare Data
    train_df = prepare_training_data(NUM_TICKERS)
    
    if len(train_df) < 100:
        logger.error("Not enough training data. Exiting.")
        return
    
    # 2. Create Environment
    env = TradingEnv(train_df, initial_balance=10000, window_size=50)
    logger.info(f"Environment created. Observation space: {env.observation_space.shape}")
    
    # 3. Create Agent with improved hyperparameters
    agent = TradingAgent(
        env, 
        lr=3e-4,
        gamma=0.99, 
        clip_eps=0.2,
        gae_lambda=0.95,
        rollout_steps=ROLLOUT_STEPS,
        ppo_epochs=4,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Log model info
    n_params = sum(p.numel() for p in agent.model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
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
        patience=30,
        mode='max',
        verbose=True
    )
    
    # 5. Train
    # Note: For RL, single GPU is usually better than DDP
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='auto',
        devices=1,  # Single device for RL
        precision='16-mixed',
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=0.5,
    )
    
    logger.info("Starting PPO Training...")
    logger.info(f"Each step: {ROLLOUT_STEPS} environment interactions, {4} PPO epochs")
    trainer.fit(agent)
    
    # 6. Save Best
    best_path = checkpoint_callback.best_model_path
    if best_path:
        import shutil
        final_path = os.path.join(checkpoint_dir, "best_ppo.ckpt")
        shutil.copy(best_path, final_path)
        logger.info(f"✅ Training complete. Best model saved to: {final_path}")
    else:
        logger.warning("No best checkpoint found.")


if __name__ == "__main__":
    main()

