"""
Signal.Engine - Optimized PPO Training Script
GPU-optimized training with vectorized environments and larger batches.
"""
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.distributions import Categorical
import logging

# Suppress gym deprecation warning
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

from src.data_loader import MVPDataLoader
from src.ticker_utils import get_nifty500_tickers

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PPOTrainer')


class LargerActorCritic(nn.Module):
    """
    Larger model for better GPU utilization.
    Original: 18K params -> New: ~500K params
    """
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        
        # Deeper network with residual connections
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x):
        # Feature extraction with residual connections
        x = F.relu(self.ln1(self.fc1(x)))
        identity = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = x + identity  # Residual
        identity = x
        x = F.relu(self.ln3(self.fc3(x)))
        x = x + identity  # Residual
        x = F.relu(self.fc4(x))
        
        # Outputs
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value


class VectorizedTradingEnv:
    """
    Vectorized environment for parallel rollouts on GPU.
    Runs multiple environment copies simultaneously.
    """
    def __init__(self, df: pd.DataFrame, n_envs: int = 64, initial_balance: float = 10000, window_size: int = 50):
        self.df = df
        self.n_envs = n_envs
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Precompute all possible observations as a tensor (on GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract OHLCV and precompute features
        self._precompute_features()
        
        # Environment state
        self.positions = torch.zeros(n_envs, device=self.device)
        self.balances = torch.full((n_envs,), initial_balance, device=self.device)
        self.current_steps = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        
        self.max_steps = len(self.features) - self.window_size - 1
        
    def _precompute_features(self):
        """Precompute all observations as a GPU tensor."""
        # Ensure required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = self.df[required].copy()
        
        # Compute returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(20).std()
        df['Volume_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # Normalize
        df = df.dropna()
        
        # Feature columns
        feature_cols = ['Returns', 'LogReturns', 'Volatility', 'Volume_Z', 'RSI']
        features = df[feature_cols].values.astype(np.float32)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        self.features = torch.from_numpy(features).to(self.device)
        self.prices = torch.from_numpy(df['Close'].values.astype(np.float32)).to(self.device)
        
        self.obs_dim = len(feature_cols) * self.window_size + 2  # +2 for position and balance ratio
        
    def reset(self):
        """Reset all environments."""
        # Random starting points for each env
        max_start = self.max_steps - 500  # Leave room for full episode
        max_start = max(1, max_start)
        self.current_steps = torch.randint(0, max_start, (self.n_envs,), device=self.device)
        self.positions = torch.zeros(self.n_envs, device=self.device)
        self.balances = torch.full((self.n_envs,), self.initial_balance, device=self.device)
        return self._get_obs()
    
    def _get_obs(self):
        """Get observations for all environments (batch)."""
        batch_obs = []
        for i in range(self.n_envs):
            step = self.current_steps[i].item()
            window = self.features[step:step + self.window_size].flatten()
            pos_balance = torch.tensor([self.positions[i], self.balances[i] / self.initial_balance], device=self.device)
            obs = torch.cat([window, pos_balance])
            batch_obs.append(obs)
        return torch.stack(batch_obs)
    
    def step(self, actions):
        """
        Step all environments in parallel.
        Actions: 0=Hold, 1=Buy, 2=Sell
        """
        # Get current and next prices
        current_prices = self.prices[self.current_steps + self.window_size]
        next_prices = self.prices[self.current_steps + self.window_size + 1]
        
        # Calculate rewards based on actions
        returns = (next_prices - current_prices) / current_prices
        
        # Action effects
        rewards = torch.zeros(self.n_envs, device=self.device)
        
        # Buy (action=1): reward = returns if we're now long
        buy_mask = (actions == 1) & (self.positions == 0)
        self.positions = torch.where(buy_mask, torch.ones_like(self.positions), self.positions)
        
        # Sell (action=2): reward = position * returns if we close
        sell_mask = (actions == 2) & (self.positions == 1)
        rewards = torch.where(sell_mask, returns * 100, rewards)  # Scale reward
        self.positions = torch.where(sell_mask, torch.zeros_like(self.positions), self.positions)
        
        # Hold with position: reward based on unrealized P&L
        hold_long = (actions == 0) & (self.positions == 1)
        rewards = torch.where(hold_long, returns * 10, rewards)
        
        # Update balances
        self.balances = self.balances * (1 + rewards / 100)
        
        # Step forward
        self.current_steps += 1
        
        # Check if done
        dones = self.current_steps >= self.max_steps
        
        # Reset done environments
        if dones.any():
            reset_idx = dones.nonzero(as_tuple=True)[0]
            max_start = self.max_steps - 500
            max_start = max(1, max_start)
            self.current_steps[reset_idx] = torch.randint(0, max_start, (len(reset_idx),), device=self.device)
            self.positions[reset_idx] = 0
            self.balances[reset_idx] = self.initial_balance
        
        return self._get_obs(), rewards, dones


class OptimizedPPOAgent(pl.LightningModule):
    """
    GPU-optimized PPO Agent with vectorized environments.
    """
    def __init__(self, env: VectorizedTradingEnv, lr=3e-4, gamma=0.99, clip_eps=0.2, 
                 rollout_steps=256, ppo_epochs=4, mini_batch_size=256):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.clip_eps = clip_eps
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        self.obs_dim = env.obs_dim
        self.action_dim = 3  # Hold, Buy, Sell
        
        self.model = LargerActorCritic(self.obs_dim, self.action_dim)
        
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['env'])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        # 1. Collect vectorized rollout (all on GPU)
        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        dones_list = []
        
        state = self.env.reset()
        
        for _ in range(self.rollout_steps):
            with torch.no_grad():
                probs, value = self.model(state)
            
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done = self.env.step(action)
            
            states_list.append(state)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            values_list.append(value.squeeze(-1))
            dones_list.append(done.float())
            
            state = next_state
        
        # Stack all data (shape: [rollout_steps, n_envs, ...])
        states = torch.stack(states_list)  # [T, N, obs_dim]
        actions = torch.stack(actions_list)  # [T, N]
        old_log_probs = torch.stack(log_probs_list)  # [T, N]
        rewards = torch.stack(rewards_list)  # [T, N]
        values = torch.stack(values_list)  # [T, N]
        dones = torch.stack(dones_list)  # [T, N]
        
        # 2. Compute advantages using GAE
        with torch.no_grad():
            _, next_value = self.model(state)
            next_value = next_value.squeeze(-1)
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(self.env.n_envs, device=self.device)
        
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Flatten for mini-batch updates
        # [T, N, ...] -> [T*N, ...]
        T, N = states.shape[0], states.shape[1]
        flat_states = states.view(T * N, -1)
        flat_actions = actions.view(T * N)
        flat_old_log_probs = old_log_probs.view(T * N)
        flat_advantages = advantages.view(T * N)
        flat_returns = returns.view(T * N)
        
        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # 3. PPO epochs with mini-batches
        total_samples = T * N
        indices = torch.randperm(total_samples, device=self.device)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(self.ppo_epochs):
            for start in range(0, total_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, total_samples)
                mb_indices = indices[start:end]
                
                mb_states = flat_states[mb_indices]
                mb_actions = flat_actions[mb_indices]
                mb_old_log_probs = flat_old_log_probs[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]
                
                # Forward pass
                probs, new_values = self.model(mb_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # Logging
        avg_reward = rewards.mean().item()
        self.log("train_loss", total_actor_loss / n_updates, prog_bar=True)
        self.log("critic_loss", total_critic_loss / n_updates, prog_bar=True)
        self.log("reward", avg_reward, prog_bar=True)
        self.log("entropy", total_entropy / n_updates, prog_bar=True)
        
        return torch.tensor(total_actor_loss / n_updates)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
    
    def train_dataloader(self):
        # Dummy dataloader - actual data comes from vectorized env
        return torch.utils.data.DataLoader(
            torch.zeros(500),  # 500 training steps per epoch
            batch_size=1,
            num_workers=0  # No workers needed, everything is on GPU
        )


def prepare_training_data(num_tickers: int = 50):
    """
    Load and prepare training data.
    Handles yfinance cache issues on Kaggle/Colab.
    """
    import yfinance as yf
    
    # CRITICAL: Disable yfinance cache to avoid SQLite lock on Kaggle
    # This prevents "database is locked" errors
    try:
        yf.set_tz_cache_location("/tmp/yf_cache")  # Use temp dir on Kaggle
    except Exception:
        pass  # If it fails, yfinance will work without cache
    
    logger.info(f"Loading training data for {num_tickers} tickers...")
    
    all_tickers = get_nifty500_tickers()[:num_tickers]
    loader = MVPDataLoader(tickers=all_tickers)
    full_df = loader.fetch_batch_data()
    
    if full_df.empty:
        raise ValueError("Failed to download any data!")
    
    is_multi_index = isinstance(full_df.columns, pd.MultiIndex)
    
    # Find a ticker with sufficient data
    selected_df = None
    selected_ticker = None
    
    if is_multi_index:
        available = [t for t in all_tickers if t in full_df.columns.get_level_values(0)]
        if not available:
            raise ValueError("No tickers available in downloaded data.")
        
        # Try each ticker until we find one with enough data
        for ticker in available:
            try:
                df = full_df[ticker].copy()
                df.dropna(inplace=True)
                train_df = df[(df.index >= '2019-01-01') & (df.index <= '2022-12-31')]
                
                if len(train_df) >= 100:  # Minimum 100 rows
                    selected_df = train_df
                    selected_ticker = ticker
                    logger.info(f"Selected ticker for RL training: {ticker} ({len(train_df)} rows)")
                    break
                else:
                    logger.warning(f"Ticker {ticker} has only {len(train_df)} rows, skipping...")
            except Exception as e:
                logger.warning(f"Failed to process {ticker}: {e}")
                continue
        
        if selected_df is None:
            raise ValueError("No ticker has sufficient training data!")
    else:
        df = full_df.copy()
        df.dropna(inplace=True)
        selected_df = df[(df.index >= '2019-01-01') & (df.index <= '2022-12-31')]
        selected_ticker = "Single ticker"
    
    logger.info(f"Training data: {len(selected_df)} rows from {selected_df.index.min()} to {selected_df.index.max()}")
    
    return selected_df


def main():
    # Hyperparameters - Optimized for GPU
    NUM_TICKERS = 20
    NUM_EPOCHS = 100  # Fewer epochs but more data per epoch
    N_ENVS = 64  # Run 64 environments in parallel
    ROLLOUT_STEPS = 256  # Steps per rollout
    
    # 1. Prepare Data
    train_df = prepare_training_data(NUM_TICKERS)
    
    if len(train_df) < 100:
        logger.error("Not enough training data. Exiting.")
        return
    
    # 2. Create Vectorized Environment (GPU-accelerated)
    env = VectorizedTradingEnv(
        train_df, 
        n_envs=N_ENVS,
        initial_balance=10000, 
        window_size=50
    )
    
    logger.info(f"Vectorized environment created: {N_ENVS} parallel envs")
    logger.info(f"Observation dim: {env.obs_dim}")
    
    # 3. Create Optimized Agent
    agent = OptimizedPPOAgent(
        env, 
        lr=3e-4, 
        gamma=0.99, 
        clip_eps=0.2,
        rollout_steps=ROLLOUT_STEPS,
        ppo_epochs=4,
        mini_batch_size=256
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in agent.model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # 4. Setup Callbacks
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='ppo-opt-{epoch:03d}-{reward:.4f}',
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
    # Note: No gradient_clip_val here - we use manual optimization with manual clipping
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu',
        devices=1,  # Single GPU is usually better for RL
        precision='16-mixed',
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    logger.info("Starting Optimized PPO Training...")
    logger.info(f"Each step processes {N_ENVS * ROLLOUT_STEPS:,} environment interactions")
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
