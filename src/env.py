import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from .tda_features import FeatureProcessor

class TradingEnv(gym.Env):
    """
    A custom Trading Environment that follows gymnasium interface.
    The observation space is the TDA feature vector.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000.0, window_size=50, tda_config=None):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Config for Feature Processor
        if tda_config is None:
            tda_config = {"embedding_dim": 3, "embedding_delay": 1, "max_homology_dim": 1}
        
        self.processor = FeatureProcessor(**tda_config)
        
        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # TDA Features (6 floats: [H0_E, H0_M, H0_Max, H1_E, H1_M, H1_Max]) 
        # + Account Info (Balance, Position Size, Unrealized PnL) -> 9 values
        # We start with TDA features only for simplicity, but let's do the full vector.
        # Actually in the implementation of tda_features.py, we have 6 features (3 per dim, 2 dims).
        
        # Determine feature size dynamically by processing a dummy window
        dummy_data = np.zeros(window_size)
        dummy_feats = self.processor.process(dummy_data)
        self.n_features = len(dummy_feats)
        
        # +4 for Balance, Position%, PnL%, current price change
        self.obs_shape = (self.n_features + 4,) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        # State variables
        self.balance = initial_balance
        self.position = 0.0 # Number of shares
        self.current_step = window_size
        self.net_worth_history = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0.0
        # Start at window_size so we have enough data for embedding
        self.current_step = self.window_size + 1 # +1 to be safe for lags
        
        # If df is smaller than window, this will fail. Assume df is large enough.
        
        observation = self._next_observation()
        info = {}
        return observation, info

    def _next_observation(self):
        # get window of close prices
        # df structure: if MultiIndex, we assume single ticker passed in.
        # If DataFrame has 'Close' column, use it.
        if isinstance(self.df, pd.DataFrame):
            price_data = self.df['Close'].values if 'Close' in self.df.columns else self.df.iloc[:, 0].values
        else:
            price_data = self.df # If series
            
        window = price_data[self.current_step - self.window_size : self.current_step]
        
        # TDA Features
        tda_feats = self.processor.process(window)
        
        # Get current price for calculations
        current_price = price_data[self.current_step - 1] if self.current_step > 0 else price_data[0]
        
        # Account info - richer features for better learning
        net_worth = self.balance + self.position * current_price
        balance_feat = np.log1p(self.balance)  # Log-scaled balance
        position_pct = (self.position * current_price) / (net_worth + 1e-7)  # Position as % of portfolio
        pnl_pct = (net_worth - self.initial_balance) / self.initial_balance  # PnL as %
        
        # Price momentum (simple return over window)
        price_return = (price_data[self.current_step - 1] - price_data[self.current_step - 10]) / (price_data[self.current_step - 10] + 1e-7) if self.current_step >= 10 else 0.0
        
        obs = np.concatenate((tda_feats, [balance_feat, position_pct, pnl_pct, price_return]))
        return obs.astype(np.float32)

    def step(self, action):
        # Get current price
        # df assumption similar to _next_observation
        if isinstance(self.df, pd.DataFrame):
             price_series = self.df['Close'] if 'Close' in self.df.columns else self.df.iloc[:, 0]
        else:
             price_series = self.df
             
        current_price = float(price_series.iloc[self.current_step])
        prev_net_worth = self.balance + self.position * current_price
        
        # Execute Action
        # 0: Hold
        # 1: Buy (Buy 1 share - simplified)
        # 2: Sell (Sell 1 share - simplified)
        
        # Position sizing: Trade 10% of portfolio value for meaningful learning signal
        trade_fraction = 0.10
        trade_value = prev_net_worth * trade_fraction
        unit = trade_value / current_price if current_price > 0 else 0
        
        if action == 1:  # Buy
            cost = current_price * unit
            if self.balance >= cost:
                self.balance -= cost
                self.position += unit
        elif action == 2:  # Sell
            sell_amount = min(unit, self.position)  # Can't sell more than we have
            if sell_amount > 0:
                self.balance += current_price * sell_amount
                self.position -= sell_amount
                 
        # Calculate Reward - SCALED for better learning signal
        current_net_worth = self.balance + self.position * current_price
        self.net_worth_history.append(current_net_worth)
        
        # Reward = Percentage change * 100 for meaningful gradient signal
        pct_change = (current_net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        reward = pct_change * 100  # Scale up significantly
        
        # Advance step
        self.current_step += 1
        
        terminated = self.current_step >= len(price_series) - 1
        truncated = False
        
        obs = self._next_observation() if not terminated else np.zeros(self.obs_shape, dtype=np.float32)
        info = {"net_worth": current_net_worth, "price": current_price}
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human', close=False):
        current_net_worth = self.balance + self.position * self.df.iloc[self.current_step]['Close']
        print(f'Step: {self.current_step}, Net Worth: {current_net_worth:.2f}')
