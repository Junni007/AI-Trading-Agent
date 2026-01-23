import os
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.train_ppo_optimized import RecurrentActorCritic, VectorizedTradingEnv

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AlpacaTrader')

class AlpacaTrader:
    """
    Live Trading Agent connecting PPO Brain to Alpaca Broker.
    """
    def __init__(self, symbol: str, quantity: float = 1, dry_run: bool = True):
        load_dotenv()
        
        self.symbol = symbol
        self.quantity = quantity
        self.dry_run = dry_run
        
        # Credentials
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Keys not found in environment (.env)")
            
        # Clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Model
        self.device = torch.device("cpu") # Inference on CPU is fine
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the best trained model architecture and weights."""
        logger.info("Loading Brain...")
        
        # Initialize Architecture (Must match training!)
        # Input: 7 (Features=5 + Pos + Bal)
        # Output: 3 (Hold, Buy, Sell)
        model = RecurrentActorCritic(input_dim=7, output_dim=3, hidden_dim=256)
        
        # Load Weights
        # Priority: Best Checkpoint > SFT Model > Random
        checkpoint_path = "checkpoints/best_ppo.ckpt" 
        sft_path = "checkpoints_sft/final_sft_model.pth"
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading Best RL Checkpoint: {checkpoint_path}")
            # Lightning checkpoints are nested
            # We need to extract the state_dict for the inner model
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                # Lightning saves as "model.lstm.weight...", we need to strip "model." prefix if loading into inner class
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith("model."):
                        state_dict[k.replace("model.", "")] = v
                model.load_state_dict(state_dict)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                
        elif os.path.exists(sft_path):
            logger.info(f"Loading SFT Model: {sft_path}")
            model.load_state_dict(torch.load(sft_path, map_location=self.device))
        else:
            logger.warning("⚠️ No weights found! Using Random Brain.")
            
        model.to(self.device)
        model.eval()
        return model

    def get_market_data(self):
        """Fetch last 50+ N candles."""
        logger.info(f"Fetching market data for {self.symbol}...")
        
        # Fetch enough data to compute 20-day rolling indicators
        # Window=50, SMA=50 -> Need at least 100 bars
        # Explicitly set start date 2 years back to ensure we get plenty of daily bars
        start_date = datetime.now() - timedelta(days=730) 
        
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Day, 
            start=start_date,
            limit=200
        )
        
        try:
            bars = self.data_client.get_stock_bars(request_params)
            df = bars.df
        except Exception as e:
            logger.error(f"Alpaca Data Error: {e}")
            return pd.DataFrame() # Return empty
        
        logger.info(f"Fetched {len(df)} bars for {self.symbol}")
        
        # Alpaca returns MultiIndex (symbol, timestamp) -> Reset
        df = df.reset_index(level=0, drop=True) 
        
        # Rename columns to match Training Env
        # Alpaca: open, high, low, close, volume, trade_count, vwap
        # Env expects: Open, High, Low, Close, Volume (Capitalized)
        df.rename(columns={
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume'
        }, inplace=True)
        
        return df

    def get_position(self):
        """Check if we currently hold the asset."""
        try:
            position = self.trading_client.get_open_position(self.symbol)
            qty = float(position.qty)
            return 1.0 if qty > 0 else 0.0
        except Exception:
            # 404 if no position
            return 0.0

    def get_balance_ratio(self):
        """Get available cash ratio (Simplified)."""
        account = self.trading_client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        # Use simple ratio
        return cash / equity if equity > 0 else 0.0

    def think_and_act(self):
        """Main Loop: Fetch -> Feature Eng -> Predict -> Act"""
        df = self.get_market_data()
        
        # Feature Engineering (Re-use VectorizedTradingEnv logic!)
        # We create a dummy env with 1 env to handle the math
        # Ideally, refactor feature engineering into a pure function
        # But instantiating this class is cheap enough
        env = VectorizedTradingEnv(df, n_envs=1, window_size=50)
        
        # The env automatically computes features in __init__ -> self.features
        # We need the LAST window (most recent data)
        # self.features shape: (Total_Steps, Features)
        
        if len(env.features) < 50:
            logger.error("Not enough data to form a window!")
            return
            
        # Get the latest window
        recent_window = env.features[-50:] # Shape (50, 5)
        
        # Get Context
        current_pos = self.get_position()
        current_bal = self.get_balance_ratio()
        
        logger.info(f"Context -> Position: {current_pos}, Balance Ratio: {current_bal:.2f}")
        
        # Create Observation Tensor
        # Shape: (1, 50, 5)
        seq_tensor = recent_window.unsqueeze(0).to(self.device)
        
        # Create Context Tensor
        # Shape: (1, 2)
        ctx_tensor = torch.tensor([[current_pos, current_bal]], device=self.device)
        
        # Broadcast Context -> (1, 50, 2)
        ctx_expanded = ctx_tensor.unsqueeze(1).expand(-1, 50, -1)
        
        # Fallback Features + Context -> (1, 50, 7)
        full_obs = torch.cat([seq_tensor, ctx_expanded], dim=-1)
        
        # Predict
        with torch.no_grad():
            action_probs, value, _ = self.model(full_obs)
            action = torch.argmax(action_probs, dim=1).item()
            confidence = action_probs[0][action].item()
            
        logger.info(f"🧠 Brain Output: Action={action} ({['HOLD', 'BUY', 'SELL'][action]}), Conf={confidence:.2f}")
        
        # Execute
        self.execute_order(action, current_pos)

    def execute_order(self, action: int, current_pos: float):
        if self.dry_run:
            logger.info("Dry Run Mode: No order placed.")
            return

        try:
            if action == 1: # BUY
                if current_pos == 0:
                    logger.info(f"📢 Placing BUY Order for {self.quantity} {self.symbol}...")
                    req = MarketOrderRequest(
                        symbol=self.symbol,
                        qty=self.quantity,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                    )
                    self.trading_client.submit_order(req)
                    logger.info("✅ Order Submitted.")
                else:
                    logger.info("Signal BUY, but already Long. Holding.")
                    
            elif action == 2: # SELL
                if current_pos > 0:
                    logger.info(f"📢 Placing SELL Order for {self.symbol}...")
                    self.trading_client.close_position(self.symbol)
                    logger.info("✅ Position Closed.")
                else:
                    logger.info("Signal SELL, but no position. Holding.")
            
            else: # HOLD
                logger.info("Signal HOLD. Staying put.")
                
        except Exception as e:
            logger.error(f"Execution Error: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock Ticker (e.g. AAPL, TSLA)")
    parser.add_argument("--qty", type=float, default=1, help="Quantity to trade")
    parser.add_argument("--live", action="store_true", help="Disable Dry Run (Real Paper Trading)")
    
    args = parser.parse_args()
    
    # Check for .env
    if not os.path.exists(".env"):
        logger.warning("No .env file found! Please create one with APCA_API_KEY_ID.")
        return

    try:
        trader = AlpacaTrader(symbol=args.symbol, quantity=args.qty, dry_run=not args.live)
        trader.think_and_act()
    except Exception as e:
        logger.error(f"Trader Crash: {e}")

if __name__ == "__main__":
    main()
