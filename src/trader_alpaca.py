import os
import logging
import time
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
        # Input: 9 (Features=7 + 2 context: Position, Balance)
        # Output: 3 (Hold, Buy, Sell)
        model = RecurrentActorCritic(input_dim=9, output_dim=3, hidden_dim=256)
        
        # Load Weights
        # Priority: Best Checkpoint > SFT Model > Random
        checkpoint_path = "checkpoints/best_ppo.ckpt" 
        sft_path = "checkpoints_sft/final_sft_model.pth"
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading Best RL Checkpoint: {checkpoint_path}")
            # Lightning checkpoints are nested
            # We need to extract the state_dict for the inner model
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
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
            model.load_state_dict(torch.load(sft_path, map_location=self.device, weights_only=True))
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
        
        if df.empty or len(df) < 100:
            logger.error("Not enough market data to make a decision!")
            return
        
        # Feature Engineering (Re-use VectorizedTradingEnv logic)
        # VectorizedTradingEnv._precompute_features() computes 7 market features:
        # [Returns, LogReturns, Volatility, Volume_Z, RSI, RSI_Rank, Momentum_Rank]
        env = VectorizedTradingEnv(df, n_envs=1, window_size=50)
        
        if len(env.features) < 50:
            logger.error("Not enough data to form a window!")
            return
            
        # Get the latest window — Shape: (50, 7)
        recent_window = env.features[-50:]
        
        # Get Context
        current_pos = self.get_position()
        current_bal = self.get_balance_ratio()
        
        logger.info(f"Context -> Position: {current_pos}, Balance Ratio: {current_bal:.2f}")
        
        # Create Observation Tensor — Shape: (1, 50, 7)
        seq_tensor = recent_window.unsqueeze(0).to(self.device)
        
        # Create Context Tensor — Shape: (1, 2)
        ctx_tensor = torch.tensor([[current_pos, current_bal]], device=self.device)
        
        # Broadcast Context -> (1, 50, 2)
        ctx_expanded = ctx_tensor.unsqueeze(1).expand(-1, 50, -1)
        
        # Full Observation: Features + Context -> (1, 50, 9)
        full_obs = torch.cat([seq_tensor, ctx_expanded], dim=-1)
        
        # Predict
        with torch.no_grad():
            action_probs, value, _ = self.model(full_obs)
            action = torch.argmax(action_probs, dim=1).item()
            confidence = action_probs[0][action].item()
            
        current_price = df.iloc[-1]['Close']
        action_names = ['HOLD', 'BUY', 'SELL']
        logger.info(f"Brain Output: Action={action} ({action_names[action]}), Conf={confidence:.2f} | Price: {current_price:.2f}")
        
        # Execute
        self.execute_order(int(action), current_pos, current_price, confidence)

    def log_trade_to_csv(self, action, price, qty, confidence, position, pnl=None):
        """Appends trade details to a CSV file."""
        file_path = "trade_log.csv"
        file_exists = os.path.isfile(file_path)
        try:
            with open(file_path, "a") as f:
                if not file_exists:
                    f.write("Timestamp,Symbol,Action,Price,Qty,Confidence,Position_Before,PnL,Balance\n")
                
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                # Fetch Balance
                try:
                    acct = self.trading_client.get_account()
                    balance = float(acct.cash)
                except Exception as e:
                    logger.warning(f"Could not fetch balance: {e}")
                    balance = 0.0

                pnl_str = f"{pnl:.2f}" if pnl is not None else "0.00"
                f.write(f"{timestamp},{self.symbol},{action},{price:.2f},{qty},{confidence:.2f},{position},{pnl_str},{balance:.2f}\n")
        except Exception as e:
            logger.error(f"Failed to write to CSV: {e}")

    def execute_order(self, action: int, current_pos: float, current_price: float, confidence: float):
        if self.dry_run:
            if action == 1:
                 logger.info(f"Dry Run: Would BUY {self.quantity} shares @ {current_price} (Conf: {confidence:.2f})")
                 self.log_trade_to_csv("BUY (Dry)", current_price, self.quantity, confidence, current_pos, 0.0)
            elif action == 2:
                 logger.info(f"Dry Run: Would SELL {self.quantity} shares @ {current_price} (Conf: {confidence:.2f})")
                 self.log_trade_to_csv("SELL (Dry)", current_price, self.quantity, confidence, current_pos, 0.0)
            return

        try:
            if action == 1: # BUY
                if current_pos == 0:
                    logger.info(f"📢 Placing BUY Order for {self.quantity} {self.symbol} (Conf: {confidence:.2f})...")
                    req = MarketOrderRequest(
                        symbol=self.symbol,
                        qty=self.quantity,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                    )
                    self.trading_client.submit_order(req)
                    logger.info("✅ Order Submitted.")
                    self.log_trade_to_csv("BUY", current_price, self.quantity, confidence, current_pos, 0.0)
                else:
                    logger.info("Signal BUY, but already Long. Holding.")
                    
            elif action == 2: # SELL
                if current_pos > 0:
                    logger.info(f"📢 Placing SELL Order for {self.symbol} (Conf: {confidence:.2f})...")
                    
                    # Calculate PnL (Estimated based on entry)
                    pnl = 0.0
                    try:
                        pos_obj = self.trading_client.get_open_position(self.symbol)
                        entry_price = float(pos_obj.avg_entry_price)
                        pnl = (current_price - entry_price) * float(current_pos) # Realized PnL approx
                        logger.info(f"💰 Realized PnL: ${pnl:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not calc PnL: {e}")

                    self.trading_client.close_position(self.symbol)
                    logger.info("✅ Position Closed.")
                    self.log_trade_to_csv("SELL", current_price, current_pos, confidence, current_pos, pnl)
                else:
                    logger.info("Signal SELL, but no position. Holding.")
            
            else: # HOLD
                logger.info("Signal HOLD. Staying put.")
                
        except Exception as e:
            logger.error(f"Execution Error: {e}")

    def run_forever(self, interval_seconds: int = 60):
        """
        Run the agent in a 24/7 loop.
        Actively monitors for exit opportunities if holding a position.
        """
        logger.info(f"🔁 Starting Continuous Trader (Interval: {interval_seconds}s)")
        logger.info("The Brain is now actively searching for opportunities...")
        
        try:
            while True:
                # 1. Check Status
                pos = self.get_position()
                if pos > 0:
                    logger.info(f"🕵️  Active Position Detected ({self.symbol}). Searching for optimal EXIT...")
                else:
                    logger.info(f"🔭 Flat. Searching for optimal ENTRY...")

                # 2. Execute Strategy
                self.think_and_act()
                
                # 3. Wait
                logger.info(f"Sleeping for {interval_seconds}s...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("🛑 Trader stopped by user.")
        except Exception as e:
            logger.error(f"Critical Loop Error: {e}")
            time.sleep(60) # Backoff before retrying

def main():
    import argparse
    import time # Ensure time is imported
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock Ticker (e.g. AAPL, TSLA)")
    parser.add_argument("--qty", type=float, default=1, help="Quantity to trade")
    parser.add_argument("--live", action="store_true", help="Disable Dry Run (Real Paper Trading)")
    parser.add_argument("--loop", action="store_true", help="Run continuously (every 60s)")
    
    args = parser.parse_args()
    
    # Check for .env
    if not os.path.exists(".env"):
        logger.warning("No .env file found! Please create one with APCA_API_KEY_ID.")
        return

    try:
        trader = AlpacaTrader(symbol=args.symbol, quantity=args.qty, dry_run=not args.live)
        
        if args.loop:
            trader.run_forever(interval_seconds=60)
        else:
            trader.think_and_act()
            
    except Exception as e:
        logger.error(f"Trader Crash: {e}")

if __name__ == "__main__":
    main()
