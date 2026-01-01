import torch
import logging
from tqdm import tqdm
from src.data_loader import MarketData, StreamData
from src.agent import TradingAgent
from src.env import TradingEnv
from src.tda_features import FeatureProcessor
import pandas as pd
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LiveTrader")

def load_model(model_path, env):
    agent = TradingAgent(env)
    # Load state dict
    # Note: If saved as lightning checkpoint, keys might have 'model.' prefix
    # If saved via torch.save(agent.state_dict()), it should valid.
    # Lightning checkpoints are usually dict with 'state_dict' key.
    
    if model_path.endswith(".ckpt"):
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
    else:
        state_dict = torch.load(model_path)
        
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent

def run_live_simulation():
    logger.info("Starting Live Trading Simulation...")
    
    # 1. Fetch Data (Recent/Test Spli)
    tickers = ["AAPL"]
    start_date = "2023-01-01"
    end_date = "2023-06-01"
    
    loader = MarketData(tickers=tickers, start_date=start_date, end_date=end_date)
    df = loader.load_data()
    ticker_data = df.loc['AAPL']
    
    # 2. Setup Streaming
    logger.info("Setting up Data Stream...")
    stream = StreamData(ticker_data)
    
    # 3. Setup Agent and customized Env for streaming
    # We need a slight modification: The Env usually manages the data. 
    # For "Live" mode with Gymnasium, we often step() with external info or just loop manually.
    # Let's simple loop manually using the model and computing features directly to simulate "API" calls.
    
    logger.info("Loading Model...")
    # Using a dummy env to initialize agent structure
    dummy_env = TradingEnv(ticker_data.iloc[:100], window_size=50)
    
    try:
        agent = load_model("final_tda_agent.pth", dummy_env)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.warning("Model file not found. Using untrained agent for demonstration.")
        agent = TradingAgent(dummy_env)
    
    # 4. Simulation Loop
    processor = FeatureProcessor(embedding_dim=3, embedding_delay=1)
    window_size = 50
    history = []
    
    balance = 10000.0
    position = 0
    portfolio_values = []
    
    # Warmup
    logger.info("Warming up history...")
    for _ in range(window_size):
        date, row = stream.next()
        history.append(row['Close'])
        
    logger.info("Starting Trading Loop...")
    
    # We don't know exact length of stream remaining easily without peeking, 
    # but let's assume valid stream.
    # We use tqdm here for visual feedback if we knew total length.
    # Let's iterate until stream ends.
    
    pbar = tqdm(desc="Trading", unit="step")
    
    while True:
        step_data = stream.next()
        if step_data is None:
            break
            
        date, row = step_data
        price = row['Close']
        history.append(price)
        
        # Get Window
        window = np.array(history[-window_size:])
        
        # Feature Extraction
        tda_feats = processor.process(window)
        
        # Construct State
        # [TDA..., LogBalance, Position]
        obs = np.concatenate((
            tda_feats, 
            [np.log1p(balance), float(position)]
        )).astype(np.float32)
        
        # Decision
        action_idx, _ = agent.select_action(obs)
        
        # Execute (Simple Logic)
        if action_idx == 1: # Buy
            if balance > price:
                balance -= price
                position += 1
                pbar.write(f"{date} [BUY] @ {price:.2f}")
        elif action_idx == 2: # Sell
            if position > 0:
                balance += price
                position -= 1
                pbar.write(f"{date} [SELL] @ {price:.2f}")
                
        # Logging
        value = balance + position * price
        portfolio_values.append(value)
        pbar.set_postfix({"Value": f"{value:.2f}", "Pos": position})
        pbar.update(1)

    pbar.close()
    
    final_value = portfolio_values[-1]
    logger.info(f"Simulation Complete. Final Portfolio Value: ${final_value:.2f}")
    
    # Simple perf metric
    ret = (final_value - 10000.0) / 10000.0 * 100
    logger.info(f"Total Return: {ret:.2f}%")

if __name__ == "__main__":
    run_live_simulation()
