import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import MVPDataLoader
from src.lstm_model import LSTMPredictor
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtest")

def calculate_metrics(returns):
    """Calculates Sharpe, Max Drawdown, Total Return."""
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    # Sharpe (assuming daily)
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    
    # Drawdown
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }

def run_backtest():
    # 1. Load Data (Test Set Only)
    TICKER = "AAPL"
    loader = MVPDataLoader(ticker=TICKER)
    splits = loader.get_data_splits()
    X_test, y_test = splits['test']
    
    if len(X_test) == 0:
        logger.error("No test data available for backtesting.")
        return
    
    # 2. Load Model
    input_dim = X_test.shape[2]
    model = LSTMPredictor(input_dim=input_dim, output_dim=3)
    try:
        model.load_state_dict(torch.load("final_lstm_model.pth", weights_only=True))
        model.eval()
    except FileNotFoundError:
        logger.error("Model not found! Train first.")
        return

    # 3. Predict
    logger.info("Generating predictions...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()
        
    # 4. Simulate Strategy
    # 0: Down, 1: Neutral, 2: Up
    # Simple Strategy:
    # Position: 1 (Long) if Up, 0 (Cash) if Neutral/Down. (Long-only for simplicity)
    
    positions = []
    for p in preds:
        if p == 2: # Up
            positions.append(1.0)
        elif p == 0: # Down
            positions.append(0.0) # Cash
        else:
            positions.append(0.0) # Neutral -> Cash (conservative)
            
    positions = np.array(positions)
    
    # Calculate Returns
    # Re-fetch and feature-engineer the raw data to get Log_Return for the test period
    full_df = loader.fetch_batch_data()
    if full_df.empty:
        logger.error("Failed to fetch data for backtest returns calculation.")
        return
    
    # Handle single vs multi-ticker download format
    if isinstance(full_df.columns, pd.MultiIndex):
        df_raw = full_df[TICKER].copy() if TICKER in full_df.columns.get_level_values(0) else full_df.copy()
    else:
        df_raw = full_df.copy()
    
    df_eng = loader.feature_engineering(df_raw, return_raw=True)
    if df_eng.empty:
        logger.error("Feature engineering returned empty DataFrame.")
        return
    
    test_mask = (df_eng.index >= '2024-01-01')
    df_test = df_eng[test_mask].iloc[loader.window_size:] # trimmed by window
    
    if df_test.empty or 'Log_Return' not in df_test.columns:
        logger.error("Test period data is empty or missing Log_Return column.")
        return
    
    market_returns = df_test['Log_Return'].shift(-1).dropna().values
    
    # Align lengths
    n = min(len(positions), len(market_returns))
    positions = positions[:n]
    market_returns = market_returns[:n]
    dates = df_test.index[:n]
    
    strategy_returns = positions * market_returns
    
    # 5. Metrics & Plotting
    # Convert log returns to simple for equity curve
    strat_equity = np.exp(np.cumsum(strategy_returns))
    market_equity = np.exp(np.cumsum(market_returns))
    
    df_res = pd.DataFrame({
        "Strategy": strat_equity,
        "Market (Buy&Hold)": market_equity
    }, index=dates)
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_res.index, df_res['Strategy'], label='AI Agent')
    plt.plot(df_res.index, df_res['Market (Buy&Hold)'], label='Market (AAPL)', alpha=0.6)
    plt.title("Backtest Result: AI vs Market (2024)")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.legend()
    plt.grid(True)
    if not os.path.exists("frontend/public"):
        os.makedirs("frontend/public", exist_ok=True) # Prepare for React
    plt.savefig("frontend/public/backtest_chart.png")
    plt.close()
    
    metrics_strat = calculate_metrics(pd.Series(strategy_returns))
    metrics_mkt = calculate_metrics(pd.Series(market_returns))
    
    logger.info("=== Backtest Results (2024) ===")
    logger.info(f"AI Agent: Return={metrics_strat['Total Return']:.2%}, Sharpe={metrics_strat['Sharpe Ratio']:.2f}, DD={metrics_strat['Max Drawdown']:.2%}")
    logger.info(f"Market:   Return={metrics_mkt['Total Return']:.2%}, Sharpe={metrics_mkt['Sharpe Ratio']:.2f}, DD={metrics_mkt['Max Drawdown']:.2%}")

if __name__ == "__main__":
    run_backtest()
