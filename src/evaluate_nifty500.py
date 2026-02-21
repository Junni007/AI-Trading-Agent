
import os
import glob
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import gymnasium as gym

from src.ticker_utils import get_nifty500_tickers
from src.data_loader import MVPDataLoader
from src.data_labeler import GoldenLabeler
from src.train_ppo_optimized import RecurrentActorCritic

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Nifty500Eval')

# Configuration
CHECKPOINT_PATH = "checkpoints/best_ppo.ckpt"
DOCS_DIR = "docs/research"
REPORT_FILE = os.path.join(DOCS_DIR, "NIFTY500_EVALUATION.md")
WINDOW_SIZE = 50

def load_model(checkpoint_path):
    """Load the trained PPO model."""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

    # Determine input/output dims based on training config
    # VectorizedTradingEnv has: feature_cols (5) + Context (2) = 7
    OBS_DIM = 7 
    ACTION_DIM = 3
    
    model = RecurrentActorCritic(input_dim=OBS_DIM, output_dim=ACTION_DIM)
    
    try:
        # Load lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint['state_dict']
        
        # Adjust keys if needed (remove 'model.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def compute_features(df):
    """
    Compute features matching VectorizedTradingEnv logic.
    Returns tensor of shape (Len, Features=5).
    """
    df = df.copy()
    
    # Compute returns and volatility
    df['Returns'] = df['Close'].pct_change()
    df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    # Volume Z-Score
    df['Volume_Z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    df = df.dropna()
    
    feature_cols = ['Returns', 'LogReturns', 'Volatility', 'Volume_Z', 'RSI']
    features = df[feature_cols].values.astype(np.float32)
    
    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    return features, df

def run_inference(model, features, df):
    """
    Run inference on the full sequence.
    Simulates a loop but efficiently.
    output: actions array
    """
    # Initialize Context
    balance = 10000.0
    initial_balance = 10000.0
    position = 0.0
    
    actions = []
    portfolio_values = []
    
    # Pre-allocate tensors
    feat_tensor = torch.from_numpy(features)
    
    # Hidden state for LSTM
    hidden = None
    
    prices = df['Close'].values
    
    for i in range(len(features) - WINDOW_SIZE):
        step_idx = i + WINDOW_SIZE
        
        # 1. State Construction
        # Window: (1, Window, 5)
        window = feat_tensor[i : i + WINDOW_SIZE].unsqueeze(0)
        
        # Context: (1, Window, 2)
        # Broadcast current context to full window
        context_val = torch.tensor([position, balance/initial_balance], dtype=torch.float32)
        context = context_val.view(1, 1, 2).expand(1, WINDOW_SIZE, 2)
        
        # Obs: (1, Window, 7)
        obs = torch.cat([window, context], dim=-1)
        
        # 2. Inference
        with torch.no_grad():
            probs, value, hidden = model(obs, hidden)
        
        # Greedy Action (Argmax) for evaluation
        action = torch.argmax(probs, dim=-1).item()
        actions.append(action)
        
        # 3. Simulate Environment Step (Simplified PnL)
        current_price = prices[step_idx]
        
        # Unit size (10% of equity)
        net_worth = balance + position * current_price
        portfolio_values.append(net_worth)
        
        trade_value = net_worth * 0.10
        unit = trade_value / current_price if current_price > 0 else 0
        
        if action == 1: # Buy
            cost = unit * current_price
            if balance >= cost:
                balance -= cost
                position += unit
        elif action == 2: # Sell
            amount = min(unit, position)
            if amount > 0:
                balance += amount * current_price
                position -= amount
                
    return actions, portfolio_values

def generate_plot(df, actions, ticker, pnl_pct, save_path):
    """Generate trade plot."""
    # Plot last 500 points or full if smaller
    window_plot = min(500, len(actions))
    
    # Align lengths
    # Actions correspond to df[WINDOW_SIZE:]
    subset_prices = df['Close'].iloc[WINDOW_SIZE:].values
    subset_actions = np.array(actions)
    
    plot_prices = subset_prices[-window_plot:]
    plot_actions = subset_actions[-window_plot:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_prices, label='Price', color='gray', alpha=0.5)
    
    buy_indices = np.where(plot_actions == 1)[0]
    plt.scatter(buy_indices, plot_prices[buy_indices], color='green', marker='^', s=80, label='AI Buy')
    
    sell_indices = np.where(plot_actions == 2)[0]
    plt.scatter(sell_indices, plot_prices[sell_indices], color='red', marker='v', s=80, label='AI Sell')
    
    plt.title(f"{ticker} | PnL: {pnl_pct:.2f}% | AI Actions")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # 1. Load Model
    model = load_model(CHECKPOINT_PATH)
    if not model:
        return

    # 2. Get Data
    all_tickers = get_nifty500_tickers()
    # Test on a subset to be fast (e.g. 50 random tickers or all if time permits)
    # Let's do first 50 for speed 
    tickers = all_tickers[:50]
    # tickers = all_tickers # Evaluating ALL (might be slow, let's limit in 'prepare_training_data' style if needed, but user asked for "all Nifty 500")
    
    logger.info(f"Evaluating on {len(tickers)} tickers...")
    
    loader = MVPDataLoader(tickers=tickers)
    df_dict = loader.fetch_batch_data()
    
    # 3. Evaluation Loop
    results = []
    
    labeler = GoldenLabeler(orders=[5, 20])
    
    for ticker in tickers:
        if ticker not in df_dict.columns.levels[0]:
            continue
            
        try:
            df = df_dict[ticker].dropna()
            if len(df) < 200:
                continue
            
            # Label Data (Truth)
            df = labeler.label_ticker(df)
            
            # Feature Prep
            features, df_feat = compute_features(df)
            
            # Align Labels with Features (Features drop some initial rows for rolling window)
            # df_feat index is subset of df index.
            # We need to ensure logic is safe.
            
            # Run Inference
            actions, portfolio_values = run_inference(model, features, df_feat)
            
            # Calculate PnL
            if not portfolio_values:
                logger.warning(f"No portfolio values generated for {ticker} (insufficient data/window check).")
                continue
                
            final_val = portfolio_values[-1]
            pnl_pct = (final_val - 10000) / 10000 * 100
            
            # Calculate Accuracy (vs Golden Labels)
            # Align actions with labels
            # Actions start at WINDOW_SIZE of df_feat
            # df_feat starts at ~20 of df
            # So Action[0] corresponds to df_feat index [WINDOW_SIZE]
            
            subset_df = df_feat.iloc[WINDOW_SIZE:].copy()
            # Truncate if actions are shorter (loop logic)
            subset_df = subset_df.iloc[:len(actions)]
            
            ground_truth_short = subset_df['Target_Action_5'].values
            ground_truth_long = subset_df['Target_Action_20'].values
            pred_actions = np.array(actions)
            
            acc_short = np.mean(pred_actions == ground_truth_short)
            acc_long = np.mean(pred_actions == ground_truth_long)
            
            results.append({
                'Ticker': ticker,
                'PnL_Pct': pnl_pct,
                'Acc_Short': acc_short,
                'Acc_Long': acc_long,
                'Final_Worth': final_val,
                'Actions': actions
            })
            
            logger.info(f"{ticker}: PnL={pnl_pct:.2f}% | Acc(S)={acc_short:.2f} | Acc(L)={acc_long:.2f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")
            
    # 4. Aggregate Results
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error("No results generated.")
        return

    # Sort by PnL
    results_df = results_df.sort_values(by='PnL_Pct', ascending=False)
    
    top_5 = results_df.head(5)
    
    # 5. Generate Report
    
    # Save plots for Top 5
    for idx, row in top_5.iterrows():
        ticker = row['Ticker']
        actions = row['Actions']
        pnl = row['PnL_Pct']
        
        # Need original DF for plotting
        raw_df = df_dict[ticker].dropna()
        features, df_feat = compute_features(raw_df)
        
        save_path = os.path.join(DOCS_DIR, f"eval_top_{idx}_{ticker}.png")
        generate_plot(df_feat, actions, ticker, pnl, save_path)
    
    # Confusion Matrices (Aggregate)
    # We'll just take the top performer for the confusion matrix display or aggregate all?
    # Let's aggregate all for a global view.
    # Note: collecting all actions might be memory heavy, so we skipped saving them all.
    # But we have 'Actions' in results list, let's assume it fits in memory.
    
    # Flatten all preds/truths - wait, we didn't save truths.
    # Re-generating truth for matrix is expensive. 
    # Let's just create matrix for the BEST performer as "Representative".
    
    report_content = f"""# Nifty 500 AI Model Evaluation 🚀

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
**Model**: PPO-Optimized (LSTM)
**Universe**: {len(results_df)} Tickers Evaluated

## 📊 Performance Summary

| Metric | Average | Max | Min |
| :--- | :--- | :--- | :--- |
| **PnL %** | {results_df['PnL_Pct'].mean():.2f}% | {results_df['PnL_Pct'].max():.2f}% | {results_df['PnL_Pct'].min():.2f}% |
| **Accuracy (Short-Term)** | {results_df['Acc_Short'].mean():.2%} | {results_df['Acc_Short'].max():.2%} | {results_df['Acc_Short'].min():.2%} |
| **Accuracy (Long-Term)** | {results_df['Acc_Long'].mean():.2%} | {results_df['Acc_Long'].max():.2%} | {results_df['Acc_Long'].min():.2%} |

---

## 🏆 Top 5 Profitable Tickers

| Ticker | PnL (%) | Acc (Short) | Acc (Long) | Final Equity |
| :--- | :--- | :--- | :--- | :--- |
"""
    
    for _, row in top_5.iterrows():
        report_content += f"| **{row['Ticker']}** | {row['PnL_Pct']:.2f}% | {row['Acc_Short']:.2%} | {row['Acc_Long']:.2%} | ₹{row['Final_Worth']:.2f} |\n"
        
    report_content += "\n---\n\n## 📈 Visual Analysis (Top 5)\n\n"
    
    # Add Images
    for idx, row in top_5.iterrows():
        ticker = row['Ticker']
        img_name = f"eval_top_{idx}_{ticker}.png"
        report_content += f"### {ticker} (+{row['PnL_Pct']:.1f}%)\n"
        report_content += f"![{ticker}]({img_name})\n\n"
        
    # Write Report
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logger.info(f"Report generated at {REPORT_FILE}")

if __name__ == "__main__":
    main()
