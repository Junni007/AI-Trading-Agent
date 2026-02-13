"""
Enhanced version of backtest metrics calculation.

Adds: Sortino Ratio, Win Rate, Profit Factor, Calmar Ratio
"""

import pandas as pd
import numpy as np

def calculate_metrics_enhanced(returns):
    """
    Calculates comprehensive performance metrics.
    
    Args:
        returns: pandas Series of returns (daily)
    
    Returns dict with:
        - Total Return
        - Sharpe Ratio
        - Sortino Ratio (downside-adjusted)
        - Max Drawdown
        - Win Rate
        - Profit Factor  
        - Calmar Ratio
    """
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    # Sharpe (assuming daily)
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
    
    # Sortino Ratio (penalize downside volatility only)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 1e-9
    sortino = returns.mean() / (downside_std + 1e-9) * np.sqrt(252)
    
    # Drawdown
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win Rate
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0
    
    # Profit Factor
    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0.0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 0.0
    profit_factor = gross_profit / (gross_loss + 1e-9)
    
    # Calmar Ratio (Return / Max Drawdown)
    calmar = total_return / abs(max_drawdown + 1e-9)
    
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Calmar Ratio": calmar
    }

# Example usage:
if __name__ == "__main__":
    # Test with sample data
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    metrics = calculate_metrics_enhanced(sample_returns)
    
    print("=== Enhanced Metrics ===")
    for key, value in metrics.items():
        if "Rate" in key or "Return" in key or "Drawdown" in key:
            print(f"{key:15}: {value:.2%}")
        else:
            print(f"{key:15}: {value:.2f}")
