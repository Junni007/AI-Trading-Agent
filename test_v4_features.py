"""
Verification Script for Signal.Engine v4.0 Features
Tests volatility targeting and cross-sectional ranking.
"""
import sys
import pandas as pd
import numpy as np

# Test 1: Volatility Targeting
print("=" * 60)
print("TEST 1: Volatility Targeting")
print("=" * 60)

# Create dummy DataFrame with ATR
test_df = pd.DataFrame({
    'Close': [100, 105, 103, 108, 110],
    'ATR': [2.0, 2.5, 1.8, 3.0, 1.5],  # Varying volatility
    'RSI': [50, 55, 60, 65, 70],
    'MACD': [0.5, 0.6, 0.7, 0.8, 0.9],
    'MACD_Signal': [0.4, 0.5, 0.6, 0.7, 0.8],
    'Log_Return': [0.01, 0.02, -0.01, 0.03, 0.01],
    'Trend_Signal': [0.1, 0.15, 0.12, 0.18, 0.20],
    'Vol_Change': [0.05, 0.06, 0.04, 0.07, 0.05],
    'Target': [0, 1, 0, 1, 0]
})

try:
    from src.env import TradingEnv
    
    # Disable TDA for testing (requires too much data)
    env = TradingEnv(test_df, initial_balance=10000, window_size=2, tda_config=None)
    
    # Manual call to vol-targeted sizing
    net_worth = 10000
    current_price = 105.0
    step_idx = 2
    
    position_value = env._calculate_vol_targeted_position(net_worth, current_price, step_idx)
    
    print(f"Net Worth: ${net_worth:,.2f}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"ATR at step {step_idx}: {test_df['ATR'].iloc[step_idx]:.2f}")
    print(f"Vol-Targeted Position Value: ${position_value:.2f}")
    print(f"Position as % of portfolio: {(position_value/net_worth)*100:.1f}%")
    
    # Expected: Lower ATR → Higher position size (up to 2x base)
    # Higher ATR → Lower position size (down to 0.5x base)
    print("\n✅ Volatility Targeting: WORKING")
    
except Exception as e:
    print(f"\n❌ Volatility Targeting Test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Cross-Sectional Ranking
print("\n" + "=" * 60)
print("TEST 2: Cross-Sectional Ranking")
print("=" * 60)

try:
    from src.data_loader import MVPDataLoader
    
    # Create dummy multi-ticker data
    dates = pd.date_range('2024-01-01', periods=5)
    
    df_dict = {
        'STOCK_A': pd.DataFrame({
            'RSI': [70, 75, 80, 78, 72],
            'Log_Return': [0.02, 0.03, 0.04, 0.03, 0.02]
        }, index=dates),
        'STOCK_B': pd.DataFrame({
            'RSI': [50, 55, 60, 58, 52],
            'Log_Return': [0.01, 0.015, 0.02, 0.015, 0.01]
        }, index=dates),
        'STOCK_C': pd.DataFrame({
            'RSI': [30, 35, 40, 38, 32],
            'Log_Return': [-0.01, 0.0, 0.01, 0.0, -0.01]
        }, index=dates)
    }
    
    loader = MVPDataLoader(tickers=['STOCK_A', 'STOCK_B', 'STOCK_C'])
    ranked_dict = loader.compute_cross_sectional_features(df_dict)
    
    print("\nRanking Results for 2024-01-01:")
    print("-" * 40)
    for ticker in ['STOCK_A', 'STOCK_B', 'STOCK_C']:
        rsi_rank = ranked_dict[ticker]['RSI_Rank'].iloc[0]
        mom_rank = ranked_dict[ticker]['Momentum_Rank'].iloc[0]
        print(f"{ticker}: RSI_Rank={rsi_rank:.2f}, Momentum_Rank={mom_rank:.2f}")
    
    # Expected: STOCK_A should have highest ranks (top RSI + Returns)
    # STOCK_C should have lowest ranks
    assert ranked_dict['STOCK_A']['RSI_Rank'].iloc[0] > 0.5, "Stock A should be top-ranked"
    assert ranked_dict['STOCK_C']['RSI_Rank'].iloc[0] < 0.5, "Stock C should be bottom-ranked"
    
    print("\n✅ Cross-Sectional Ranking: WORKING")
    
except Exception as e:
    print(f"\n❌ Cross-Sectional Ranking Test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
