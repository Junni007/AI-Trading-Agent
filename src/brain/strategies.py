import pandas as pd
import numpy as np
import logging
from src.ticker_utils import get_extended_tickers
from src.data_loader import MVPDataLoader
from src.patterns import CandlestickDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScanStrategies")

def strategy_hammer_reversal(df):
    """
    Rule: Buy if Hammer AND RSI < 40 (Oversold) AND Up Trend (Price > EMA50)
    Note: Trend filter checks if we are generally healthy, RSI checks for a dip.
    """
    return df['Pattern_Hammer'] & (df['RSI'] < 45) & (df['Close'] > df['EMA_50'])

def strategy_engulfing_trend(df):
    """
    Rule: Buy if Bullish Engulfing AND Price > EMA50.
    """
    return df['Pattern_Engulfing'] & (df['Close'] > df['EMA_50'])

def strategy_simple_dip(df):
    """
    Rule: Buy if RSI < 30 (Deep Oversold) - Simple Mean Reversion.
    """
    return df['RSI'] < 30

def strategy_golden_cross(df):
    """
    Long Term Rule: Price > EMA 50 > EMA 200.
    Standard 'Investor' signal for steady growth.
    """
    # Calculate EMA 200 locally since it wasn't in loader default
    ema_200 = df['Close'].ewm(span=200, adjust=False).mean()
    return (df['Close'] > df['EMA_50']) & (df['EMA_50'] > ema_200)

def scan_strategies():
    logger.info("Loading Data...")
    tickers = get_extended_tickers(limit=500000)
    # Using window_size=50 just to init standard loader features
    loader = MVPDataLoader(tickers=tickers, window_size=50) 
    
    # We want the RAW DateFrames, not the X/y sequences.
    # So we use fetch_batch_data + process_single_ticker_data logic manually
    full_df = loader.fetch_batch_data()
    
    results = []
    
    # Iterate Tickers
    is_multi = isinstance(full_df.columns, pd.MultiIndex)
    
    total_trades = 0
    
    strategies = {
        "Hammer + RSI Dip": strategy_hammer_reversal,
        "Bullish Engulfing Trend": strategy_engulfing_trend,
        "RSI < 30 (Pure)": strategy_simple_dip,
        "Golden Cross (Long Term)": strategy_golden_cross
    }
    
    # Init stats
    stats = {name: {'wins': 0, 'total': 0, 'returns': []} for name in strategies}
    recent_opportunities = []
    
    # Scale Up: Use ALL available tickers (approx 550)
    logger.info("Scanning Strategies on ALL tickers...")
    
    for t in tickers:
        try:
            if is_multi:
                if t in full_df.columns.get_level_values(0):
                    df = full_df[t].copy()
                else:
                    continue
            else:
                if len(tickers) == 1: df = full_df.copy()
                else: continue
            
            # Debug: Check columns if 'Open' missing
            if 'Open' not in df.columns:
                logger.warning(f"Ticker {t} columns: {df.columns.tolist()}")
                # Try simple fix if columns are MultiIndex inside
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(0) # Maybe it's (Ticker, Price) inside?
            
            # 1. Feature Engineering (EMA, RSI, Patterns)
            # FIX: return_raw=True to keep Open/High/Low for Pattern Detection
            df = loader.feature_engineering(df, return_raw=True)
            if df.empty: continue
            
            # Add Patterns
            df = CandlestickDetector.add_patterns(df)
            
            # 2. Backtest Logic (Managed Trade)
            # Core Problem: Close-to-Close ignores intraday potential.
            # Solution: Check if High > TP before Low < SL.
            
            # Assumptions for "Sniper" Mode:
            # TP = 2.0% (0.02)
            # SL = 1.0% (0.01)
            # Ratio = 2:1
            
            next_open = df['Open'].shift(-1)
            next_high = df['High'].shift(-1)
            next_low = df['Low'].shift(-1)
            next_close = df['Close'].shift(-1)
            
            # Entry usually at Close today (or Open tomorrow). Let's assume Close Today (Signal Trigger).
            entry_price = df['Close']
            
            # Vectorized Logic for Managed Outcome
            tp_price = entry_price * 1.02
            sl_price = entry_price * 0.99
            
            # Did we hit SL?
            hit_sl = next_low < sl_price
            # Did we hit TP?
            hit_tp = next_high > tp_price
            
            # Logic: 
            # If hit_tp AND NOT hit_sl: Win
            # If hit_sl AND NOT hit_tp: Loss
            # If Both: Ambiguous (Volatile). Assume Loss (Conservative) or Stopped Out first.
            #     (Realistically need minute data to know which happened first, but we assume SL hit first for safety).
            # If Neither: Trade held to Close.
            
            # Outcome: 1 (Win), -1 (Loss), 0 (Neutral/Hold)
            # Vectorize:
            # Win: (High >= TP) & (Low > SL) -> Ideal Win
            # Loss: (Low <= SL)
            # Neutral: Else (Close trade at End of Day)
            
            managed_outcome = pd.Series(0, index=df.index)
            # Wins: Reached TP and didn't touch SL
            managed_outcome[(next_high >= tp_price) & (next_low > sl_price)] = 1 
            # Losses: Touched SL
            managed_outcome[next_low <= sl_price] = -1
            # Ambiguous (Both Hit): Treat as Loss (Stopped Out)
            managed_outcome[(next_high >= tp_price) & (next_low <= sl_price)] = -1 
            
            # Returns for stats
            # Win = +2%
            # Loss = -1%
            # Neutral = (Next Close - Entry) / Entry
            
            for name, strategy_func in strategies.items():
                signals = strategy_func(df)
                
                # Filter valid signals (drop last row nan)
                valid_indices = signals & next_close.notna()
                outcomes = managed_outcome[valid_indices]
                
                if len(outcomes) > 0:
                    # Calculate Stats
                    wins = (outcomes == 1).sum()
                    losses = (outcomes == -1).sum()
                    neutrals = (outcomes == 0).sum()
                    
                    # Neutral Returns
                    neutral_indices = valid_indices & (managed_outcome == 0)
                    neutral_rets = (next_close[neutral_indices] - entry_price[neutral_indices]) / entry_price[neutral_indices]
                    
                    # Total PnL points (Approx)
                    total_pnl = (wins * 0.02) + (losses * -0.01) + neutral_rets.sum()
                    avg_pnl = total_pnl / len(outcomes)
                    
                    stats[name]['wins'] += wins
                    stats[name]['total'] += len(outcomes)
                    stats[name]['returns'].append(avg_pnl) # storing avg of batch, simpler
                    stats[name]['pnl_sum'] = stats[name].get('pnl_sum', 0.0) + total_pnl
                    
            # Store Recent Signals for "Live Opportunities" Report
            # Get last 5 days
            recent_df = df.iloc[-5:].copy()
            
            for name, strategy_func in strategies.items():
                signals = strategy_func(recent_df)
                active_days = recent_df[signals]
                
                for date, row in active_days.iterrows():
                    # Generate Rationale
                    reason = []
                    if name == "Hammer + RSI Dip":
                        reason.append("Bullish Hammer pattern detected")
                        reason.append(f"RSI is oversold ({row['RSI']:.1f})")
                        reason.append("Price above EMA50 Trend")
                    elif name == "Bullish Engulfing Trend":
                        reason.append("Bullish Engulfing pattern (Strong Reversal)")
                        reason.append("Confirmed by Up Trend (Above EMA50)")
                    elif name == "RSI < 30 (Pure)":
                        reason.append(f"Deep Oversold Condition (RSI {row['RSI']:.1f})")
                        reason.append("Mean Reversion Potential")
                    elif name == "Golden Cross (Long Term)":
                        reason.append("Golden Cross (EMA50 > EMA200)")
                        reason.append("Long-term Bullish Trend confirmed")
                        
                    recent_opportunities.append({
                        'Ticker': t,
                        'Date': date.date(),
                        'Strategy': name,
                        'Price': row['Close'],
                        'Rationale': " + ".join(reason)
                    })
                    
        except Exception as e:
            logger.error(f"Error on {t}: {e}")
            continue

    print("\n" + "="*70)
    print("📊 MANAGED STRATEGY RESULTS (TP=+2%, SL=-1%) 📊")
    print("="*70)
    print(f"{'Strategy':<30} | {'Win Rate':<10} | {'Trades':<8} | {'Avg PnL':<10}")
    print("-" * 70)
    
    for name, data in stats.items():
        total = data['total']
        if total == 0:
            print(f"{name:<30} | {'N/A':<10} | {0:<8} | {'N/A':<10}")
        else:
            win_rate = (data['wins'] / total) * 100
            avg_pnl = data.get('pnl_sum', 0.0) / total
            print(f"{name:<30} | {win_rate:.2f}%     | {total:<8} | {avg_pnl:.5f}")
            
    print("="*70)
    
    # Print Recent Opportunities
    print("\n" + "="*80)
    print(f"🚀 RECENT OPPORTUNITIES (Last 5 Days) - WHY YOU SHOULD TRADE 🚀")
    print("="*80)
    if not recent_opportunities:
        print("No signals found in the last 5 days.")
    else:
        print(f"{'Ticker':<8} | {'Date':<12} | {'Strategy':<25} | {'Rationale'}")
        print("-" * 80)
        # Sort by Date descending
        recent_opportunities.sort(key=lambda x: x['Date'], reverse=True)
        
        for op in recent_opportunities[:20]: # Show top 20 latest
            print(f"{op['Ticker']:<8} | {str(op['Date']):<12} | {op['Strategy']:<25} | {op['Rationale']}")
            
    print("="*80 + "\n")

if __name__ == "__main__":
    scan_strategies()
