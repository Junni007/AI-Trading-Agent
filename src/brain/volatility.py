import pandas as pd
import numpy as np
import logging
from src.data_loader import MVPDataLoader
from src.ticker_utils import get_extended_tickers

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IncomeExpert')

class VolatilityEngine:
    """
    Expert 2: Volatility "Income" Scanner.
    Logic: Uses Historical Volatility (HV) Rank to detect Regimes.
    
    Why HV? Real Options IV data is expensive/sparse. HV is a robust proxy.
    - High HV Rank (>80%): Expect Mean Reversion -> Sell Premium (Income).
    - Low HV Rank (<20%): Expect Expansion -> Buy Premium (Sniper).
    """
    
    def __init__(self):
        self.loader = MVPDataLoader()
        # Full Indian Market
        from src.ticker_utils import get_nifty500_tickers
        self.universe = ["^NSEI", "^NSEBANK"] + get_nifty500_tickers()
        
    def calculate_hv_rank(self, df: pd.DataFrame, window: int = 20) -> dict:
        """
        Calculates HV Rank (0-100).
        HV = Annualized Std Dev of Log Returns.
        Rank = Percentile of current HV over last 252 days.
        """
        if len(df) < 260: return None
        
        # 1. Log Returns
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Realized Volatility (Annualized)
        # Window = 20 (approx 1 month)
        df['HV'] = df['Log_Ret'].rolling(window=window).std() * np.sqrt(252) * 100
        
        current_hv = df['HV'].iloc[-1]
        
        # 3. HV Rank (Percentile over last year)
        last_year_hv = df['HV'].iloc[-252:]
        rank = (last_year_hv < current_hv).mean() * 100
        
        return {
            'Current_HV': current_hv,
            'HV_Rank': rank,
            'High_HV': last_year_hv.max(),
            'Low_HV': last_year_hv.min()
        }

    def get_vote(self, ticker: str, df: pd.DataFrame) -> dict:
        """
        Generates a Vote based on Volatility Regime.
        """
        stats = self.calculate_hv_rank(df)
        if not stats:
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Insufficient Data'}
            
        rank = stats['HV_Rank']
        
        # Logic:
        # A. High Volatility Regime (>80th percentile)
        # Why? Volatility makes mean-reverting.
        # Action: INCOME (Sell Premium)
        if rank > 80:
            return {
                'Signal': 'INCOME', # Sell Premium
                'Confidence': 0.8 + ((rank - 80) / 100), # Higher rank = Higher conf
                'Reason': f"High Volatility (Rank {rank:.0f}%) -> Sell Premium"
            }
            
        # B. Low Volatility Regime (<20th percentile)
        # Why? Volatility is coiled. Expect explosion.
        # Action: SNIPER (Buy Premium / Breakout)
        elif rank < 20:
            return {
                'Signal': 'SNIPER_PREP', # Buy Premium
                'Confidence': 0.70,
                'Reason': f"Low Volatility (Rank {rank:.0f}%) -> Expect Compression Breakout"
            }
            
        # C. Neutral Regime (20-80)
        else:
            return {
                'Signal': 'NEUTRAL',
                'Confidence': 0.0,
                'Reason': f"Normal Volatility (Rank {rank:.0f}%)"
            }

    def run_scan(self):
        """
        Scans universe for Volatility Regimes.
        """
        results = []
        logger.info(f"Scanning {len(self.universe)} tickers for Income/Vol Setups...")
        
        # Initialize Loader with Universe
        loader = MVPDataLoader(tickers=self.universe)
        full_df = loader.fetch_batch_data()
        
        # Check if MultiIndex (Ticker, Price) or just (Price,)
        is_multi = isinstance(full_df.columns, pd.MultiIndex)
        
        for t in self.universe:
            try:
                # Extract Ticker Data
                if is_multi:
                    if t in full_df.columns.get_level_values(0):
                        df = full_df[t].copy()
                    else:
                        continue # Ticker not in download
                else:
                    # Single ticker case
                    if len(self.universe) == 1: df = full_df.copy()
                    else: continue
                
                # Validation
                if df.empty or 'Close' not in df.columns: continue

                vote = self.get_vote(t, df)
                
                # Always append result (for Search visibility)
                results.append({
                    'Ticker': t,
                    'Signal': vote['Signal'],
                    'Confidence': vote['Confidence'],
                    'Reason': vote['Reason'],
                    'HV_Rank': f"{vote.get('Reason').split('(')[1].split(')')[0]}" if '(' in vote['Reason'] else 'N/A'
                })
            except Exception as e:
                logger.error(f"Error scanning {t}: {e}")
                continue
                
        return results

if __name__ == "__main__":
    bot = VolatilityEngine()
    opportunities = bot.run_scan()
    
    print("\n" + "="*70)
    print("💰 INCOME ENGINE OUTPUT (Expert 2 - Volatility) 💰")
    print("="*70)
    if not opportunities:
        print("No extreme volatility regimes found (Market is Normal).")
    else:
        print(f"{'Ticker':<8} | {'Signal':<12} | {'Conf':<6} | {'Reason'}")
        print("-" * 70)
        for op in opportunities:
            print(f"{op['Ticker']:<8} | {op['Signal']:<12} | {op['Confidence']:.2f}   | {op['Reason']}")
    print("="*70 + "\n")
