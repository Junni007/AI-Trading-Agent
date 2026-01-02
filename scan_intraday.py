import pandas as pd
import logging
from src.data_loader_intraday import IntradayDataLoader

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SniperExpert')

class SniperEngine:
    """
    Expert 1: Intraday Momentum Scanner ("The Sniper").
    Logic: VWAP + RSI + Volume Z-Score.
    """
    
    def __init__(self):
        self.loader = IntradayDataLoader()
        # Focusing on High Beta / High Liquid stocks for Intraday
        self.universe = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NFLX', 'QCOM']
        
    def get_vote(self, ticker: str, df: pd.DataFrame) -> dict:
        """
        Analyzes the latest candle to generate a Vote.
        Returns: {Signal, Confidence, Reason}
        """
        if df is None or df.empty:
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'No Data'}
            
        last_row = df.iloc[-1]
        
        # Features
        price = last_row['Close']
        vwap = last_row['VWAP']
        rsi = last_row['RSI']
        vol_z = last_row['Vol_Z']
        
        # Logic: Bullish Sniper
        # 1. Price > VWAP (Institutional Support)
        # 2. RSI > 55 (Momentum Picking Up) but < 75 (Not Exhausted)
        # 3. Volume > 1.5 Std Dev (Volume Spike)
        
        score = 0
        reasons = []
        
        if price > vwap:
            score += 1
            reasons.append("Price > VWAP")
        
        if 55 < rsi < 75:
            score += 1
            reasons.append(f"RSI Bullish ({rsi:.1f})")
        elif rsi < 30:
            # Reversal Sniper?
            score += 0.5
            reasons.append(f"RSI Oversold ({rsi:.1f})")
            
        if vol_z > 1.0:
            score += 1
            reasons.append(f"Volume Spike (Z={vol_z:.1f})")
            
        # Decision
        if score >= 3:
            return {
                'Signal': 'BUY',
                'Confidence': 0.85, # High Conviction
                'Reason': " + ".join(reasons)
            }
        elif score >= 2:
            return {
                'Signal': 'BUY',
                'Confidence': 0.60, # Moderate
                'Reason': " + ".join(reasons)
            }
        else:
            return {
                'Signal': 'NEUTRAL',
                'Confidence': 0.0, 
                'Reason': "Wait for setup"
            }

    def run_scan(self):
        """
        Scans values and returns a Report List.
        """
        results = []
        logger.info(f"Scanning {len(self.universe)} tickers for Sniper Setups (15m)...")
        
        for t in self.universe:
            df = self.loader.fetch_data(t, interval='15m')
            df = self.loader.add_technical_indicators(df)
            
            vote = self.get_vote(t, df)
            
            if vote['Signal'] != 'NEUTRAL':
                results.append({
                    'Ticker': t,
                    'Signal': vote['Signal'],
                    'Confidence': vote['Confidence'],
                    'Reason': vote['Reason'],
                    'Price': df.iloc[-1]['Close']
                })
                
        return results

if __name__ == "__main__":
    bot = SniperEngine()
    opportunities = bot.run_scan()
    
    print("\n" + "="*60)
    print("🔭 SNIPER ENGINE OUTPUT (Expert 1) 🔭")
    print("="*60)
    if not opportunities:
        print("No active Sniper setups found right now.")
    else:
        print(f"{'Ticker':<8} | {'Signal':<6} | {'Conf':<6} | {'Reason'}")
        print("-" * 60)
        for op in opportunities:
            print(f"{op['Ticker']:<8} | {op['Signal']:<6} | {op['Confidence']:<6} | {op['Reason']}")
    print("="*60 + "\n")
