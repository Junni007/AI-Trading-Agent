import logging
import pandas as pd
from scan_intraday import SniperEngine
from scan_volatility import VolatilityEngine

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MetaBrain')

class HybridBrain:
    """
    The Meta-Model (Brain).
    Aggregates votes from Experts and decides the best course of action.
    """
    
    def __init__(self):
        self.sniper_expert = SniperEngine()
        self.income_expert = VolatilityEngine()
        
    def think(self):
        """
        Runs the Recursive Thinking Process.
        1. Get Volatility Context (Income Expert).
        2. Get Momentum Context (Sniper Expert).
        3. Resolve Conflict.
        """
        logger.info("🧠 Brain is thinking... Querying Experts...")
        
        # 1. Ask Income Expert (Context/Regime)
        income_votes = self.income_expert.run_scan()
        income_map = {v['Ticker']: v for v in income_votes}
        
        # 2. Ask Sniper Expert (Momentum)
        sniper_votes = self.sniper_expert.run_scan()
        sniper_map = {v['Ticker']: v for v in sniper_votes}
        
        final_decisions = []
        
        # We process the union of tickers found by either expert
        all_tickers = set(income_map.keys()).union(set(sniper_map.keys()))
        
        for t in all_tickers:
            income = income_map.get(t, {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'N/A'})
            sniper = sniper_map.get(t, {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'N/A'})
            
            # --- Thinking Logic ---
            decision = {
                'Ticker': t,
                'Action': 'WAIT',
                'Confidence': 0.0,
                'Rational': []
            }
            
            # A. High Volatility Regime (Income Expert Dominates)
            if income['Signal'] == 'INCOME':
                decision['Rational'].append(f"Regime: High Volatility ({income['Confidence']:.2f})")
                
                # Check if Sniper agrees (Momentum is huge?)
                if sniper['Signal'] == 'BUY':
                    # Conflict: High Vol but Bullish Momentum?
                    # Result: Bullish Put Spread (Defined Risk) instead of Naked Calls
                    decision['Action'] = 'BULL_PUT_SPREAD'
                    decision['Confidence'] = (income['Confidence'] + sniper['Confidence']) / 2
                    decision['Rational'].append(f"Solution: Hybrid. High Vol + Bullish Momentum -> Credit Spread.")
                else:
                    # Pure Income
                    decision['Action'] = 'IRON_CONDOR'
                    decision['Confidence'] = income['Confidence']
                    decision['Rational'].append("Solution: Pure Volatility Play (Sell Neutral Premium).")
            
            # B. Low Volatility Regime (Sniper Prep)
            elif income['Signal'] == 'SNIPER_PREP':
                 decision['Rational'].append(f"Regime: Low Volatility (Coiled).")
                 
                 if sniper['Signal'] == 'BUY':
                     # The Perfect Storm: Low Vol + Breakout
                     decision['Action'] = 'LONG_CALL_SNIPER'
                     decision['Confidence'] = max(income['Confidence'], sniper['Confidence']) + 0.1 # Boost!
                     decision['Rational'].append("Solution: PERFECT SETUP. Vol Expansion + Trend.")
                 else:
                     # Waiting for the move
                     decision['Action'] = 'WATCH_FOR_BREAKOUT'
                     decision['Confidence'] = 0.5
                     decision['Rational'].append("Solution: Stalking. Vol is low, waiting for Sniper Trigger.")
            
            # C. Normal Regime (Sniper leads)
            else:
                 if sniper['Signal'] == 'BUY':
                     decision['Action'] = 'LONG_STOCK'
                     decision['Confidence'] = sniper['Confidence']
                     decision['Rational'].append(f"Regime: Normal. Following Momentum.")
                 else:
                     continue # Nothing interesting
            
            final_decisions.append(decision)
            
        return final_decisions

if __name__ == "__main__":
    brain = HybridBrain()
    thoughts = brain.think()
    
    print("\n" + "="*90)
    print("🧠 THE THINKING ENGINE REPORT (Hybrid Consensus) 🧠")
    print("="*90)
    
    if not thoughts:
        print("Brain Verdict: Market is efficient/noisy. No high-confidence Setups.")
    else:
        print(f"{'Ticker':<8} | {'Final Action':<20} | {'Conf':<6} | {'Thinking Process'}")
        print("-" * 90)
        # Sort by Confidence
        thoughts.sort(key=lambda x: x['Confidence'], reverse=True)
        
        for t in thoughts:
            reason = " | ".join(t['Rational'])
            print(f"{t['Ticker']:<8} | {t['Action']:<20} | {t['Confidence']:.2f}   | {reason}")
            
    print("="*90 + "\n")
