import logging
import pandas as pd
from src.brain.intraday import SniperEngine
from src.brain.volatility import VolatilityEngine
from src.brain.rl_expert import RLExpert
from src.brain.quant_expert import QuantExpert

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MetaBrain')

class HybridBrain:
    """
    The Meta-Model (Brain).
    Aggregates votes from Experts and decides the best course of action.
    Now enriched with RLExpert and QuantExpert (Funnel Architecture).
    """
    
    def __init__(self):
        # Level 1: Fast Scanners
        self.sniper_expert = SniperEngine()
        self.income_expert = VolatilityEngine()
        self.rl_expert = RLExpert()
        
        # Level 2: Deep Quant
        self.quant_expert = QuantExpert(num_paths=5000) # 5k paths for speed vs accuracy balance
        
    def think(self):
        """
        Runs the Recursive Thinking Process (The Funnel).
        1. Level 1: Get Votes from Sniper, Income, and RL.
        2. Shortlist top candidates.
        3. Level 2: Run Quant Simulation on Shortlist.
        4. Final Decision.
        """
        logger.info("🧠 Brain is thinking... Querying Experts (Level 1)...")
        
        # --- Level 1: Fast Scan ---
        income_votes = self.income_expert.run_scan()
        income_map = {v['Ticker']: v for v in income_votes}
        
        sniper_votes = self.sniper_expert.run_scan()
        sniper_map = {v['Ticker']: v for v in sniper_votes}
        
        final_decisions = []
        
        all_tickers = set(income_map.keys()).union(set(sniper_map.keys()))
        
        # Filter for efficient RL inference: Only run RL on tickers that have *some* signal
        # or just run on all active tickers found? 
        # For now, run RL on the union.
        
        candidates = []
        
        for t in all_tickers:
            income = income_map.get(t, {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'N/A'})
            sniper = sniper_map.get(t, {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'N/A'})
            
            # RL Inference (On-Demand)
            # Fetch DF from Sniper Loader (which has cache) to avoid re-fetching
            # Hack: Access loader from sniper expert
            try:
                # We need features. Sniper loader returns DF with indicators.
                df = self.sniper_expert.loader.fetch_data(t, interval='15m')
                # Add indicators if missed (Sniper loader might adds them inside run_scan but not strictly cached in loader?)
                # Actually fetch_data just gets raw, run_scan adds indicators locally.
                # We re-calc indicators or use a shared state.
                # Let's re-calc for safety.
                df = self.sniper_expert.loader.add_technical_indicators(df)
                rl_vote = self.rl_expert.get_vote(t, df)
            except Exception as e:
                rl_vote = {'Signal': 'ERROR', 'Confidence': 0.0, 'Reason': str(e)}

            # --- Consensus Logic (Level 1) ---
            decision = {
                'Ticker': t,
                'Action': 'WAIT',
                'Confidence': 0.0,
                'Rational': [],
                'QuantRisk': None
            }
            
            # Base Confidence from Heuristics
            # ... (Existing Logic Preserved) ...
            
            # A. High Volatility Regime
            if income['Signal'] == 'INCOME':
                decision['Rational'].append(f"Regime: High Vol")
                if sniper['Signal'] == 'BUY':
                    decision['Action'] = 'BULL_PUT_SPREAD'
                    decision['Confidence'] = (income['Confidence'] + sniper['Confidence']) / 2
                else:
                    decision['Action'] = 'IRON_CONDOR'
                    decision['Confidence'] = income['Confidence']
            
            # B. Low Volatility Regime
            elif income['Signal'] == 'SNIPER_PREP':
                 decision['Rational'].append(f"Regime: Coiled")
                 if sniper['Signal'] == 'BUY':
                     decision['Action'] = 'LONG_CALL_SNIPER'
                     decision['Confidence'] = max(income['Confidence'], sniper['Confidence']) + 0.1
                 else:
                     decision['Action'] = 'WATCH_FOR_BREAKOUT'
                     decision['Confidence'] = 0.5
            
            # C. Normal Regime
            else:
                 if sniper['Signal'] == 'BUY':
                     decision['Action'] = 'LONG_STOCK'
                     decision['Confidence'] = sniper['Confidence']
                 else:
                     decision['Action'] = 'WAIT'
            
            # RL BOOSTER
            if rl_vote['Signal'] == 'BUY':
                decision['Rational'].append(f"RL Agent: BUY ({rl_vote['Confidence']:.2f})")
                if decision['Action'] == 'WAIT':
                    # RL found something humans missed?
                    if rl_vote['Confidence'] > 0.7:
                        decision['Action'] = 'RL_SNIPE'
                        decision['Confidence'] = rl_vote['Confidence']
                else:
                    # Confluence!
                    decision['Confidence'] = min(0.99, decision['Confidence'] + 0.15)
                    decision['Rational'].append("AI Confirmation 🤖")
            
            # Append History (Sparkline)
            try:
                if df is not None and not df.empty:
                    subset = df.tail(60).reset_index()
                    decision['History'] = [
                        {
                            "Time": row['Datetime'].strftime('%H:%M') if pd.notnull(row['Datetime']) else str(row['Datetime']), 
                            "Close": round(row['Close'], 2),
                            "Volume": int(row['Volume'])
                        } 
                        for _, row in subset.iterrows()
                    ]
            except Exception as e:
                logger.debug(f"Failed to build history for {t}: {e}")
                decision['History'] = []

            # Add to Candidate List if Conf > 0.6
            if decision['Confidence'] >= 0.6:
                decision['ContextDF'] = df # Pass DF for Quant
                candidates.append(decision)
            
            final_decisions.append(decision)
            
        # --- Level 2: The Funnel (Quant Expert) ---
        if candidates:
            logger.info(f"🧪 Running Quant Simulation on {len(candidates)} candidates...")
            for cand in candidates:
                q_res = self.quant_expert.get_probability(cand['Ticker'], cand.pop('ContextDF', None))
                
                # Append Quant Insights
                cand['Rational'].append(f"Monte Carlo: {q_res['WinRate']*100:.1f}% Win Prob")
                cand['QuantRisk'] = q_res
                
                # Adjust Confidence based on Math
                if q_res['WinRate'] > 0.65:
                    cand['Confidence'] = min(0.99, cand['Confidence'] + 0.1)
                    cand['Rational'].append("Math Approved 📐")
                elif q_res['WinRate'] < 0.45:
                    cand['Confidence'] -= 0.2
                    cand['Rational'].append("Math Reject ❌")
                    cand['Action'] = "WAIT (Quant Reject)"

        # Cleanup ContextDF from final list (not JSON serializable usually)
        for d in final_decisions: 
            d.pop('ContextDF', None)
            
        return final_decisions

if __name__ == "__main__":
    brain = HybridBrain()
    thoughts = brain.think()
    
    print("\n" + "="*90)
    print("🧠 THE THINKING ENGINE REPORT (Hybrid + RL + Quant) 🧠")
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
