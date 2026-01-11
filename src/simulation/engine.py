import logging
import json
import os
from datetime import datetime

from src.config import settings

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimEngine')

DB_PATH = settings.STATE_FILE

class SimulationEngine:
    """
    Manages a Paper Trading Portfolio with RL-style scoring.
    """
    
    def __init__(self, initial_balance=10000.0):
        self.state = self.load_state() or {
            "balance": initial_balance,
            "cash": initial_balance,
            "positions": {}, # {ticker: {qty, avg_price}}
            "history": [], # Trade logs
            "score": 0, # RL Score (+1 Profit, -4 Loss)
            "level": "Novice (Risk Taker)",
            "status": "ALIVE" # ALIVE or DEAD
        }
        
    def load_state(self):
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
        
    def save_state(self):
        with open(DB_PATH, 'w') as f:
            json.dump(self.state, f, indent=4)
            
    def get_portfolio(self):
        return self.state

    def reset(self):
        self.state["balance"] = 10000.0
        self.state["cash"] = 10000.0
        self.state["positions"] = {}
        self.state["history"] = []
        self.state["score"] = 0
        self.state["level"] = "Novice (Risk Taker)"
        self.state["status"] = "ALIVE"
        self.save_state()

    def process_tick(self, market_data: list, regime: str = "NEUTRAL"):
        """
        Processes a 'tick' of market data decisions.
        """
        logs = []
        
        # Create a map for quick lookup
        market_map = {item['Ticker']: item for item in market_data}
        
        # 1. Manage Existing Positions
        active_tickers = list(self.state['positions'].keys())
        for ticker in active_tickers:
            if ticker not in market_map: continue
            
            position = self.state['positions'][ticker]
            current_price = market_map[ticker]['Price']
            
            entry = position['avg_price']
            qty = position['qty']
            pnl_pct = (current_price - entry) / entry * 100
            
            # Action Logic with Regime Modifiers
            action = None
            reward = 0
            
            # Regime Modifiers
            tp_target = 1.0
            sl_target = -0.5
            
            if regime == "HIGH_VOLATILITY":
                tp_target = 2.0 # Greedier
                sl_target = -1.5 # Looser stop
            elif regime == "LOW_VOLATILITY":
                tp_target = 0.8 # Quick scalp
                sl_target = -0.3 # Tight stop
            
            # Take Profit
            if pnl_pct >= tp_target:
                action = "SELL_TP"
                reward = 1
            # Stop Loss
            elif pnl_pct <= sl_target:
                action = "SELL_SL"
                reward = -4
            
            if action:
                # Execute Sell
                proceeds = qty * current_price
                self.state['cash'] += proceeds
                del self.state['positions'][ticker]
                
                # Update RL Score
                self.state['score'] += reward
                self.update_level()
                
                # Log
                log_entry = f"{action} {ticker} @ {current_price}. PnL: {pnl_pct:.2f}%. Reward: {reward} pts."
                self.state['history'].insert(0, log_entry)
                logs.append(log_entry)
        
        # 2. KeyLogic: Open New Positions
        current_value = self.state['cash'] + sum([
            pos['qty'] * market_map.get(t, {'Price': pos['avg_price']}).get('Price', pos['avg_price']) 
            for t, pos in self.state['positions'].items()
        ])
        
        # Update Equity Curve for Stats
        if 'equity_curve' not in self.state: self.state['equity_curve'] = []
        self.state['equity_curve'].append(current_value)
        if len(self.state['equity_curve']) > 252: self.state['equity_curve'].pop(0) # Keep 1 year daily approx
        
        # Only if we have cash and strict criteria
        for item in market_data:
            ticker = item['Ticker']
            price = item.get('Price', 0)
            
            if ticker in self.state['positions']: continue # Already hold it
            if price == 0: continue
            
            # --- RL AGENT LOGIC (Dynamic Risk) ---
            # 1. Determine Threshold based on Level
            threshold = 0.85 # Default
            level = self.state['level']
            
            if "Novice" in level: threshold = 0.10 
            elif "Apprentice" in level: threshold = 0.40
            elif "Pro" in level: threshold = 0.70
            elif "Wolf" in level: threshold = 0.90
            
            # Regime Penalty: If Pro, require higher conf in High Vol
            if regime == "HIGH_VOLATILITY" and "Pro" in level:
                threshold += 0.05
            
            # 2. Educated Guessing
            adjusted_conf = item['Confidence']
            if "Volume" in item.get('Reason', ''): 
                adjusted_conf += 0.10
            
            # 3. Execution
            is_buy_signal = ("BUY" in item.get('Signal', '') or 
                             "LONG" in item.get('Action', '') or
                             "SNIPER" in item.get('Signal', ''))
                             
            if (adjusted_conf >= threshold) and is_buy_signal:
                allocation = self.state['balance'] * 0.20
                if self.state['cash'] > allocation:
                    qty = int(allocation / price)
                    if qty > 0:
                        cost = qty * price
                        self.state['cash'] -= cost
                        self.state['positions'][ticker] = {
                            "qty": qty,
                            "avg_price": price
                        }
                        log_entry = f"BOUGHT {ticker} @ {price}. Conf: {adjusted_conf:.2f} (Risk: {level})"
                        self.state['history'].insert(0, log_entry)
                        logs.append(log_entry)

        # Final Balance Update
        self.state['balance'] = current_value
        
        # Stats Calculation
        self.calculate_stats()
        
        self.save_state()
        self.check_survival() 
        if not logs:
            logs.append(f"Scan Complete. Regime: {regime}. No Actions.")
            
        return logs

    def calculate_stats(self):
        """Calculates Max Drawdown and Sharpe Ratio"""
        import numpy as np
        curve = self.state.get('equity_curve', [])
        if len(curve) < 2: return
        
        # Max Drawdown
        peak = curve[0]
        max_dd = 0
        for val in curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
        self.state['max_drawdown'] = max_dd * 100
        
        # Sharpe (Simplified)
        returns = np.diff(curve) / curve[:-1]
        if len(returns) > 0:
            vol = np.std(returns)
            if vol > 0:
                sharpe = np.mean(returns) / vol * np.sqrt(252) # Annualized
                self.state['sharpe_ratio'] = sharpe
            else:
                self.state['sharpe_ratio'] = 0.0


    def check_survival(self):
        """
        The Perma-Death Mechanic.
        If Balance <= 0:
        - Novice: DIE (Game Over, must manual reset).
        - Pro: Demote (Lose Level, keep playing).
        """
        if self.state['balance'] <= 0:
            current_level = self.state['level']
            
            if "Novice" in current_level:
                self.state['status'] = "DEAD" # Frontend will see this and Unlock
                self.state['history'].insert(0, "☠️ ACCOUNT BLOWN! GAME OVER. ☠️")
                # Reset Score but keep history for shame
                self.state['score'] = 0
            else:
                # Second Chance (Demotion)
                self.state['balance'] = 10000.0 # Refill
                self.state['cash'] = 10000.0
                self.state['history'].insert(0, "⚠️ MARGIN CALL! Level Lost. Balance Reset.")
                
                # Demote Logic
                if "Wolf" in current_level: self.state['level'] = "Pro Trader"
                elif "Grandmaster" in current_level: self.state['level'] = "Pro Trader"
                elif "Pro" in current_level: self.state['level'] = "Apprentice"
                elif "Apprentice" in current_level: self.state['level'] = "Novice (Risk Taker)"
                
                self.state['score'] = max(0, self.state['score'] - 50) # Big Penalty

        self.save_state()

    def update_level(self):
        score = self.state['score']
        # Level Up Logic
        if score >= 500: self.state['level'] = "Wolf of Wall Street"
        elif score >= 200: self.state['level'] = "Grandmaster"
        elif score >= 100: self.state['level'] = "Pro Trader"
        elif score >= 50: self.state['level'] = "Apprentice"
        else: self.state['level'] = "Novice (Risk Taker)"

    def is_alive(self):
        return self.state.get('status', 'ALIVE') != "DEAD"
