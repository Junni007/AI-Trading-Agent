import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.env import TradingEnv

class TestVerifiableReward(unittest.TestCase):
    def setUp(self):
        # Create dummy data: 100 steps, price increasing clearly
        prices = np.linspace(100, 200, 100) 
        # Make a dip in the middle to test SMA logic
        prices[50:60] = prices[50:60] * 0.8
        
        self.df = pd.DataFrame({'Close': prices})
        self.env = TradingEnv(self.df, window_size=10)
        
    def test_trend_reward_buy_uptrend(self):
        """Test that buying in uptrend gives positive trend reward"""
        # Advance to step 90 (uptrend restored)
        self.env.current_step = 90
        # Price at 90 is high, SMA50 (40-90) should be lower
        
        # Action 1 = BUY
        # Need to mock the internal state update to invoke _calculate_verifiable_reward effectively via step
        # or just call the helper directly
        
        # Let's call step
        obs, reward, done, truncated, info = self.env.step(1)
        
        # We expect positive reward. 
        # PnL might be slightly neg due to spread/cost or 0 if immediate
        # Trend reward should be +0.05
        
        print(f"Step 90 (Uptrend) Buy Reward: {reward}")
        # Note: PnL reward is 0 on step 0 change if price doesn't change instant (env uses next price usually)
        # But here price changes step to step.
        
        # Let's verify via helper directly for precision
        cur_price = self.df['Close'].iloc[90]
        # SMA 50 from 40 to 90
        sma = self.df['Close'].iloc[40:90].mean()
        print(f"Price: {cur_price}, SMA50: {sma}")
        
        # Start fresh for helper calc
        # action, prev, cur, price, step
        r = self.env._calculate_verifiable_reward("BUY", 10000, 10000, cur_price, 90)
        
        # Trend comp should be +0.05 because Price > SMA
        # PnL 0
        self.assertAlmostEqual(r, 0.05, delta=0.001)

    def test_trend_penalty_buy_downtrend(self):
        """Test that buying in downtrend (below SMA) gives penalty"""
        # Step 55 is in the dip
        self.env.current_step = 55
        
        cur_price = self.df['Close'].iloc[55]
        # SMA 50 from 5 to 55 (mostly higher pre-dip prices)
        sma = self.df['Close'].iloc[5:55].mean()
        print(f"Step 55 (Dip) Price: {cur_price}, SMA50: {sma}")
        
        # Should be penalty
        r = self.env._calculate_verifiable_reward("BUY", 10000, 10000, cur_price, 55)
        
        # Trend comp should be -0.05 because Price < SMA
        self.assertAlmostEqual(r, -0.05, delta=0.001)

if __name__ == '__main__':
    unittest.main()
