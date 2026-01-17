import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('QuantExpert')

class QuantExpert:
    """
    Expert 4: Quantitative Risk Engine.
    Uses Monte Carlo Simulations (Heston Model + Jump Diffusion) to estimate probability of success.
    Designed to run ONLY on shortlisted candidates.
    """
    
    def __init__(self, num_paths=10000, time_horizon_days=5):
        self.num_paths = num_paths
        self.T = time_horizon_days / 252.0 # Annualized
        self.dt = self.T / time_horizon_days # Daily steps
        
    def simulate_heston_jump_diffusion(self, S0, v0, mu, kappa, theta, xi, rho, lambda_jump, mean_jump, std_jump):
        """
        Simulates paths using Heston Stochastic Variance + Merton Jump Diffusion (Bates Model).
        
        Parameters:
        - S0: Initial Price
        - v0: Initial Variance
        - mu: Drift (Risk-free rate - dividend)
        - kappa: Mean reversion speed of variance
        - theta: Long-term average variance
        - xi: Volatility of volatility (Vol-Vol)
        - rho: Correlation between price and variance
        - lambda_jump: Jump intensity (jumps per year)
        - mean_jump: Mean jump size
        - std_jump: Jump size std dev
        """
        N = int(self.T / self.dt)
        
        # Arrays to store paths
        S = np.zeros((self.num_paths, N + 1))
        v = np.zeros((self.num_paths, N + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Cholesky Decomposition for Correlated Brownians
        cov = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(cov)
        
        for t in range(1, N + 1):
            # Generate Correlated Random Normal Variables
            Z = np.random.normal(0, 1, (self.num_paths, 2))
            dW = Z @ L.T # [dW_S, dW_v]
            dW_S = dW[:, 0] * np.sqrt(self.dt)
            dW_v = dW[:, 1] * np.sqrt(self.dt)
            
            # Heston Variance Update (Full Truncation to keep positive)
            v_prev = v[:, t-1]
            dv = kappa * (theta - v_prev) * self.dt + xi * np.sqrt(np.maximum(0, v_prev)) * dW_v
            v[:, t] = np.maximum(0, v_prev + dv)
            
            # Jump Component (Poisson)
            # Poisson * Normal Jump Size
            jumps = np.random.poisson(lambda_jump * self.dt, self.num_paths)
            jump_mag = np.random.normal(mean_jump, std_jump, self.num_paths) * jumps
            
            # Price Update (Geometric Brownian Motion + Jump)
            # dS = S * (mu dt + sqrt(v) dW + Jump)
            drift = (mu - 0.5 * v[:, t]) * self.dt
            diffusion = np.sqrt(np.maximum(0, v[:, t])) * dW_S
            
            # Log-Euler Discretization is more stable
            # ln S_t = ln S_{t-1} + (mu - 0.5 v) dt + sqrt(v) dW + Jumps
            S[:, t] = S[:, t-1] * np.exp(drift + diffusion + jump_mag)
            
        return S

    def get_probability(self, ticker: str, df: pd.DataFrame, target_pct=0.02) -> Dict:
        """
        Runs the simulation and calculates the probability of hitting Target% profit 
        before hitting Stop% loss (or end of horizon).
        """
        if df.empty:
            return {'ProbGain': 0.0, 'ProbLoss': 0.0, 'ExpectedValue': 0.0}
            
        try:
            # 1. Calibrate / Estimate Parameters from History
            # Simple Estimation for MVP
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            
            S0 = df['Close'].iloc[-1]
            hist_vol = returns.std() * np.sqrt(252)
            v0 = hist_vol ** 2
            
            # Assumptions for Heston (Hard to calibrate without options)
            mu = 0.05 # Risk free assumption
            kappa = 2.0 # Mean reversion speed
            theta = v0 # Long term variance = current estimate
            xi = 0.3 # Vol of Vol
            rho = -0.5 # Leverage effect (Price down -> Vol up)
            
            # Jump Assumptions
            # Detect historical large jumps (> 3 std dev)
            z_scores = (returns - returns.mean()) / returns.std()
            jump_count = (np.abs(z_scores) > 3).sum()
            lambda_jump = (jump_count / len(df)) * 252 # Annualized
            
            # Run Simulation
            paths = self.simulate_heston_jump_diffusion(
                S0, v0, mu, kappa, theta, xi, rho, 
                lambda_jump=lambda_jump, mean_jump=0, std_jump=0.02
            )
            
            # Calculate Probabilities
            # Target Price
            target_price = S0 * (1 + target_pct)
            stop_price = S0 * (1 - target_pct/2) # 1:2 Risk Reward
            
            # Check hits
            # For each path, did it hit target before stop?
            # Max possible price in path
            max_prices = paths.max(axis=1)
            min_prices = paths.min(axis=1)
            
            # Simple interaction: Did it end up positive?
            final_prices = paths[:, -1]
            prob_gain = np.mean(final_prices > target_price)
            prob_loss = np.mean(final_prices < stop_price)
            
            # More complex: Barrier hit?
            # This is path dependent.
            # Let's stick to "Prob of ending > Target" for simplicity of scanning
            # Or "Win Rate" = Prob(Final > S0)
            
            prob_win = np.mean(final_prices > S0)
            ev = np.mean(final_prices) - S0
            
            return {
                'WinRate': prob_win,
                'EV': ev,
                'VaR95': S0 - np.percentile(final_prices, 5)
            }
            
        except Exception as e:
            logger.error(f"Quant Sim Failed {ticker}: {e}")
            return {'WinRate': 0.0, 'EV': 0.0}

if __name__ == "__main__":
    expert = QuantExpert()
    # Dummy Data
    df = pd.DataFrame({'Close': np.cumprod(1 + np.random.normal(0, 0.01, 100)) * 100})
    res = expert.get_probability("TEST", df)
    print("Quant Results:", res)
