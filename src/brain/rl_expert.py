import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, Optional, Tuple
from pathlib import Path

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RLExpert')

class RLExpert:
    """
    Expert 3: Deep Reinforcement Learning (PPO) - Lightweight Inference.
    Uses Numpy-only implementation (No PyTorch dependency).
    """
    
    def __init__(self, model_path: str = "checkpoints/best_ppo_light.npz"):
        self.weights = {}
        self.is_trained = False
        self._load_model(model_path)
        
        # Dimensions (must match training)
        self.input_dim = 8
        self.output_dim = 3
        
    def _load_model(self, path: str):
        """
        Loads weights from a .npz file (numpy archive).
        """
        try:
            p = Path(path)
            if not p.exists():
                logger.warning(f"⚠️ Light Checkpoint not found at {path}. RLExpert running with RANDOM DECISIONS.")
                self.is_trained = False
                return

            # Load Numpy weights
            data = np.load(path)
            self.weights = {k: data[k] for k in data.files}
            
            self.is_trained = True
            logger.info(f"✅ RLExpert loaded lightweight model from {path}")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to load light model: {e}")
            self.is_trained = False

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Manual Forward Pass of the Actor-Critic Network.
        Architecture: 
          - Shared FC1 (64) -> Tanh
          - Shared FC2 (64) -> Tanh
          - Actor Head (Output Dim) -> Softmax
          - Critic Head (1) -> Linear
        """
        if not self.is_trained or not self.weights:
            # Random fallback
            return np.random.rand(1, 3), 0.0

        # Helper for linear layer
        def linear(input_x, w, b):
            return np.dot(input_x, w.T) + b

        # Helper for activation
        def tanh(x):
            return np.tanh(x)
            
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        try:
            # Access weights (Expect names: shared_net.0.weight, actor.weight, etc.)
            # We map standard PyTorch names to our dict keys
            
            # Layer 1 (Shared)
            h1 = linear(x, self.weights['shared_net.0.weight'], self.weights['shared_net.0.bias'])
            h1 = tanh(h1)
            
            # Layer 2 (Shared)
            h2 = linear(h1, self.weights['shared_net.2.weight'], self.weights['shared_net.2.bias'])
            h2 = tanh(h2)
            
            # Actor Head
            logits = linear(h2, self.weights['actor.weight'], self.weights['actor.bias'])
            probs = softmax(logits)
            
            # Critic Head
            value = linear(h2, self.weights['critic.weight'], self.weights['critic.bias'])
            
            return probs, value.item()
            
        except KeyError as e:
            logger.error(f"Missing weight key: {e}")
            return np.random.rand(1, 3), 0.0

    def get_vote(self, ticker: str, df: pd.DataFrame) -> Dict:
        """
        Runs the PPO Policy on the latest data point (Numpy Version).
        """
        if df.empty or len(df) < 50:
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Model Not Ready'}
            
        try:
            # 1. Feature Engineering (Match Env)
            last_row = df.iloc[-1]
            features = [
                last_row.get('RSI', 50) / 100.0,
                last_row.get('MACD', 0),
                last_row.get('Log_Return', 0),
                last_row.get('ATR', 0)
            ]
            
            # Padding to match input_dim=8 (Simulate account info as 0s/neutral for inference if not available)
            # Real env has Balance/Postion. Here we assume static/normalized or just feed 0s.
            # Ideally we pass current portfolio state, but for pure signal generation, 0s are safe-ish.
            features += [0.0] * (self.input_dim - len(features))
            
            state_vec = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # 2. Inference
            probs, value = self._forward(state_vec)
            action = np.argmax(probs)
            confidence = probs[0][action]
            
            # 3. Map Action
            signals = {0: "WAIT", 1: "BUY", 2: "SELL"}
            signal = signals.get(action, "WAIT")
            
            return {
                'Signal': signal,
                'Confidence': float(confidence),
                'Reason': f"RL Agent (Prob={confidence:.2f})"
            }
            
        except Exception as e:
            logger.error(f"Inference Error {ticker}: {e}")
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Inference Failed'}

if __name__ == "__main__":
    expert = RLExpert()
    # Dummy Data
    df = pd.DataFrame({'Close': [100]*60, 'RSI': [30]*60})
    print(expert.get_vote("TEST", df))
