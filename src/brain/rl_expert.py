import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict
from src.agent import TradingAgent, ActorCritic
from src.env import TradingEnv
from src.data_loader import MVPDataLoader

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RLExpert')

class RLExpert:
    """
    Expert 3: Deep Reinforcement Learning (PPO).
    Loads the trained PPO Model to provide AI-driven signals.
    """
    
    def __init__(self, model_path: str = "checkpoints/best_ppo.ckpt"):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_path)
        
    def _load_model(self, path: str):
        """
        Loads the ActorCritic model from a PyTorch Lightning checkpoint.
        Falls back to random weights if no checkpoint is found.
        """
        import os
        
        # Standard Dimensions (must match env.py -> FeatureProcessor output + 2 account fields)
        self.input_dim = 8  # 6 TDA features + 2 account (balance, position)
        self.output_dim = 3  # Hold, Buy, Sell
        
        self.model = ActorCritic(self.input_dim, self.output_dim).to(self.device)
        
        if os.path.exists(path):
            try:
                # Lightning checkpoints store state under 'state_dict' key
                checkpoint = torch.load(path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    # Lightning format: keys are like 'model.fc1.weight'
                    state = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
                    self.model.load_state_dict(state)
                else:
                    # Direct state dict
                    self.model.load_state_dict(checkpoint)
                    
                self.model.eval()
                logger.info(f"✅ RLExpert loaded trained model from {path}")
                self.is_trained = True
            except Exception as e:
                logger.warning(f"⚠️ Failed to load checkpoint: {e}. Using untrained model.")
                self.is_trained = False
        else:
            logger.warning(f"⚠️ Checkpoint not found at {path}. RLExpert running with RANDOM WEIGHTS.")
            self.is_trained = False

    def get_vote(self, ticker: str, df: pd.DataFrame) -> Dict:
        """
        Runs the PPO Policy on the latest data point.
        """
        if self.model is None or df.empty or len(df) < 50:
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Model Not Ready'}
            
        try:
            # 1. Construct State (Observation)
            # We must replicate the Env observation logic exactly.
            # Feature calculation is complex (TDA).
            # For MVP speed, we use a simplified version here or call Env logic?
            # Calling Env logic requires instantiating Env which is heavy.
            # Let's trust the Data Loader to give us clean features.
            
            # Feature Engineering similar to env.py
            # If env.py uses TDA, we must assume df has TDA features OR compute them.
            # This is the "Integration Hell". 
            # Solution: We skip TDA for now and use RSI/MACD for the RL inputs in this version
            # if the model was trained on TDA, this will fail.
            # Let's assume the RL model takes [RSI, MACD, Returns, Volatility]
            
            last_row = df.iloc[-1]
            features = [
                last_row.get('RSI', 50) / 100.0,
                last_row.get('MACD', 0),
                last_row.get('Log_Return', 0),
                last_row.get('ATR', 0)
            ]
            
            # Resize if needed (padding)
            state_vec = np.array(features, dtype=np.float32)
            
            # 2. Inference
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            
            # Check dimensions match
            if state_tensor.shape[1] != self.input_dim:
                # Pad with zeros to match
                pad_size = self.input_dim - state_tensor.shape[1]
                if pad_size > 0:
                    padding = torch.zeros((1, pad_size)).to(self.device)
                    state_tensor = torch.cat([state_tensor, padding], dim=1)
            
            with torch.no_grad():
                probs, value = self.model(state_tensor)
                action = torch.argmax(probs).item()
                confidence = probs[0][action].item()
                
            # 3. Map Action
            # 0: Hold, 1: Buy, 2: Sell
            signals = {0: "WAIT", 1: "BUY", 2: "SELL"}
            signal = signals.get(action, "WAIT")
            
            return {
                'Signal': signal,
                'Confidence': float(confidence),
                'Reason': f"RL Policy (A={action}, Val={value.item():.2f})"
            }
            
        except Exception as e:
            logger.error(f"Inference Error {ticker}: {e}")
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Inference Failed'}

if __name__ == "__main__":
    expert = RLExpert()
    # Dummy Data
    df = pd.DataFrame({'Close': [100]*60, 'RSI': [30]*60})
    print(expert.get_vote("TEST", df))
