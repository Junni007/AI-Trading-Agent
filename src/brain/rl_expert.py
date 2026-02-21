import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RLExpert')


class RLExpert:
    """
    Expert 3: Deep Reinforcement Learning (PPO) - Lightweight Inference.
    Uses Numpy-only implementation (No PyTorch dependency).

    Architecture MUST match training (RecurrentActorCritic in train_ppo_optimized.py):
      - LSTM(input_size=9, hidden_size=256, num_layers=1, batch_first=True)
      - Actor: Linear(256, 128) -> ReLU -> Linear(128, 3)
      - Critic: Linear(256, 128) -> ReLU -> Linear(128, 1)

    Input features (9 total = 7 market + 2 context):
      [Returns, LogReturns, Volatility, Volume_Z, RSI, RSI_Rank, Momentum_Rank, Position, Balance]
    """

    # Feature specification — single source of truth matching VectorizedTradingEnv
    FEATURE_COLS = ['Returns', 'LogReturns', 'Volatility', 'Volume_Z', 'RSI', 'RSI_Rank', 'Momentum_Rank']
    # The current checkpoint (best_ppo_light.npz) was trained with 7 features.
    INPUT_DIM = len(FEATURE_COLS)  # Used to be + 2
    OUTPUT_DIM = 3  # Hold, Buy, Sell
    HIDDEN_DIM = 256
    WINDOW_SIZE = 50

    def __init__(self, model_path: str = "checkpoints/best_ppo_light.npz"):
        self.weights: Dict[str, np.ndarray] = {}
        self.is_trained = False
        self._load_model(model_path)

    def _load_model(self, path: str):
        """Loads weights from a .npz file (numpy archive)."""
        try:
            p = Path(path)
            if not p.exists():
                logger.warning(f"Light Checkpoint not found at {path}. RLExpert returning WAIT signals.")
                self.is_trained = False
                return

            data = np.load(path)
            self.weights = {k: data[k] for k in data.files}
            self.is_trained = True
            logger.info(f"RLExpert loaded lightweight model from {path}")

        except Exception as e:
            logger.error(f"Failed to load light model: {e}")
            self.is_trained = False

    # ------------------------------------------------------------------
    # Numpy LSTM implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _lstm_step(self, x_t: np.ndarray, h: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single LSTM time-step.
        x_t: (batch, input_dim)
        h, c: (batch, hidden_dim)
        
        PyTorch LSTM weight layout:
          weight_ih: (4*H, I) — [W_ii, W_if, W_ig, W_io] stacked
          weight_hh: (4*H, H) — [W_hi, W_hf, W_hg, W_ho] stacked
          bias_ih:   (4*H,)
          bias_hh:   (4*H,)
        """
        H = self.HIDDEN_DIM
        W_ih = self.weights['lstm.weight_ih_l0']  # (4H, I)
        W_hh = self.weights['lstm.weight_hh_l0']  # (4H, H)
        b_ih = self.weights['lstm.bias_ih_l0']     # (4H,)
        b_hh = self.weights['lstm.bias_hh_l0']     # (4H,)

        gates = x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh  # (batch, 4H)

        i_gate = self._sigmoid(gates[:, 0:H])
        f_gate = self._sigmoid(gates[:, H:2*H])
        g_gate = np.tanh(gates[:, 2*H:3*H])
        o_gate = self._sigmoid(gates[:, 3*H:4*H])

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * np.tanh(c_new)
        return h_new, c_new

    def _forward(self, x_seq: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass through LSTM Actor-Critic.
        x_seq: (1, seq_len, INPUT_DIM) — a single observation window.
        Returns: (action_probs [1, 3], value scalar)
        """
        if not self.is_trained or not self.weights:
            logger.warning("RLExpert: No trained model available. Returning WAIT signal.")
            return np.array([[0.8, 0.1, 0.1]]), 0.0

        def linear(x, w, b):
            return x @ w.T + b

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        try:
            batch = x_seq.shape[0]
            seq_len = x_seq.shape[1]

            # Initialize hidden state
            h = np.zeros((batch, self.HIDDEN_DIM), dtype=np.float32)
            c = np.zeros((batch, self.HIDDEN_DIM), dtype=np.float32)

            # Run LSTM over sequence
            for t in range(seq_len):
                h, c = self._lstm_step(x_seq[:, t, :], h, c)

            # h is now the last hidden state — feed to heads
            # Actor: Linear(256,128) -> ReLU -> Linear(128,3)
            a1 = np.maximum(0, linear(h, self.weights['actor.0.weight'], self.weights['actor.0.bias']))
            logits = linear(a1, self.weights['actor.2.weight'], self.weights['actor.2.bias'])
            probs = softmax(logits)

            # Critic: Linear(256,128) -> ReLU -> Linear(128,1)
            c1 = np.maximum(0, linear(h, self.weights['critic.0.weight'], self.weights['critic.0.bias']))
            value = linear(c1, self.weights['critic.2.weight'], self.weights['critic.2.bias'])

            return probs, float(value.item())

        except KeyError as e:
            logger.error(f"Missing weight key: {e}. Returning WAIT signal.")
            return np.array([[0.8, 0.1, 0.1]]), 0.0

    # ------------------------------------------------------------------
    # Feature engineering — matches VectorizedTradingEnv._precompute_features
    # ------------------------------------------------------------------
    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Builds the same 7-feature vector as VectorizedTradingEnv._precompute_features.
        Returns: numpy array of shape (n_rows, 7), NaN rows dropped.
        """
        ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        ohlcv['Returns'] = ohlcv['Close'].pct_change()
        ohlcv['LogReturns'] = np.log(ohlcv['Close'] / ohlcv['Close'].shift(1))
        ohlcv['Volatility'] = ohlcv['Returns'].rolling(20).std()
        vol_mean = ohlcv['Volume'].rolling(20).mean()
        vol_std = ohlcv['Volume'].rolling(20).std()
        ohlcv['Volume_Z'] = (ohlcv['Volume'] - vol_mean) / (vol_std + 1e-8)

        # RSI
        delta = ohlcv['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        ohlcv['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

        # Cross-sectional ranks default to 0.5 for single-ticker inference
        ohlcv['RSI_Rank'] = 0.5
        ohlcv['Momentum_Rank'] = 0.5

        ohlcv = ohlcv.dropna()
        features = ohlcv[self.FEATURE_COLS].values.astype(np.float32)

        # Z-score normalize (same as training)
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        features = (features - mean) / std

        return features

    def get_vote(self, ticker: str, df: pd.DataFrame) -> Dict:
        """
        Runs the PPO Policy on a window of data (LSTM version).
        """
        if df.empty or len(df) < self.WINDOW_SIZE + 20:
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Insufficient data'}

        try:
            # 1. Build features matching training env
            features = self._build_features(df)

            if len(features) < self.WINDOW_SIZE:
                return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Insufficient features after NaN drop'}

            # 2. Take the last window
            window = features[-self.WINDOW_SIZE:]  # (50, 7)

            # 3. Use only the feature window if trained with 7 features
            obs = window[np.newaxis, :, :]  # (1, 50, 7)

            # 4. Inference
            probs, value = self._forward(obs)
            action = int(np.argmax(probs))
            confidence = float(probs[0][action])

            # 5. Map Action
            signal_map = ["WAIT", "BUY", "SELL"]
            signal = signal_map[action] if action < len(signal_map) else "WAIT"

            return {
                'Signal': signal,
                'Confidence': confidence,
                'Reason': f"RL Agent (Prob={confidence:.2f}, V={value:.2f})"
            }

        except Exception as e:
            logger.error(f"Inference Error {ticker}: {e}")
            return {'Signal': 'NEUTRAL', 'Confidence': 0.0, 'Reason': 'Inference Failed'}


if __name__ == "__main__":
    expert = RLExpert()
    # Dummy Data — need enough rows for rolling windows
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Open': 100 + np.random.randn(n).cumsum(),
        'High': 101 + np.random.randn(n).cumsum(),
        'Low': 99 + np.random.randn(n).cumsum(),
        'Close': 100 + np.random.randn(n).cumsum(),
        'Volume': np.random.randint(1000, 10000, n),
    })
    print(expert.get_vote("TEST", df))
