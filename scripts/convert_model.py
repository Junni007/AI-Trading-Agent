"""
Convert PyTorch Lightning checkpoint (RecurrentActorCritic LSTM)
to a Numpy .npz archive for lightweight inference in RLExpert.

Architecture (must match train_ppo_optimized.py::RecurrentActorCritic):
  - LSTM(input_size=9, hidden_size=256, num_layers=1, batch_first=True)
  - Actor:  Linear(256, 128) -> ReLU -> Linear(128, 3)
  - Critic: Linear(256, 128) -> ReLU -> Linear(128, 1)
"""
import numpy as np
import torch
from pathlib import Path


INPUT_DIM = 9    # 7 features + 2 context
HIDDEN_DIM = 256
OUTPUT_DIM = 3

# Expected weight keys after stripping 'model.' prefix
EXPECTED_KEYS = [
    'lstm.weight_ih_l0',   # (4*H, I) = (1024, 9)
    'lstm.weight_hh_l0',   # (4*H, H) = (1024, 256)
    'lstm.bias_ih_l0',     # (4*H,)   = (1024,)
    'lstm.bias_hh_l0',     # (4*H,)   = (1024,)
    'actor.0.weight',      # (128, 256)
    'actor.0.bias',        # (128,)
    'actor.2.weight',      # (3, 128)
    'actor.2.bias',        # (3,)
    'critic.0.weight',     # (128, 256)
    'critic.0.bias',       # (128,)
    'critic.2.weight',     # (1, 128)
    'critic.2.bias',       # (1,)
]


def convert_ckpt_to_npz(ckpt_path: str, output_path: str = "checkpoints/best_ppo_light.npz"):
    """
    Extracts weights from a PyTorch Lightning Checkpoint and saves them as a Numpy .npz archive.
    This allows loading the model without PyTorch installed!
    """
    path = Path(ckpt_path)
    if not path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Creating DUMMY weights for testing (LSTM architecture)...")
        H = HIDDEN_DIM
        weights = {
            # LSTM weights
            'lstm.weight_ih_l0': np.random.randn(4 * H, INPUT_DIM).astype(np.float32) * 0.01,
            'lstm.weight_hh_l0': np.random.randn(4 * H, H).astype(np.float32) * 0.01,
            'lstm.bias_ih_l0': np.zeros(4 * H, dtype=np.float32),
            'lstm.bias_hh_l0': np.zeros(4 * H, dtype=np.float32),
            # Actor head
            'actor.0.weight': np.random.randn(H // 2, H).astype(np.float32) * 0.01,
            'actor.0.bias': np.zeros(H // 2, dtype=np.float32),
            'actor.2.weight': np.random.randn(OUTPUT_DIM, H // 2).astype(np.float32) * 0.01,
            'actor.2.bias': np.zeros(OUTPUT_DIM, dtype=np.float32),
            # Critic head
            'critic.0.weight': np.random.randn(H // 2, H).astype(np.float32) * 0.01,
            'critic.0.bias': np.zeros(H // 2, dtype=np.float32),
            'critic.2.weight': np.random.randn(1, H // 2).astype(np.float32) * 0.01,
            'critic.2.bias': np.zeros(1, dtype=np.float32),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **weights)
        print(f"Dummy weights saved to {output_path}")
        return

    try:
        print(f"Loading {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        state_dict = checkpoint.get('state_dict', checkpoint)

        numpy_weights = {}
        for key, tensor in state_dict.items():
            # Clean key names (remove 'model.' prefix from Lightning)
            clean_key = key.replace('model.', '')

            numpy_weights[clean_key] = tensor.numpy()
            print(f"   Exporting {clean_key} {tuple(tensor.shape)}")

        # Validate all expected keys are present
        missing = [k for k in EXPECTED_KEYS if k not in numpy_weights]
        if missing:
            print(f"WARNING: Missing expected keys: {missing}")
            print("The checkpoint may use a different architecture than RecurrentActorCritic.")

        # Save compressed npz
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **numpy_weights)
        print(f"Conversion Complete! Saved to {output_path}")
        print(f"   Keys: {list(numpy_weights.keys())}")

    except Exception as e:
        print(f"Conversion Failed: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Torch Checkpoint to Numpy (.npz)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_ppo.ckpt",
                        help="Input Checkpoint path (.ckpt)")
    parser.add_argument("--out", type=str, default="checkpoints/best_ppo_light.npz",
                        help="Output .npz path")

    args = parser.parse_args()
    convert_ckpt_to_npz(args.ckpt, args.out)
