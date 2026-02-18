
import numpy as np
import torch
import sys
from pathlib import Path

def convert_ckpt_to_npz(ckpt_path: str, output_path: str = "checkpoints/best_ppo_light.npz"):
    """
    Extracts weights from a PyTorch Lightning Checkpoint and saves them as a Numpy .npz archive.
    This allows loading the model without PyTorch installed!
    """
    path = Path(ckpt_path)
    if not path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        # Create dummy weights for testing if no checkpoint
        print("💡 Creating DUMMY weights for testing...")
        weights = {
            'shared_net.0.weight': np.random.randn(64, 8).astype(np.float32) * 0.1,
            'shared_net.0.bias': np.zeros(64, dtype=np.float32),
            'shared_net.2.weight': np.random.randn(64, 64).astype(np.float32) * 0.1,
            'shared_net.2.bias': np.zeros(64, dtype=np.float32),
            'actor.weight': np.random.randn(3, 64).astype(np.float32) * 0.1,
            'actor.bias': np.zeros(3, dtype=np.float32),
            'critic.weight': np.random.randn(1, 64).astype(np.float32) * 0.1,
            'critic.bias': np.zeros(1, dtype=np.float32)
        }
        np.savez(output_path, **weights)
        print(f"✅ Dummy weights saved to {output_path}")
        return

    try:
        print(f"📂 Loading {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        numpy_weights = {}
        for key, tensor in state_dict.items():
            # Clean key names (remove 'model.' prefix)
            clean_key = key.replace('model.', '')
            
            # Convert to numpy
            numpy_weights[clean_key] = tensor.numpy()
            print(f"   Exporting {clean_key} {tensor.shape}")
            
        # Save compressed npz
        np.savez_compressed(output_path, **numpy_weights)
        print(f"✅ Conversion Complete! Saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Conversion Failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Torch Checkpoint to Numpy")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_ppo.ckpt", help="Input Checkpoint path")
    parser.add_argument("--out", type=str, default="checkpoints/best_ppo_light.npz", help="Output .npz path")
    
    args = parser.parse_args()
    
    convert_ckpt_to_npz(args.ckpt, args.out)
