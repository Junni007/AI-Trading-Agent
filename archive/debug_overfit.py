import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from src.lstm_model import LSTMPredictor

def test_overfit_single_batch():
    print("running overfit test...")
    # 1. Create Synthetic Data (Stationary-ish with clear patterns)
    # Batch=32, SeqDiff=50, Features=4
    # We force specific patterns for classes
    X = torch.randn(32, 50, 4)
    y = torch.randint(0, 3, (32,))
    
    # Make features predictive: 
    # If Class 0, make feature 0 negative
    # If Class 2, make feature 0 positive
    for i in range(32):
        if y[i] == 0:
            X[i, :, 0] = -1.0 # Strong Down signal
        elif y[i] == 2:
            X[i, :, 0] = 1.0  # Strong Up signal
        else:
            X[i, :, 0] = 0.0
            
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    
    # 2. Model
    model = LSTMPredictor(input_dim=4, hidden_dim=64, num_layers=2, lr=0.01)
    
    # 3. Train on SAME batch repeatedly
    trainer = pl.Trainer(max_epochs=50, overfit_batches=1, accelerator="cpu", enable_checkpointing=False, logger=False)
    trainer.fit(model, loader, val_dataloaders=loader)
    
    # 4. Check accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
    print(f"\nOverfit Accuracy: {acc:.4f}")
    if acc > 0.9:
        print("✅ Model logic is sound (can learn signal). Problem is likely data noise/normalization.")
    else:
        print("❌ Model failed to overfit simple signal. Architecture/Optimization is broken.")

if __name__ == "__main__":
    test_overfit_single_batch()
