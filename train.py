import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import logging
import os
import json

from src.data_loader import MVPDataLoader
from src.lstm_model import LSTMPredictor
from src.ticker_utils import get_extended_tickers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainMassive")

def main():
    # 0. Configuration
    # Auto-scale batch size if GPU is available (Colab T4 has ~15GB VRAM usually)
    if torch.cuda.is_available():
        BATCH_SIZE = 64 # Larger batch for better stability
        ACCELERATOR = "gpu"
        PRECISION = "16-mixed" # Faster training on T4
    else:
        BATCH_SIZE = 32
        ACCELERATOR = "cpu"
        PRECISION = "32-true"

    MAX_EPOCHS = 50
    
    # SCALING UP: 500+ Tickers (S&P 500 + Nifty 50)
    # Caution: 5000 is risky on free Colab RAM/Time. 
    # We default to ~550 (S&P + Nifty) which is "Massive" compared to 1.
    TICKERS = get_extended_tickers(limit=None) 
    
    # 1. Data Loading
    logger.info(f"Loading Massive Data for {len(TICKERS)} tickers...")
    loader = MVPDataLoader(tickers=TICKERS, window_size=50)
    splits = loader.get_data_splits()
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # High-Performance Loader (num_workers=0 to prevent deadlocks in Colab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, persistent_workers=False)
    
    # 2. Model Setup (Deep LSTM)
    input_dim = X_train.shape[2]
    
    # Load Best Hyperparameters if available
    hp_file = "best_hyperparameters.json"
    if os.path.exists(hp_file):
        logger.info(f"Loading optimized hyperparameters from {hp_file}")
        with open(hp_file, "r") as f:
            params = json.load(f)
            hidden_dim = params.get("hidden_dim", 256)
            num_layers = params.get("num_layers", 3)
            dropout = params.get("dropout", 0.2)
            lr = params.get("lr", 0.001)
            # Note: Batch Size optimization requires restart of loader, we skip for now or set earlier.
    else:
        logger.info("Using default hyperparameters (Deep LSTM)")
        hidden_dim, num_layers, dropout, lr = 256, 3, 0.2, 0.001

    model = LSTMPredictor(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        output_dim=3,
        dropout=dropout,
        lr=lr
    )
    
    # 3. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_mvp",
        filename="lstm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10, # Longer patience
        verbose=True,
        mode="min"
    )
    
    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=ACCELERATOR,
        devices=1,
        precision=PRECISION, # Mixed precision
        enable_progress_bar=True,
        log_every_n_steps=5
    )
    
    # 5. Train
    logger.info("Starting Training...")
    trainer.fit(model, train_loader, val_loader)
    
    logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
    
    # 6. Evaluation (The missing piece on local disk)
    logger.info("Loading Best Model for Evaluation...")
    best_model = LSTMPredictor.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    
    # Run Test Evaluation
    evaluate_model(best_model, test_loader, device)

    # 7. Save Final
    torch.save(best_model.state_dict(), "final_lstm_model.pth")
    logger.info("Model saved to final_lstm_model.pth")

def evaluate_model(model, dataloader, device):
    """Runs evaluation on the test set and prints metrics."""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            
    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    print("\n" + "="*40)
    print(f"📊 FINAL TEST RESULTS (2024 Data)")
    print("="*40)
    print(f"✅ Accuracy: {acc:.4f}\n")
    
    print("📉 Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=["DOWN", "NEUTRAL", "UP"]))
    
    print("\nHz Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
