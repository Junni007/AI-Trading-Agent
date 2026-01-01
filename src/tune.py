import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
import logging
from src.data_loader import MVPDataLoader
from src.lstm_model import LSTMPredictor
from src.ticker_utils import get_extended_tickers

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tuner")

def objective(trial):
    # 1. Hyperparameters to Tune
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # 2. Data Loading (Using Global Variables to save time)
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # Debug: num_workers=0 to prevent deadlocks, enable_progress_bar=True to see movement
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    
    # 3. Model
    model = LSTMPredictor(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        output_dim=3, 
        dropout=dropout, 
        lr=learning_rate
    )
    
    # 4. Trainer with Pruning
    from optuna.integration import PyTorchLightningPruningCallback
    
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=10, # Short epochs for tuning
        accelerator="auto",
        devices=1, 
        enable_checkpointing=False,
        callbacks=[early_stop, pruning_callback],
        logger=False,
        enable_progress_bar=True # Show progress to avoid "Stuck" feeling
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # Return metric
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss

if __name__ == "__main__":
    # Global Data Load (Run once)
    print("Loading Data for Tuning...")
    # Use a moderate number of tickers for tuning speed (e.g. S&P 100)
    TICKERS = get_extended_tickers(limit=100) 
    loader = MVPDataLoader(tickers=TICKERS, window_size=50)
    splits = loader.get_data_splits()
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    input_dim = X_train.shape[2]
    
    print(f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Optimization
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20, timeout=None) # 20 Trials
    
    print("\n" + "="*40)
    print("🏆 Best Trial:")
    print(study.best_trial.params)
    print("="*40 + "\n")
    
    # Save best params
    import json
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f)
