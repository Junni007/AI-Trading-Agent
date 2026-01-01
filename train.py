import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import logging
import os

from src.data_loader import MVPDataLoader
from src.lstm_model import LSTMPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainMVP")

def main():
    # 0. Configuration
    BATCH_SIZE = 32
    MAX_EPOCHS = 50
    TICKER = "AAPL"
    
    # 1. Data Loading
    logger.info("Loading Data...")
    loader = MVPDataLoader(ticker=TICKER, window_size=50)
    splits = loader.get_data_splits()
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffle ok for training batches in generic ML, but strictly sequence is already built.
    # Note: Shuffling batches does not break time series order within the sample (window). 
    # It removes correlation between batches which is usually good.
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 2. Model Setup
    input_dim = X_train.shape[2]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3)
    
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
        patience=8,
        verbose=True,
        mode="min"
    )
    
    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto", # Should detect GPU/CPU
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=5
    )
    
    # 5. Train
    logger.info("Starting Training...")
    trainer.fit(model, train_loader, val_loader)
    
    logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
    
    # 6. Save TorchScript/Final for inference
    # Load best
    best_model = LSTMPredictor.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    
    # Trace for faster inference/portability
    # example_input = torch.rand(1, 50, input_dim)
    # traced = torch.jit.trace(best_model, example_input)
    # torch.jit.save(traced, "final_lstm_model.pt")
    
    # Just save weights for now
    torch.save(best_model.state_dict(), "final_lstm_model.pth")
    logger.info("Model saved to final_lstm_model.pth")

if __name__ == "__main__":
    main()
