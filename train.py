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
    
    # 2. Model Setup
    input_dim = X_train.shape[2]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=3) # Increased hidden/layers for "Best Model"
    
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
