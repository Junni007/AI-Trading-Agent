import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import logging
import os

from src.sft_dataset import GoldenDataset
from src.train_ppo_optimized import RecurrentActorCritic
from src.ticker_utils import get_nifty500_tickers

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SFTTrainer')

class SFTLightningModule(pl.LightningModule):
    """
    Lightning Module for Supervised Fine-Tuning.
    """
    def __init__(self, model, class_weights=None, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        
        # Weighted Cross Entropy Loss for Imbalanced Classes
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # x: (Batch, Seq_Len, Features)
        # y: (Batch)
        x, y = batch
        
        # Forward pass
        # RecurrentActorCritic returns: action_probs, state_value, hidden
        action_probs, _, _ = self.model(x)
        
        # Loss Calculation
        # CrossEntropyLoss expects logits usually, but we have probs.
        # Ideally we should use logits for numerical stability, but RecurrentActorCritic output is softmaxed.
        # Workaround: Log(Probs) -> NLLLoss (equivalent to CrossEntropy on logits)
        # Or change model to return logits. 
        # For now: torch.log(probs + epsilon)
        
        log_probs = torch.log(action_probs + 1e-8)
        loss = nn.NLLLoss(weight=self.criterion.weight)(log_probs, y)
        
        # Logging
        preds = torch.argmax(action_probs, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 5  # Start small
    LR = 1e-3
    NUM_TICKERS = 20 # Train on 20 tickers first
    
    # 1. Prepare Data
    logger.info("Loading SFT Dataset...")
    tickers = get_nifty500_tickers()[:NUM_TICKERS]
    dataset = GoldenDataset(tickers)
    
    if len(dataset) == 0:
        logger.error("Dataset is empty! Exiting.")
        return

    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0 # Windows
    )
    
    # 2. Initialize Model
    # Input Dim = 7 (5 Features + Position + Balance)
    # Output Dim = 3 (Hold, Buy, Sell)
    model = RecurrentActorCritic(input_dim=7, output_dim=3, hidden_dim=256)
    
    # 3. Initialize Trainer
    sft_module = SFTLightningModule(model, class_weights=dataset.class_weights, lr=LR)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_sft',
        filename='sft-{epoch:02d}-{train_acc:.2f}',
        save_top_k=1,
        monitor='train_acc',
        mode='max'
    )
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=CSVLogger("logs", name="sft_training")
    )
    
    # 4. Train
    logger.info("Starting SFT Training...")
    trainer.fit(sft_module, train_loader)
    
    # 5. Save Final Weights
    save_path = "checkpoints_sft/final_sft_model.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved SFT model to {save_path}")

if __name__ == "__main__":
    main()
