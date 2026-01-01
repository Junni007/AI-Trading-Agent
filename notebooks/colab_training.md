# Student MVP Trading Agent - Colab Training

Run this notebook in Google Colab to train the LSTM model using free GPU resources.

## 1. Setup

```python
# Clone your repo (replace with your actual URL or upload files manually)
!git clone https://github.com/your-username/trading-agent.git
%cd trading-agent

# Install Dependencies
!pip install -r requirements.txt
```

## 2. Verify GPU

```python
import torch
print("GPU Available:", torch.cuda.is_available())
!nvidia-smi
```

## 3. Train Model

```python
# Run the training script
!python train.py
```

## 4. Download Results

```python
from google.colab import files

# Download the best model
files.download('final_lstm_model.pth')
files.download('checkpoints_mvp/') # You might need to zip this folder first
```

## Tips
- Use **Runtime > Change runtime type > T4 GPU** for faster training.
- If data download fails (yfinance rate limits), upload your local data CSVs to Colab.
