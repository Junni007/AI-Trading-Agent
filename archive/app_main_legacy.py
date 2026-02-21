from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import os
from src.data_loader import MVPDataLoader
from src.lstm_model import LSTMPredictor

app = FastAPI(title="AI Trading Agent API")

# CORS for local dev (React port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model (Global)
MODEL_PATH = "final_lstm_model.pth"
model = None
loader = None

@app.on_event("startup")
async def startup_event():
    global model, loader
    # Initialize Loader
    loader = MVPDataLoader(ticker="AAPL")
    
    # Load Model if exists
    if os.path.exists(MODEL_PATH):
        # We need input_dim from data to init model structure ??
        # Or hardcode/save config. 
        # For MVP, assume structure matches training default (feature size from loader)
        # Let's peek at data to get dim
        df = loader.fetch_data()
        df = loader.feature_engineering(df)
        input_dim = 4 # Close, RSI, MACD, Log_Return (after scaling) - Scaler dim check?
        # Ideally save hyperparams. 
        # Using feature list length from data_loader.create_sequences -> 4.
        
        model = LSTMPredictor(input_dim=4, output_dim=3)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Model loaded.")
    else:
        print("Warning: Model not found. /prediction will fail.")

class PredictionResponse(BaseModel):
    date: str
    ticker: str
    prediction: str # UP, DOWN, NEUTRAL
    confidence: float
    features: dict

@app.get("/prediction", response_model=PredictionResponse)
async def get_prediction():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
        
    # Fetch latest data
    df = loader.fetch_data()
    df_eng = loader.feature_engineering(df)
    
    # Get last window
    # Note: Using LAST available window to predict TOMORROW/NEXT DAY.
    # Scaler fitting? Need the saved scalers from training!
    # Issue: MVPDataLoader fits scaler on Train split call.
    # Fix: We need to load scalers. Ideally pickle them.
    # For MVP Hack: Fit scaler on the WHOLE history just for inference? 
    # Or strict: Re-fit on 2019-2022 subset inside loader.
    # Let's assume loader logic handles the fit if we call the right method.
    # We will trigger `get_data_splits` ONCE at startup to fit scalers properly.
    
    # Quick fix in startup:
    loader.get_data_splits() # Fits scalers
    
    # Prepare sequence
    # Taking the very last 50 data points
    # Need to normalize using the fitted scaler
    last_window = df_eng.iloc[-loader.window_size:]
    features = last_window[['Close', 'RSI', 'MACD', 'Log_Return']].values
    
    scaler = loader.scalers['features']
    features_scaled = scaler.transform(features)
    
    # Predict
    with torch.no_grad():
        x = torch.FloatTensor(features_scaled).unsqueeze(0) # (1, 50, 4)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()
        
    mapping = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
    
    return {
        "date": str(df_eng.index[-1].date()),
        "ticker": loader.ticker,
        "prediction": mapping[pred_idx],
        "confidence": round(confidence * 100, 2),
        "features": {
            "Close": float(last_window['Close'].iloc[-1]),
            "RSI": float(last_window['RSI'].iloc[-1])
        }
    }

@app.get("/backtest")
async def run_backtest_endpoint():
    # Trigger the backtest script logic essentially
    # For simplicity, assume backtest.py saves chart to frontend/public
    # We just return the stats
    from src.backtest import run_backtest
    try:
        run_backtest()
        return {"status": "success", "message": "Backtest complete. Refresh dashboard."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# Serve React App (Build) - For MVP we might run them separately (FastAPI:8000, React:3000)
# But strictly, serving from one is nicer.
# If frontend/build exists, mount it.
if os.path.isdir("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
