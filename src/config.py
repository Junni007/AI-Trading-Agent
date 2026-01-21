from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Signal.Engine"
    VERSION: str = "2.5.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    SRC_DIR: Path = BASE_DIR / "src"
    
    # Files
    STATE_FILE: Path = BASE_DIR / "simulation_state.json"
    NIFTY_DATA_PATH: Path = SRC_DIR / "nifty500.csv"
    
    # Simulation Defaults
    INITIAL_BALANCE: float = 10000.0
    
    # Data Provider Configuration
    DATA_PROVIDER: Literal["alpaca", "yfinance"] = "yfinance"  # Default to yfinance for compatibility
    
    # Alpaca API (optional - for real-time data)
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_PAPER: bool = True  # Use paper trading endpoint
    
    class Config:
        env_file = ".env"

settings = Settings()

