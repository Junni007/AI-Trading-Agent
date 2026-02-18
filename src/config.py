from pydantic import Field, AliasChoices
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
    DATA_PROVIDER: Literal["alpaca", "yfinance"] = "alpaca"
    
    # Alpaca API (Loaded from .env file or environment variables)
    # Supports both our internal naming (ALPACA_*) and standard SDK naming (APCA_*)
    ALPACA_API_KEY: str = Field(default="", validation_alias=AliasChoices('ALPACA_API_KEY', 'APCA_API_KEY_ID'))
    ALPACA_SECRET_KEY: str = Field(default="", validation_alias=AliasChoices('ALPACA_SECRET_KEY', 'APCA_API_SECRET_KEY'))
    ALPACA_PAPER: bool = True  # Use paper trading endpoint
    TDA_ENABLED: bool = False # Disable TDA features for lightweight mode
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

