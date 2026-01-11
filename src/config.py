from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Sniper Trading Agent"
    VERSION: str = "1.0.0"
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
    
    class Config:
        env_file = ".env"

settings = Settings()
