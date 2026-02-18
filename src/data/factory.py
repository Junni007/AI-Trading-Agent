"""
Data Provider Factory.
Returns the configured data provider based on settings.
"""
import logging
from typing import Union, Optional

from src.config import settings
from src.data.providers import DataProvider, YFinanceProvider, ALPACA_AVAILABLE

logger = logging.getLogger(__name__)

def get_data_provider() -> DataProvider:
    """
    Get the configured data provider.
    
    Uses DATA_PROVIDER setting to determine which provider to use.
    Falls back to yfinance if Alpaca is not available.
    """
    # Safely get provider name
    raw_provider = getattr(settings, "DATA_PROVIDER", None)
    provider_name = raw_provider.lower() if raw_provider else "yfinance"
    
    if provider_name == "alpaca":
        if not ALPACA_AVAILABLE:
            logger.warning("Alpaca requested but alpaca-py not installed. Falling back to yfinance.")
            return YFinanceProvider()
        
        try:
            from src.data.providers import AlpacaProvider
            return AlpacaProvider(
                api_key=settings.ALPACA_API_KEY or None,
                secret_key=settings.ALPACA_SECRET_KEY or None,
                paper=settings.ALPACA_PAPER
            )
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}. Falling back to yfinance.")
            return YFinanceProvider()
    
    # Default: yfinance
    return YFinanceProvider()


# Singleton instance
_provider: Optional[DataProvider] = None

def get_provider() -> DataProvider:
    """Get or create singleton provider instance."""
    global _provider
    if _provider is None:
        _provider = get_data_provider()
        logger.info(f"Initialized data provider: {type(_provider).__name__}")
    return _provider
