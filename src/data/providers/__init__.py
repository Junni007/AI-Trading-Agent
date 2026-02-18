"""
Data Providers Package.
Provides unified interface for market data from various sources.
"""
from .base import DataProvider
from .yfinance_provider import YFinanceProvider

# Try to import Alpaca (optional dependency)
try:
    from .alpaca import AlpacaProvider
    ALPACA_AVAILABLE = True
except ImportError:
    AlpacaProvider = None
    ALPACA_AVAILABLE = False

__all__ = ['DataProvider', 'YFinanceProvider', 'AlpacaProvider', 'ALPACA_AVAILABLE']
