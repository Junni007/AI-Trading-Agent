"""
Data package for Signal.Engine.
Contains data providers and loaders.
"""
from .providers import DataProvider, YFinanceProvider, ALPACA_AVAILABLE

try:
    from .providers import AlpacaProvider
except ImportError:
    AlpacaProvider = None

__all__ = ['DataProvider', 'YFinanceProvider', 'AlpacaProvider', 'ALPACA_AVAILABLE']
